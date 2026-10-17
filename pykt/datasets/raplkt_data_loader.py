#!/usr/bin/env python
# coding=utf-8

import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from .multikc_protocol import (
    build_raw_q2cs_map,
    normalize_raw_q2cs_map,
    q2cs_mapping_signature,
)


class RAPLKTDataset(Dataset):
    """
    RAPLKT dataset.

    与原 KTDataset 的主要区别：
    1. 原 KTDataset 读取 cseqs 后仍是单 KC 序列；
    2. 这里根据 is_repeat 把同一道题的多个 KC 合并成 cseqs_multi；
    3. 模型输入从 [B, S] 的 concept id 变成 [B, S, max_concepts] 的 candidate KC set。
    """

    def __init__(
            self,
            file_path,
            input_type,
            folds,
            max_concepts=8,
            qtest=False,
            item_window=False,
            window_size=200,
            sequence_mode="collapse",
            q2cs_map=None,
    ):
        super().__init__()

        self.file_path = file_path
        self.input_type = input_type
        self.folds = sorted(list(folds))
        if max_concepts < 1:
            raise ValueError(f"max_concepts must be >= 1, got: {max_concepts}.")
        if item_window and window_size < 2:
            raise ValueError(
                f"window_size must be >= 2 when item_window is enabled, got: {window_size}."
            )
        self.max_concepts = max_concepts
        self.qtest = qtest

        if sequence_mode not in {"collapse", "preserve"}:
            raise ValueError(
                f"sequence_mode must be 'collapse' or 'preserve', got: {sequence_mode}"
            )
        self.sequence_mode = sequence_mode

        self.item_window = item_window
        self.window_size = window_size

        self.id_offset = 1
        self.raw_q2cs_map = normalize_raw_q2cs_map(q2cs_map, self.max_concepts)
        mapping_signature = q2cs_mapping_signature(
            self.raw_q2cs_map,
            self.max_concepts,
        )
        # RAPLKT reserves 0 for padding; the shared benchmark mapping stays raw.
        self.q2cs_map = {
            question + 1: [concept + 1 for concept in concepts]
            for question, concepts in self.raw_q2cs_map.items()
        }

        folds_str = "_" + "_".join([str(_) for _ in self.folds])

        if self.sequence_mode == "preserve":
            processed_data = (
                    file_path
                    + folds_str
                    + f"_raplkt_{self.sequence_mode}_{mapping_signature}_v7_k{max_concepts}.pkl"
            )
        elif item_window:
            processed_data = (
                    file_path
                    + folds_str
                    + f"_raplkt_{self.sequence_mode}_itemwin{window_size}_{mapping_signature}_v7_k{max_concepts}.pkl"
            )
        else:
            processed_data = (
                    file_path
                    + folds_str
                    + f"_raplkt_{self.sequence_mode}_{mapping_signature}_v7_k{max_concepts}.pkl"
            )
        self.processed_data = processed_data

        if not os.path.exists(processed_data):
            print(f"Start preprocessing RAPLKT data: {file_path}, folds: {folds_str}")
            self.dori = self.__load_data__(file_path, self.folds)
            pd.to_pickle(self.dori, processed_data)
        else:
            print(f"Read RAPLKT data from processed file: {processed_data}")
            self.dori = pd.read_pickle(processed_data)

        print(
            f"file path: {file_path}, "
            f"qlen: {len(self.dori['qseqs'])}, "
            f"clen: {len(self.dori['cseqs'])}, "
            f"rlen: {len(self.dori['rseqs'])}, "
            f"sequence_mode: {self.sequence_mode}, "
            f"max_concepts: {self.max_concepts}"
        )

    def __len__(self):
        return len(self.dori["rseqs"])

    @staticmethod
    def _parse_int_list(value):
        if pd.isna(value):
            return []
        return [int(float(x)) for x in str(value).split(",") if str(x) != ""]

    @classmethod
    def build_q2cs_map(cls, file_path, folds=None, pad_val=-1):
        """Compatibility wrapper returning the shared raw 0-based mapping."""
        return build_raw_q2cs_map(file_path, folds=folds, pad_val=pad_val)

    def _group_multi_concept_row_preserve_time(self, row, pad_val=-1):
        """
        保留 pyKT 原始 concept-expanded 时间轴，同时为每个位置构造 multi-KC set。

        目标：
        1. 时间步数量与 AKT 一致；
        2. repeat KC 位置保留为历史输入；
        3. 只在 is_repeat=0 的 question-level 位置评价；
        4. 每个 question 起点位置使用完整 multi-KC set；
        5. repeat 位置也保留，但不参与评价。
        """
        qs = self._parse_int_list(row["questions"])
        cs = self._parse_int_list(row["concepts"])
        rs = self._parse_int_list(row["responses"])
        sms = self._parse_int_list(row["selectmasks"])
        cidxs = (
            self._parse_int_list(row["cidxs"])
            if "cidxs" in row.index and not pd.isna(row["cidxs"])
            else [-1 for _ in cs]
        )

        has_is_repeat = "is_repeat" in row.index and not pd.isna(row["is_repeat"])
        if has_is_repeat:
            reps = self._parse_int_list(row["is_repeat"])
        else:
            reps = [0 for _ in cs]

        n = min(len(qs), len(cs), len(rs), len(sms), len(reps), len(cidxs))

        out_qs = []
        out_rs = []
        out_sms = []
        out_cs_multi = []
        out_evidence_updates = []
        out_cidxs = []

        i = 0
        while i < n:
            q = qs[i]
            c = cs[i]
            r = rs[i]

            if q == pad_val or c == pad_val or r == pad_val:
                i += 1
                continue

            # 找到当前 contiguous question block。
            # 正常情况：rep=0 开始，后面若干 rep=1。
            # window 从中间开始时，也允许 rep=1 开始形成 partial block。
            block_start = i
            block_q = qs[i]
            block_r = rs[i]

            j = i + 1
            while j < n:
                if qs[j] == pad_val or cs[j] == pad_val or rs[j] == pad_val:
                    break

                same_question = qs[j] == block_q and rs[j] == block_r

                if has_is_repeat:
                    # 同一道题的 repeat KC 一般 rep=1。
                    # 如果 q/r 相同且 rep=1，就视为同一个 question block。
                    if same_question and reps[j] == 1:
                        j += 1
                        continue
                    else:
                        break
                else:
                    # 没有 is_repeat 时，只能按 q/r 连续相同做保守合并。
                    if same_question:
                        j += 1
                        continue
                    else:
                        break

            visible_kc_set = []
            for t in range(block_start, j):
                if cs[t] != pad_val:
                    kc_id = cs[t] + 1
                    if kc_id not in visible_kc_set:
                        visible_kc_set.append(kc_id)

            if len(visible_kc_set) == 0:
                i = j
                continue

            q_key = block_q + 1
            mapped_kc_set = self.q2cs_map.get(q_key, [])

            if q_key in self.q2cs_map and len(self.q2cs_map[q_key]) > 0:
                full_kc_set = self.q2cs_map[q_key][:self.max_concepts]
            else:
                full_kc_set = visible_kc_set[:self.max_concepts]

            if len(full_kc_set) == 0:
                i = j
                continue

            # 对 block 内每个原始 concept-expanded 位置都保留一个时间步。
            for t in range(block_start, j):
                q_t = qs[t]
                c_t = cs[t]
                r_t = rs[t]
                sm_t = sms[t]
                rep_t = reps[t]

                if q_t == pad_val or c_t == pad_val or r_t == pad_val:
                    continue

                q_t = q_t + 1
                c_t = c_t + 1

                out_qs.append(q_t)
                out_rs.append(r_t)
                out_cidxs.append(cidxs[t])
                # A concept-expanded question contributes evidence exactly once.
                # The first visible position also covers a window that starts in
                # the middle of a multi-KC question.
                out_evidence_updates.append(t == block_start)

                if has_is_repeat and rep_t == 1:
                    # repeat KC 位置只作为历史，不参与评价，只放当前 KC。
                    out_sms.append(-1)
                    out_cs_multi.append([c_t])
                else:
                    out_sms.append(sm_t)

                    # 关键：
                    # 只有真正的 question-level target 位置才用 q2cs_map 补全。
                    # 普通历史位置仍使用当前 window/sequence 可见 KC。
                    if sm_t != -1 and len(mapped_kc_set) > 0:
                        out_cs_multi.append(mapped_kc_set[:self.max_concepts])
                    else:
                        out_cs_multi.append(visible_kc_set[:self.max_concepts])

            i = j

        return (
            out_qs,
            out_cs_multi,
            out_rs,
            out_sms,
            out_evidence_updates,
            out_cidxs,
        )

    def _group_multi_concept_row(self, row, pad_val=-1):
        """
        将 pyKT 展开的多 KC 序列还原为 question-level multi-KC 序列。

        评价协议：
        - 每道 question 只形成一个 item-level 预测目标；
        - 同一道题的多个 KC 合并到同一个 candidate KC set；
        - is_repeat=1 的重复 KC 不单独作为评价目标；
        - 如果窗口从多 KC 题中间开始，保留该 partial item 作为历史，
          但将它的 selectmask 设为 -1，避免把残缺 item 计入 AUC/ACC。
        """
        qs = self._parse_int_list(row["questions"])
        cs = self._parse_int_list(row["concepts"])
        rs = self._parse_int_list(row["responses"])
        sms = self._parse_int_list(row["selectmasks"])
        cidxs = (
            self._parse_int_list(row["cidxs"])
            if "cidxs" in row.index and not pd.isna(row["cidxs"])
            else [-1 for _ in cs]
        )

        has_is_repeat = "is_repeat" in row.index and not pd.isna(row["is_repeat"])
        if has_is_repeat:
            reps = self._parse_int_list(row["is_repeat"])
        else:
            reps = [0 for _ in cs]

        n = min(len(qs), len(cs), len(rs), len(sms), len(reps), len(cidxs))

        grouped_qs = []
        grouped_rs = []
        grouped_sms = []
        grouped_cs = []
        grouped_evidence_updates = []
        grouped_cidxs = []

        last_q = None
        last_r = None

        for i in range(n):
            q = qs[i]
            c = cs[i]
            r = rs[i]
            sm = sms[i]
            rep = reps[i]

            if q == pad_val or c == pad_val or r == pad_val:
                continue

            # 有效 id 统一 +1，保留 0 作为 padding。
            q = q + 1
            c = c + 1

            if has_is_repeat and rep == 1:
                # 情况 1：当前 repeat KC 确实属于上一道 question
                if (
                        len(grouped_cs) > 0
                        and last_q == q
                        and last_r == r
                ):
                    # The rep=0 target may already have been expanded from the
                    # complete q2cs map. Do not append its visible repeat KC a
                    # second time, otherwise a true 2-KC item can be counted as
                    # K>=3 and the duplicate also reaches the model input.
                    if (
                            c not in grouped_cs[-1]
                            and len(grouped_cs[-1]) < self.max_concepts
                    ):
                        grouped_cs[-1].append(c)

                # 情况 2：窗口从多 KC 题中间开始
                # 不能把它当作可评价 target，但应保留为历史。
                else:
                    grouped_qs.append(q)
                    grouped_rs.append(r)

                    # 关键：orphan repeat 只作为历史，不作为 question-level target。
                    grouped_sms.append(-1)

                    grouped_cs.append([c])
                    grouped_evidence_updates.append(1)
                    grouped_cidxs.append(cidxs[i])

                    last_q = q
                    last_r = r

                continue

            # rep == 0 或没有 is_repeat 字段：开启一个完整 question-level item。
            grouped_qs.append(q)
            grouped_rs.append(r)
            grouped_sms.append(sm)
            grouped_evidence_updates.append(1)
            # The rep=0 expanded position is the benchmark item identity.
            grouped_cidxs.append(cidxs[i])
            mapped_kc_set = self.q2cs_map.get(q, [])
            if len(mapped_kc_set) > 0:
                grouped_cs.append(mapped_kc_set[:self.max_concepts])
            else:
                grouped_cs.append([c])

            last_q = q
            last_r = r

        return (
            grouped_qs,
            grouped_cs,
            grouped_rs,
            grouped_sms,
            grouped_evidence_updates,
            grouped_cidxs,
        )

    def _pad_1d(self, seq, max_len, pad_val=0):
        seq = seq[:max_len]
        return seq + [pad_val] * (max_len - len(seq))

    def _pad_multi_concepts(self, seq, max_len):
        """
        seq: list[list[int]]
        return:
            c_multi: [max_len, max_concepts]
            c_mask:  [max_len, max_concepts]
        """
        c_multi = []
        c_mask = []

        seq = seq[:max_len]

        for cs in seq:
            cs = cs[:self.max_concepts]
            mask = [1] * len(cs)

            cs = cs + [0] * (self.max_concepts - len(cs))
            mask = mask + [0] * (self.max_concepts - len(mask))

            c_multi.append(cs)
            c_mask.append(mask)

        while len(c_multi) < max_len:
            c_multi.append([0] * self.max_concepts)
            c_mask.append([0] * self.max_concepts)

        return c_multi, c_mask

    def _make_item_windows(self, raw_qs, raw_cs_multi, raw_rs, raw_cidxs):
        """
        从 item-level multi-KC 序列构造 one-step windows。

        每个 window 的最后一个 item 是预测目标；
        前面的 item 是历史。
        """
        win_qs = []
        win_cs_multi = []
        win_rs = []
        win_sms = []
        win_evidence_updates = []
        win_cidxs = []

        for qs, cs_multi, rs, cidxs in zip(raw_qs, raw_cs_multi, raw_rs, raw_cidxs):
            n = len(qs)

            # 至少需要一个历史 item 和一个目标 item
            if n < 2:
                continue

            for t in range(1, n):
                start = max(0, t - self.window_size + 1)

                w_qs = qs[start:t + 1]
                w_cs = cs_multi[start:t + 1]
                w_rs = rs[start:t + 1]
                w_cidxs = cidxs[start:t + 1]

                # 只评估 window 的最后一个目标 item
                w_sms = [-1] * len(w_qs)
                w_sms[-1] = 1

                win_qs.append(w_qs)
                win_cs_multi.append(w_cs)
                win_rs.append(w_rs)
                win_sms.append(w_sms)
                win_evidence_updates.append([1] * len(w_qs))
                win_cidxs.append(w_cidxs)

        return (
            win_qs,
            win_cs_multi,
            win_rs,
            win_sms,
            win_evidence_updates,
            win_cidxs,
        )

    def __load_data__(self, sequence_path, folds):
        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)]

        raw_qs = []
        raw_cs_first = []
        raw_cs_multi = []
        raw_cs_mask = []
        raw_rs = []
        raw_sms = []
        raw_evidence_updates = []
        raw_cidxs = []

        max_len = 0

        for _, row in df.iterrows():
            if self.sequence_mode == "preserve":
                qs, cs_multi, rs, sms, evidence_updates, cidxs = (
                    self._group_multi_concept_row_preserve_time(row)
                )
            else:
                qs, cs_multi, rs, sms, evidence_updates, cidxs = (
                    self._group_multi_concept_row(row)
                )

            if len(qs) == 0:
                continue

            max_len = max(max_len, len(qs))

            raw_qs.append(qs)
            raw_cs_first.append([cs[0] if len(cs) > 0 else 0 for cs in cs_multi])
            raw_cs_multi.append(cs_multi)
            raw_rs.append(rs)
            raw_sms.append(sms)
            raw_evidence_updates.append(evidence_updates)
            raw_cidxs.append(cidxs)

        if self.item_window:
            (
                raw_qs,
                raw_cs_multi,
                raw_rs,
                raw_sms,
                raw_evidence_updates,
                raw_cidxs,
            ) = self._make_item_windows(
                raw_qs,
                raw_cs_multi,
                raw_rs,
                raw_cidxs,
            )

            raw_cs_first = [
                [cs[0] if len(cs) > 0 else 0 for cs in cs_multi]
                for cs_multi in raw_cs_multi
            ]

            if len(raw_qs) == 0:
                raise ValueError(
                    f"No item-level windows were created from {sequence_path}. "
                    f"Check window_size={self.window_size} and input data."
                )

            max_len = max(len(qs) for qs in raw_qs)

        padded_qs = []
        padded_cs_first = []
        padded_cs_multi = []
        padded_cs_mask = []
        padded_rs = []
        padded_sms = []
        padded_evidence_updates = []
        padded_cidxs = []

        for qs, cs_first, cs_multi, rs, sms, evidence_updates, cidxs in zip(
            raw_qs,
            raw_cs_first,
            raw_cs_multi,
            raw_rs,
            raw_sms,
            raw_evidence_updates,
            raw_cidxs,
        ):
            padded_qs.append(self._pad_1d(qs, max_len, 0))
            padded_cs_first.append(self._pad_1d(cs_first, max_len, 0))
            padded_rs.append(self._pad_1d(rs, max_len, 0))

            # selectmask padding 保持 -1，便于和原 pyKT 逻辑一致。
            padded_sms.append(self._pad_1d(sms, max_len, -1))
            padded_evidence_updates.append(
                self._pad_1d(evidence_updates, max_len, 0)
            )
            padded_cidxs.append(self._pad_1d(cidxs, max_len, -1))

            c_multi, c_mask = self._pad_multi_concepts(cs_multi, max_len)
            padded_cs_multi.append(c_multi)
            padded_cs_mask.append(c_mask)

        dori = {
            "qseqs": torch.LongTensor(padded_qs),
            "cseqs": torch.LongTensor(padded_cs_first),
            "cseqs_multi": torch.LongTensor(padded_cs_multi),
            "cmasks_multi": torch.BoolTensor(padded_cs_mask),
            "rseqs": torch.FloatTensor(padded_rs),
            "evidence_update_seqs": torch.BoolTensor(padded_evidence_updates),
            "cidxs": torch.LongTensor(padded_cidxs),
        }

        valid_pos = dori["cseqs"] != 0
        masks = valid_pos[:, :-1] & valid_pos[:, 1:]

        smasks_tensor = torch.LongTensor(padded_sms)
        smasks = (smasks_tensor[:, 1:] != -1) & masks

        dori["masks"] = masks
        dori["smasks"] = smasks

        print(f"RAPLKT item-level interaction num: {int(smasks.sum().item())}")
        return dori

    def __getitem__(self, index):
        dcur = {}
        mseqs = self.dori["masks"][index]

        for key in ["qseqs", "cseqs", "rseqs"]:
            seqs = self.dori[key][index][:-1]
            shft_seqs = self.dori[key][index][1:]

            if key != "rseqs":
                seqs = seqs * mseqs.long()
                shft_seqs = shft_seqs * mseqs.long()
            else:
                seqs = seqs * mseqs.float()
                shft_seqs = shft_seqs * mseqs.float()

            dcur[key] = seqs
            dcur["shft_" + key] = shft_seqs

        c_multi = self.dori["cseqs_multi"][index]
        c_mask = self.dori["cmasks_multi"][index]

        dcur["cseqs_multi"] = c_multi[:-1] * mseqs.long().unsqueeze(-1)
        dcur["shft_cseqs_multi"] = c_multi[1:] * mseqs.long().unsqueeze(-1)

        dcur["cmasks_multi"] = c_mask[:-1] & mseqs.bool().unsqueeze(-1)
        dcur["shft_cmasks_multi"] = c_mask[1:] & mseqs.bool().unsqueeze(-1)

        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]

        evidence_updates = self.dori["evidence_update_seqs"][index]
        dcur["evidence_update_seqs"] = evidence_updates[:-1] & mseqs.bool()
        dcur["shft_evidence_update_seqs"] = evidence_updates[1:] & mseqs.bool()

        cidxs = self.dori["cidxs"][index]
        dcur["cidxs"] = cidxs[:-1]
        dcur["shft_cidxs"] = cidxs[1:]

        return dcur
