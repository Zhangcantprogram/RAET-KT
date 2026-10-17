#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

from .multikc_protocol import normalize_raw_q2cs_map, q2cs_mapping_signature

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
else:
    from torch import FloatTensor, LongTensor

class KTDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).
    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """
    def __init__(
        self,
        file_path,
        input_type,
        folds,
        qtest=False,
        sequence_mode="original",
        max_concepts=8,
        q2cs_map=None,
    ):
        super(KTDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        if sequence_mode not in {"original", "collapse"}:
            raise ValueError(
                f"sequence_mode must be 'original' or 'collapse', got: {sequence_mode}"
            )
        self.sequence_mode = sequence_mode
        self.max_concepts = max_concepts
        self.id_offset = 0
        self.q2cs_map = normalize_raw_q2cs_map(q2cs_map, self.max_concepts)
        mapping_signature = q2cs_mapping_signature(
            self.q2cs_map,
            self.max_concepts,
        )
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = (
                file_path
                + folds_str
                + f"_{self.sequence_mode}_qtest_{mapping_signature}_v5_k{self.max_concepts}.pkl"
            )
        else:
            processed_data = (
                file_path
                + folds_str
                + f"_{self.sequence_mode}_{mapping_signature}_v5_k{self.max_concepts}.pkl"
            )
        self.processed_data = processed_data

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dqtest = self.__load_data__(sequence_path, folds)
                save_data = [self.dori, self.dqtest]
            else:
                self.dori = self.__load_data__(sequence_path, folds)
                save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori = pd.read_pickle(processed_data)
                for key in self.dori:
                    self.dori[key] = self.dori[key]#[:100]
        print(
            f"file path: {file_path}, "
            f"qlen: {len(self.dori['qseqs'])}, "
            f"clen: {len(self.dori['cseqs'])}, "
            f"rlen: {len(self.dori['rseqs'])}, "
            f"sequence_mode: {self.sequence_mode}, "
            f"max_concepts: {self.max_concepts}"
        )

    def __len__(self):
        """return the dataset length
        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get
        Returns:
            (tuple): tuple containing:
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            key_mask = mseqs.to(self.dori[key].device)
            if self.dori[key][index].dim() == 2:
                seqs = self.dori[key][index][:-1] * key_mask.unsqueeze(-1)
                shft_seqs = self.dori[key][index][1:] * key_mask.unsqueeze(-1)
            else:
                seqs = self.dori[key][index][:-1] * key_mask
                shft_seqs = self.dori[key][index][1:] * key_mask
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        # print("tseqs", dcur["tseqs"])
        if not self.qtest:
            return dcur
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dqtest

    @staticmethod
    def _parse_int_list(value):
        if pd.isna(value):
            return []
        return [int(float(x)) for x in str(value).split(",") if str(x) != ""]

    def _pad_1d(self, seq, max_len, pad_val):
        seq = seq[:max_len]
        return seq + [pad_val] * (max_len - len(seq))

    def _pad_multi_concepts(self, seq, max_len):
        c_multi = []
        c_mask = []

        for cs in seq[:max_len]:
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

    def _group_multi_concept_row(self, row, pad_val=-1):
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
        grouped_cs = []
        grouped_rs = []
        grouped_sms = []
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

            if has_is_repeat and rep == 1:
                if len(grouped_cs) > 0 and last_q == q and last_r == r:
                    if c not in grouped_cs[-1] and len(grouped_cs[-1]) < self.max_concepts:
                        grouped_cs[-1].append(c)
                else:
                    grouped_qs.append(q)
                    grouped_rs.append(r)
                    grouped_sms.append(-1)
                    grouped_cs.append([c])
                    grouped_cidxs.append(cidxs[i])
                    last_q = q
                    last_r = r
                continue

            grouped_qs.append(q)
            grouped_rs.append(r)
            grouped_sms.append(sm)
            grouped_cidxs.append(cidxs[i])
            mapped = self.q2cs_map.get(q, [])
            grouped_cs.append(mapped if mapped else [c])
            last_q = q
            last_r = r

        return grouped_qs, grouped_cs, grouped_rs, grouped_sms, grouped_cidxs

    def __load_collapse_data__(self, sequence_path, folds, pad_val=-1):
        dori = {
            "qseqs": [],
            "cseqs": [],
            "cseqs_multi": [],
            "cmasks_multi": [],
            "rseqs": [],
            "tseqs": [],
            "utseqs": [],
            "smasks": []
        }
        dori["cidxs"] = []

        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)]

        raw_qs = []
        raw_cs_first = []
        raw_cs_multi = []
        raw_rs = []
        raw_sms = []
        raw_cidxs = []
        max_len = 0

        for _, row in df.iterrows():
            qs, cs_multi, rs, sms, cidxs = self._group_multi_concept_row(row, pad_val)
            if len(qs) == 0:
                continue

            max_len = max(max_len, len(qs))
            raw_qs.append(qs)
            raw_cs_multi.append(cs_multi)
            raw_cs_first.append([cs[0] if len(cs) > 0 else pad_val for cs in cs_multi])
            raw_rs.append(rs)
            raw_sms.append(sms)
            raw_cidxs.append(cidxs)

        if len(raw_qs) == 0:
            raise ValueError(
                f"No collapse-mode AKT sequences were created from {sequence_path}."
            )

        for qs, cs_first, cs_multi, rs, sms, cidxs in zip(
            raw_qs,
            raw_cs_first,
            raw_cs_multi,
            raw_rs,
            raw_sms,
            raw_cidxs,
        ):
            dori["qseqs"].append(self._pad_1d(qs, max_len, pad_val))
            dori["cseqs"].append(self._pad_1d(cs_first, max_len, pad_val))
            dori["rseqs"].append(self._pad_1d(rs, max_len, pad_val))
            dori["smasks"].append(self._pad_1d(sms, max_len, pad_val))
            dori["cidxs"].append(self._pad_1d(cidxs, max_len, pad_val))

            c_multi, c_mask = self._pad_multi_concepts(cs_multi, max_len)
            dori["cseqs_multi"].append(c_multi)
            dori["cmasks_multi"].append(c_mask)

        for key in dori:
            if len(dori[key]) == 0:
                continue

            if key == "rseqs":
                dori[key] = FloatTensor(dori[key])
            elif key == "cmasks_multi":
                dori[key] = torch.BoolTensor(dori[key])
            else:
                dori[key] = LongTensor(dori[key])

        mask_seqs = (dori["cseqs"][:, :-1] != pad_val) * (dori["cseqs"][:, 1:] != pad_val)
        dori["masks"] = mask_seqs
        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val) & mask_seqs

        print(f"AKT collapse item-level interaction num: {int(dori['smasks'].sum().item())}")
        return dori

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.
        Returns: 
            (tuple): tuple containing
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        if self.sequence_mode == "collapse":
            if self.qtest:
                raise ValueError("AKT collapse mode does not support qtest=True in this version.")
            return self.__load_collapse_data__(sequence_path, folds, pad_val)

        # dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}
        dori = {
            "qseqs": [],
            "cseqs": [],
            "rseqs": [],
            "tseqs": [],
            "utseqs": [],
            "smasks": []
        }

        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)]

        has_is_repeat = "is_repeat" in df.columns
        if has_is_repeat:
            dori["is_repeat"] = []
        has_cidxs = "cidxs" in df.columns
        if has_cidxs:
            dori["cidxs"] = []

        interaction_num = 0
        dqtest = {"qidxs": [], "rests": [], "orirow": []}

        for i, row in df.iterrows():
            if "concepts" in self.input_type:
                dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])

            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])

            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])

            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])

            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            if has_is_repeat:
                dori["is_repeat"].append([int(_) for _ in row["is_repeat"].split(",")])
            if has_cidxs:
                dori["cidxs"].append([int(_) for _ in row["cidxs"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])

        for key in dori:
            if len(dori[key]) == 0:
                continue

            if key == "rseqs":
                dori[key] = FloatTensor(dori[key])
            else:
                dori[key] = LongTensor(dori[key])

        mask_seqs = (dori["cseqs"][:, :-1] != pad_val) * (dori["cseqs"][:, 1:] != pad_val)
        dori["masks"] = mask_seqs

        # 原始 select mask：排除 padding
        smasks = dori["smasks"][:, 1:] != pad_val

        # 关键修复：排除多知识点展开产生的重复 KC 位置
        if has_is_repeat:
            repeat_masks = dori["is_repeat"][:, 1:] == 0
            smasks = smasks & repeat_masks

        dori["smasks"] = smasks

        print(f"interaction_num before repeat filtering: {interaction_num}")
        print(f"selected interaction_num after repeat filtering: {int(dori['smasks'].sum().item())}")

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]

            return dori, dqtest

        return dori
