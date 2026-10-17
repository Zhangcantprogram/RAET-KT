"""Reproducible Eedi Task 1/2 student subset and canonical pyKT adapter.

Rank users by seeded SHA256 as in the EdNet subset adapter. Use only official
subjects, removing all transitive ancestors of other tags on the same question.
This removes hierarchy redundancy; it does not imply independent skills.
"""

import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from tqdm import tqdm

SEED = 3407
SAMPLE_SIZE = 5000
NAMESPACE = "RAPLKT_EEDI_TASK12_5K_V1"


def integer_id(value):
    try:
        value = Decimal(str(value).strip())
        if value.is_finite() and value >= 0 and value == value.to_integral_value():
            return int(value)
    except (InvalidOperation, ValueError, TypeError):
        pass
    return None


def timestamp_ms(value):
    try:
        date = datetime.fromisoformat(str(value).strip())
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        delta = date.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
        milliseconds = delta.days * 86400000 + delta.seconds * 1000 + delta.microseconds // 1000
        return milliseconds if milliseconds >= 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def selection_hash(uid, seed=SEED):
    return hashlib.sha256(f"{NAMESPACE}|{seed}|{uid}".encode("utf-8")).hexdigest()


def csv_rows(path, required, description):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        yield from tqdm(reader, desc=description, unit="row")


def load_question_to_kcs(raw_root):
    metadata = Path(raw_root) / "metadata"
    parents = {}
    for row in csv_rows(metadata / "subject_metadata.csv", ["SubjectId", "ParentId"], "Loading subject tree"):
        sid = integer_id(row["SubjectId"])
        token = (row["ParentId"] or "").strip()
        parent = None if token.lower() in ("", "null", "nan", "none") else integer_id(token)
        if sid is None or sid in parents or (parent is None and token.lower() not in ("", "null", "nan", "none")):
            raise ValueError(f"Invalid or duplicate official subject: {row}")
        parents[sid] = parent
    ancestors = {}
    for sid in parents:
        chain = set()
        parent = parents[sid]
        while parent is not None:
            if parent == sid or parent in chain:
                raise ValueError(f"Cycle in official subject tree at {sid}")
            if parent not in parents:
                raise ValueError(f"Missing parent subject {parent} for {sid}")
            chain.add(parent)
            parent = parents[parent]
        ancestors[sid] = chain
    mapping, raw_sets = {}, {}
    for row in csv_rows(metadata / "question_metadata_task_1_2.csv", ["QuestionId", "SubjectId"], "Loading question KCs"):
        qid = integer_id(row["QuestionId"])
        try:
            tags = ast.literal_eval(row["SubjectId"])
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Invalid SubjectId list for question {qid}") from exc
        if qid is None or not isinstance(tags, list) or any(isinstance(s, bool) or not isinstance(s, int) or s not in parents for s in tags):
            raise ValueError(f"Invalid official question metadata: {row}")
        raw = frozenset(tags)
        if qid in raw_sets and raw_sets[qid] != raw:
            raise ValueError(f"Conflicting original SubjectId sets for question {qid}")
        raw_sets[qid] = raw
        redundant = set().union(*(ancestors[s] for s in raw)) if raw else set()
        retained = tuple(sorted(raw - redundant))
        if any(a in ancestors[b] for a in retained for b in retained if a != b):
            raise AssertionError(f"Residual ancestor pair for question {qid}")
        mapping[qid] = retained
    return mapping


def primary_fields(row, mapping):
    uid, qid, aid = (integer_id(row[key]) for key in ("UserId", "QuestionId", "AnswerId"))
    response = integer_id(row["IsCorrect"])
    if uid is None or qid is None or aid is None:
        return None, "invalid_id"
    if qid not in mapping:
        return None, "unknown_question"
    if not mapping[qid]:
        return None, "no_kc"
    if response not in (0, 1):
        return None, "invalid_answer"
    return (uid, qid, aid, response), None


def read_data_from_csv(raw_root, write_file, sample_size=SAMPLE_SIZE, seed=SEED, min_seq_len=3):
    """Select the first eligible hash-ranked users, then write their sequences.

    Preliminary counts ignore dates only to exclude users who cannot possibly
    be eligible. Final eligibility includes valid answer dates. Refill from the
    same ranking when a provisional user has too few timestamp-valid records.
    No correctness rate, sequence ranking or KC density is used for selection.
    """
    if sample_size < 1 or min_seq_len < 1:
        raise ValueError("sample_size and min_seq_len must be positive")
    root = Path(raw_root)
    train = root / "train_data/train_task_1_2.csv"
    answers = root / "metadata/answer_metadata_task_1_2.csv"
    inputs = [train, answers, root / "metadata/subject_metadata.csv", root / "metadata/question_metadata_task_1_2.csv"]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Missing official Eedi Task 1/2 input: {path}")
    input_stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in inputs}
    mapping = load_question_to_kcs(root)
    required = ["UserId", "QuestionId", "AnswerId", "IsCorrect"]
    raw_counts, preliminary_counts, global_removed = Counter(), Counter(), Counter()
    for row in csv_rows(train, required, "Counting candidate students"):
        uid = integer_id(row["UserId"])
        if uid is not None:
            raw_counts[uid] += 1
        fields, reason = primary_fields(row, mapping)
        if reason:
            global_removed[reason] += 1
        else:
            preliminary_counts[fields[0]] += 1
    ranked = sorted((uid for uid in raw_counts if preliminary_counts[uid] >= min_seq_len), key=lambda uid: (selection_hash(uid, seed), uid))
    selected, sequences, per_user_removed = [], {}, {}
    cursor = 0
    while len(selected) < sample_size and cursor < len(ranked):
        batch = ranked[cursor:cursor + sample_size - len(selected)]
        cursor += len(batch)
        batch_set = set(batch)
        records = defaultdict(list)
        removed = {uid: Counter() for uid in batch}
        for original_index, row in enumerate(csv_rows(train, required, "Collecting candidate interactions")):
            uid = integer_id(row["UserId"])
            if uid not in batch_set:
                continue
            fields, reason = primary_fields(row, mapping)
            if reason:
                removed[uid][reason] += 1
            else:
                _, qid, aid, response = fields
                records[uid].append((qid, aid, response, original_index))
        needed = {r[1] for user_records in records.values() for r in user_records}
        dates = {}
        for row in csv_rows(answers, ["AnswerId", "DateAnswered"], "Reading candidate answer dates"):
            aid = integer_id(row["AnswerId"])
            if aid not in needed:
                continue
            timestamp = timestamp_ms(row["DateAnswered"])
            if aid in dates and dates[aid] != timestamp:
                raise ValueError(f"Conflicting official dates for AnswerId {aid}")
            dates[aid] = timestamp
        for uid in batch:
            valid = []
            for qid, aid, response, index in records[uid]:
                timestamp = dates.get(aid)
                if timestamp is None:
                    removed[uid]["invalid_timestamp"] += 1
                else:
                    valid.append((timestamp, index, qid, response))
            if len(valid) >= min_seq_len:
                selected.append(uid)
                sequences[uid] = sorted(valid, key=lambda r: (r[0], r[1]))
                per_user_removed[uid] = removed[uid]
        print(f"Selected {len(selected)}/{sample_size} students; examined {cursor} candidate students", flush=True)
    if len(selected) != sample_size:
        raise ValueError(f"Only {len(selected)} eligible students; required {sample_size}. No subset written.")
    for path in inputs:
        if (path.stat().st_size, path.stat().st_mtime_ns) != input_stats[str(path)]:
            raise RuntimeError(f"Input changed during preprocessing: {path}")
    output = Path(write_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    histogram, wrong, wrong_multi = Counter(), 0, 0
    used_questions, used_concepts = set(), set()
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for uid in tqdm(selected, desc="Writing Eedi subset", unit="student"):
            seq = sequences[uid]
            handle.write(f"{uid},{len(seq)}\n")
            for values in ([r[2] for r in seq], ["_".join(map(str, mapping[r[2]])) for r in seq], [r[3] for r in seq], [r[0] for r in seq]):
                handle.write(",".join(map(str, values)) + "\n")
            handle.write("NA\n")
            for _, _, qid, response in seq:
                kcs = mapping[qid]
                histogram[len(kcs)] += 1
                used_questions.add(qid)
                used_concepts.update(kcs)
                wrong += response == 0
                wrong_multi += response == 0 and len(kcs) >= 2
    temporary.replace(output)
    ids_text = "".join(f"{uid}\n" for uid in selected)
    manifest = {
        "sample_size": sample_size, "seed": seed, "selection_namespace": NAMESPACE,
        "selection_algorithm": "SHA256(namespace|seed|numeric UserId), ascending; numeric ID breaks ties",
        "eligibility_rule": f"at least {min_seq_len} valid interactions including valid answer dates",
        "candidate_students": len(raw_counts), "preliminary_eligible_students": len(ranked),
        "examined_students": cursor, "selected_students": len(selected),
        "selected_user_ids": selected,
        "selected_user_ids_sha256": hashlib.sha256(ids_text.encode("utf-8")).hexdigest(),
        "sha256_encoding": "UTF-8 ranked decimal user IDs, one per line, trailing LF",
        "raw_root": root.resolve().as_posix(),
        "response_rate_agnostic_selection": True, "kc_density_agnostic_selection": True,
    }
    with (output.parent / "selected_users.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sample_rank", "user_id", "raw_interactions", "valid_interactions", "selection_hash"])
        for rank, uid in enumerate(selected, 1):
            writer.writerow([rank, uid, raw_counts[uid], len(sequences[uid]), selection_hash(uid, seed)])
    removed = sum((per_user_removed[uid] for uid in selected), Counter())
    total = sum(histogram.values())
    multi = sum(n for k, n in histogram.items() if k >= 2)
    multi_questions = sum(len(mapping[q]) >= 2 for q in used_questions)
    audit = {
        "dataset": "eedi_task12", "selected_students": len(selected), "processed_students": len(sequences),
        "raw_interactions": sum(raw_counts[uid] for uid in selected), "valid_interactions": total,
        "num_questions": len(used_questions), "num_concepts": len(used_concepts), "max_concepts": max(histogram),
        "interaction_k1": histogram[1], "interaction_k2": histogram[2],
        "interaction_k3plus": sum(n for k, n in histogram.items() if k >= 3),
        "correct_interactions": total - wrong, "incorrect_interactions": wrong, "incorrect_multi_kc": wrong_multi,
        "multi_kc_questions": multi_questions, "multi_kc_question_ratio": multi_questions / len(used_questions),
        "multi_kc_interaction_ratio": multi / total, "incorrect_multi_ratio_all": wrong_multi / total,
        "incorrect_multi_ratio_incorrect": wrong_multi / wrong if wrong else 0.0,
        "incorrect_multi_ratio_multi": wrong_multi / multi if multi else 0.0,
        "question_kc_source": "official SubjectId sets with all transitive ancestors removed",
        "ordering": "DateAnswered (UTC milliseconds) -> original training row index",
        "atomic_kc_interpretation": "hierarchically non-redundant; no statistical or cognitive independence claimed",
        "global_primary_filter_counts": dict(global_removed),
    }
    for reason in ("invalid_id", "unknown_question", "no_kc", "invalid_answer", "invalid_timestamp"):
        audit[f"removed_{reason}"] = removed[reason]
    for name, payload in (("sampling_manifest.json", manifest), ("preprocess_audit.json", audit)):
        with (output.parent / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return manifest
