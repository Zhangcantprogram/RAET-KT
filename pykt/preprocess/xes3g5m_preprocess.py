"""Adapt the distributed XES3G5M question-level files to canonical pyKT data.

The local release contains segmented train sequences and unsegmented test
sequences, not a raw event table. Reassemble the available training chunks by
student and combine with test students before the existing shared split. Never
consume window files or expanded KC-level rows: those duplicate interactions.
Any short tails already omitted by the release cannot be reconstructed.
"""

import csv
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from itertools import groupby
from pathlib import Path

from tqdm import tqdm

SOURCE_FILES = (
    "question_level/train_valid_sequences_quelevel.csv",
    "question_level/test_quelevel.csv",
)


def parse_integer(value):
    try:
        number = Decimal(str(value).strip())
        if number.is_finite() and number >= 0 and number == number.to_integral_value():
            return int(number)
    except (InvalidOperation, ValueError, TypeError):
        pass
    return None


def load_question_to_kcs(raw_root):
    """Resolve the official leaf KC names to their published numeric IDs.

    A route's ancestors describe its hierarchy, not additional atomic KCs.
    Multiple routes may end at the same leaf; deduplicate those IDs in official
    annotation order. Do not derive metadata from correctness or student rows.
    Preserve whitespace in official names: distinct published IDs can have
    names that differ only by leading or trailing spaces.
    """
    metadata = Path(raw_root) / "metadata"
    with (metadata / "questions.json").open(encoding="utf-8-sig") as handle:
        questions = json.load(handle)
    with (metadata / "kc_routes_map.json").open(encoding="utf-8-sig") as handle:
        kc_names = json.load(handle)
    if not isinstance(questions, dict) or not isinstance(kc_names, dict):
        raise ValueError("XES3G5M metadata must contain JSON objects")
    name_to_id = {}
    for cid, name in kc_names.items():
        if parse_integer(cid) is None or not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid official KC metadata: {cid}")
        if name in name_to_id and name_to_id[name] != str(cid):
            raise ValueError(f"Ambiguous official leaf KC name: {name}")
        name_to_id[name] = str(cid)
    question_to_kcs = {}
    for question, info in questions.items():
        if parse_integer(question) is None or not isinstance(info, dict):
            raise ValueError(f"Invalid question metadata: {question}")
        routes = info.get("kc_routes") or []
        if not isinstance(routes, list):
            raise ValueError(f"kc_routes must be a list for question {question}")
        concepts = []
        for route in routes:
            if not isinstance(route, str):
                raise ValueError(f"Invalid KC route for question {question}")
            leaf = route.split("----")[-1]
            if not leaf.strip():
                continue
            if leaf not in name_to_id:
                raise ValueError(f"Unmapped official KC leaf for question {question}: {leaf}")
            cid = name_to_id[leaf]
            if cid not in concepts:
                concepts.append(cid)
        question_to_kcs[str(question)] = tuple(concepts)
    return question_to_kcs


def iter_source_rows(path, segmented):
    required = {"uid", "questions", "concepts", "responses", "timestamps"}
    if segmented:
        required.add("selectmasks")
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            uid = (row["uid"] or "").strip()
            if parse_integer(uid) is None:
                raise ValueError(f"{path}:{line}: invalid uid {uid!r}")
            row["uid"] = uid
            yield line, row


def read_data_from_csv(read_file, write_file):
    """Write one six-line sequence per student, leaving shared split untouched."""
    raw_root = Path(read_file)
    output = Path(write_file)
    # Published original test sequences can exceed csv's default field limit.
    previous_field_limit = csv.field_size_limit()
    csv.field_size_limit(max(previous_field_limit, 100_000_000))
    temporary = output.with_name(output.name + ".tmp")
    audit = dict.fromkeys((
        "source_rows", "raw_interactions", "padding_positions",
        "processed_students", "empty_students_after_filter", "valid_interactions",
        "removed_unknown_question", "removed_no_kc", "removed_invalid_answer",
        "removed_invalid_timestamp", "source_kc_set_mismatches",
        "correct_interactions", "incorrect_interactions", "incorrect_multi_kc",
        "interaction_k1", "interaction_k2", "interaction_k3plus",
    ), 0)
    try:
        for relative in SOURCE_FILES:
            if not (raw_root / relative).is_file():
                raise FileNotFoundError(raw_root / relative)
        print("Loading official XES3G5M question/KC metadata", flush=True)
        question_to_kcs = load_question_to_kcs(raw_root)
        seen_students, used_questions, used_concepts = set(), set(), set()
        sources = {}
        output.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for source_index, relative in enumerate(SOURCE_FILES):
                path = raw_root / relative
                source_stats = Counter()
                # Shared pyKT emits each student's training chunks contiguously.
                # Reject noncontiguous/overlapping IDs instead of silently writing
                # duplicate student sequences or leaking one student across splits.
                groups = groupby(iter_source_rows(path, source_index == 0),
                                 key=lambda entry: entry[1]["uid"])
                for uid, chunks in tqdm(groups, desc=f"Reading {path.name}", unit="student"):
                    if uid in seen_students:
                        raise ValueError(f"Repeated/noncontiguous student {uid} in {path}; cannot safely reassemble")
                    seen_students.add(uid)
                    source_stats["students"] += 1
                    interactions = []
                    original_index = 0
                    for line, row in chunks:
                        source_stats["rows"] += 1
                        audit["source_rows"] += 1
                        fields = {key: (row[key] or "").split(",") for key in
                                  ("questions", "concepts", "responses", "timestamps")}
                        lengths = {len(values) for values in fields.values()}
                        if len(lengths) != 1:
                            raise ValueError(f"{path}:{line}: unequal sequence lengths")
                        masks = (row["selectmasks"] or "").split(",") if "selectmasks" in row else None
                        if masks is not None and len(masks) != len(fields["questions"]):
                            raise ValueError(f"{path}:{line}: selectmasks length mismatch")
                        for pos, (question, source_kcs, response, timestamp) in enumerate(zip(
                            fields["questions"], fields["concepts"], fields["responses"], fields["timestamps"]
                        )):
                            index = original_index
                            original_index += 1
                            question, response, timestamp = question.strip(), response.strip(), timestamp.strip()
                            mask = masks[pos].strip() if masks is not None else None
                            if mask is not None and mask not in {"-1", "0", "1"}:
                                raise ValueError(f"{path}:{line}: invalid selectmask {mask!r}")
                            if question == "-1":
                                if response != "-1" or timestamp != "-1" or source_kcs.strip() != "-1" or (mask is not None and mask != "-1"):
                                    raise ValueError(f"{path}:{line}: inconsistent padding")
                                audit["padding_positions"] += 1
                                continue
                            if mask == "-1":
                                raise ValueError(f"{path}:{line}: nonpadding interaction has padding mask")
                            audit["raw_interactions"] += 1
                            source_stats["raw_interactions"] += 1
                            if question not in question_to_kcs:
                                audit["removed_unknown_question"] += 1
                                continue
                            concepts = question_to_kcs[question]
                            if not concepts:
                                audit["removed_no_kc"] += 1
                                continue
                            if response not in {"0", "1"}:
                                audit["removed_invalid_answer"] += 1
                                continue
                            time = parse_integer(timestamp)
                            if time is None:
                                audit["removed_invalid_timestamp"] += 1
                                continue
                            # Source concepts are audited, never used to infer KC.
                            if set(source_kcs.strip().split("_")) != set(concepts):
                                audit["source_kc_set_mismatches"] += 1
                            interactions.append((time, index, question, "_".join(concepts), response))
                    if not interactions:
                        audit["empty_students_after_filter"] += 1
                        continue
                    interactions.sort(key=lambda event: (event[0], event[1]))
                    handle.write(f"{uid},{len(interactions)}\n")
                    for field in (2, 3, 4, 0):
                        handle.write(",".join(str(event[field]) for event in interactions) + "\n")
                    handle.write("NA\n")
                    audit["processed_students"] += 1
                    audit["valid_interactions"] += len(interactions)
                    source_stats["valid_interactions"] += len(interactions)
                    for time, index, question, kcs, response in interactions:
                        used_questions.add(question)
                        used_concepts.update(kcs.split("_"))
                        count = len(question_to_kcs[question])
                        audit["interaction_k1" if count == 1 else "interaction_k2" if count == 2 else "interaction_k3plus"] += 1
                        audit["correct_interactions" if response == "1" else "incorrect_interactions"] += 1
                        audit["incorrect_multi_kc"] += int(response == "0" and count > 1)
                sources[relative] = dict(source_stats)
        if not audit["valid_interactions"]:
            raise ValueError("XES3G5M has no valid interactions after filtering")
        question_counts = [len(question_to_kcs[q]) for q in used_questions]
        multi_questions = sum(count > 1 for count in question_counts)
        audit.update(
            dataset="xes3g5m", source_files=sources, raw_students=len(seen_students),
            num_questions=len(used_questions), num_concepts=len(used_concepts),
            max_concepts=max(question_counts), multi_kc_questions=multi_questions,
            multi_kc_question_ratio=multi_questions / len(used_questions),
            multi_kc_interaction_ratio=(audit["interaction_k2"] + audit["interaction_k3plus"]) / audit["valid_interactions"],
            incorrect_multi_ratio_all=audit["incorrect_multi_kc"] / audit["valid_interactions"],
            incorrect_multi_ratio_incorrect=(audit["incorrect_multi_kc"] / audit["incorrect_interactions"] if audit["incorrect_interactions"] else 0.0),
            question_kc_source="official questions.json kc_routes leaves resolved via kc_routes_map.json",
            ordering="timestamp -> original position across contiguous student chunks",
            source_limitation="Distributed train sequences may have omitted short tails; unavailable interactions cannot be recovered",
            split_policy="Merge available train/test students; existing shared split regenerates student partitions and windows",
        )
        temporary.replace(output)
        with (output.parent / "preprocess_audit.json").open("w", encoding="utf-8") as handle:
            json.dump(audit, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
        return audit
    finally:
        csv.field_size_limit(previous_field_limit)
        if temporary.exists():
            temporary.unlink()
