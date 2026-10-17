"""EdNet adapter: original seed=2 sampling, canonical six-line interactions.

Select the first 5000 existing files in shuffled range(840473), before any
interaction filtering. Shared split code handles sequence-length filtering,
multi-KC expansion, ID mapping and train/validation/test generation.
"""

import csv
import hashlib
import json
import random
from decimal import Decimal, InvalidOperation
from pathlib import Path

from tqdm import tqdm

SEED = 2
USER_ID_UPPER_BOUND = 840473


def normalize_answer(value):
    value = str(value or "").strip().lower()
    return value if value in {"a", "b", "c", "d"} else None


def parse_integer(value):
    """Parse exact nonnegative integers without timestamp float rounding."""
    try:
        number = Decimal(str(value).strip())
        if number.is_finite() and number >= 0 and number == number.to_integral_value():
            return int(number)
    except (InvalidOperation, ValueError, TypeError):
        pass
    return None


def normalize_tags(value):
    """Preserve official atomic tag order and remove missing/invalid tags."""
    tags = []
    for token in str(value or "").split(";"):
        token = token.strip()
        if token.isascii() and token.isdigit():
            tag = str(int(token))
            if tag not in tags:
                tags.append(tag)
    return tuple(tags)


def read_rows(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        yield from reader


def read_data_from_csv(read_file, write_file, dataset_name=None):
    dataset_name = dataset_name or "ednet"
    if dataset_name not in {"ednet", "ednet5w"}:
        raise ValueError("Supported datasets: ednet and ednet5w")
    raw_root = Path(read_file)
    write_dir = raw_root if dataset_name == "ednet" else raw_root.parent / "ednet5w"
    output_file = write_dir / Path(write_file).name
    sample_size = 5000 if dataset_name == "ednet" else 50000
    skip_students = 0 if dataset_name == "ednet" else 5000
    required_students = skip_students + sample_size

    print(f"Shuffling range({USER_ID_UPPER_BOUND}) with seed={SEED}", flush=True)
    candidate_ids = list(range(USER_ID_UPPER_BOUND))
    random.Random(SEED).shuffle(candidate_ids)
    selected = []
    existing_students = examined_ids = 0
    with tqdm(total=required_students, desc="Selecting existing EdNet students", unit="student") as progress:
        for uid in candidate_ids:
            examined_ids += 1
            path = raw_root / "KT1" / f"u{uid}.csv"
            if not path.is_file():
                continue
            existing_students += 1
            progress.update(1)
            if existing_students > skip_students:
                selected.append((uid, path))
            if existing_students == required_students:
                break
    if len(selected) != sample_size:
        raise ValueError(f"Only {len(selected)} selected student files; requested {sample_size}")

    metadata_path = raw_root / "Contents" / "questions.csv"
    if not metadata_path.is_file():
        metadata_path = raw_root / "contents" / "questions.csv"
    print(f"Loading official question metadata: {metadata_path}", flush=True)
    questions = {}
    for row in read_rows(metadata_path, ("question_id", "correct_answer", "tags")):
        question = (row["question_id"] or "").strip()
        if not question:
            continue
        metadata = (normalize_answer(row["correct_answer"]), normalize_tags(row["tags"]))
        if question in questions and questions[question] != metadata:
            raise ValueError(f"Conflicting official metadata for {question}")
        questions[question] = metadata

    audit = dict.fromkeys((
        "raw_interactions", "valid_interactions", "processed_students",
        "removed_unknown_question", "removed_no_kc", "removed_invalid_answer",
        "removed_invalid_timestamp", "empty_students_after_filter",
        "correct_interactions", "incorrect_interactions", "multi_kc_interactions",
    ), 0)
    used_questions, used_concepts = set(), set()
    max_concepts = 0
    write_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(output_file.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for uid, path in tqdm(selected, desc="Preprocessing EdNet students", unit="student"):
                interactions = []
                for index, row in enumerate(read_rows(
                    path, ("timestamp", "solving_id", "question_id", "user_answer", "elapsed_time")
                )):
                    audit["raw_interactions"] += 1
                    question = (row["question_id"] or "").strip()
                    if question not in questions:
                        audit["removed_unknown_question"] += 1
                        continue
                    correct_answer, tags = questions[question]
                    if not tags:
                        audit["removed_no_kc"] += 1
                        continue
                    answer = normalize_answer(row["user_answer"])
                    if answer is None or correct_answer is None:
                        audit["removed_invalid_answer"] += 1
                        continue
                    timestamp = parse_integer(row["timestamp"])
                    if timestamp is None:
                        audit["removed_invalid_timestamp"] += 1
                        continue
                    solving_id = parse_integer(row["solving_id"])
                    # Missing solving IDs sort after valid IDs at equal timestamps.
                    order = (timestamp, solving_id is None, solving_id or 0, index)
                    response = int(answer == correct_answer)
                    interactions.append((order, question, "_".join(tags), str(response), str(timestamp)))
                    used_questions.add(question)
                    used_concepts.update(tags)
                    max_concepts = max(max_concepts, len(tags))
                    audit["correct_interactions" if response else "incorrect_interactions"] += 1
                    audit["multi_kc_interactions"] += int(len(tags) > 1)
                if not interactions:
                    audit["empty_students_after_filter"] += 1
                    continue
                # Keep repeat attempts and one canonical row per question response.
                interactions.sort(key=lambda interaction: interaction[0])
                handle.write(f"{uid},{len(interactions)}\n")
                for field in range(1, 5):
                    handle.write(",".join(interaction[field] for interaction in interactions) + "\n")
                # Like Algebra2005, duration is unused.
                handle.write("NA\n")
                audit["processed_students"] += 1
                audit["valid_interactions"] += len(interactions)
        if not audit["valid_interactions"]:
            raise ValueError("No valid interactions remain after filtering")
        temporary.replace(output_file)
    finally:
        if temporary.exists():
            temporary.unlink()

    digest = hashlib.sha256("\n".join(str(uid) for uid, _ in selected).encode("utf-8")).hexdigest()
    # Preserve the selection before cleaning: students are never replaced.
    with (write_dir / "selected_users.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_rank", "user_id", "filename"))
        writer.writerows((rank, uid, path.name) for rank, (uid, path) in enumerate(selected, 1))
    manifest = {
        "dataset_name": dataset_name, "seed": SEED, "sample_size": sample_size,
        "user_id_upper_bound_exclusive": USER_ID_UPPER_BOUND,
        "examined_user_ids": examined_ids, "skipped_existing_students": skip_students,
        "selected_students": len(selected), "selected_user_ids_sha256": digest,
        "sha256_encoding": "UTF-8 ranked decimal user IDs joined by LF; no trailing LF",
        "selection_rule": "first existing files in seeded shuffled ID range; no eligibility filtering",
    }
    with (write_dir / "sampling_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    audit.update(selected_students=len(selected), num_questions=len(used_questions),
                 num_concepts=len(used_concepts), max_concepts=max_concepts)
    with (write_dir / "preprocess_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
        handle.write("\n")
    print(json.dumps(audit, indent=2), flush=True)
    print(f"Selected user IDs SHA256: {digest}", flush=True)
    print(f"Canonical data written to {output_file}", flush=True)
    # Config paths must also work when Windows-preprocessed data moves to Linux.
    return write_dir.as_posix(), output_file.as_posix()
