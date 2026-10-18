"""Deterministic EdNet-KT1 5k subset selection and raw adapter."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

from .utils import format_list2str, replace_text, write_txt


ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "ednet_kt1_5k"
SUBSET_NAME = "ednet_kt1_5k_v1"
SELECTION_NAMESPACE = "RAETKT_EDNET_KT1_5K_V1"
SELECTION_SEED = 3407
TARGET_STUDENTS = 5000
STUDENT_PATTERN = re.compile(r"^u([0-9]+)\.csv$", re.IGNORECASE)
ANSWER_PATTERN = re.compile(r"^[a-d]$", re.IGNORECASE)
STUDENT_COLUMNS = {
    "timestamp", "solving_id", "question_id", "user_answer", "elapsed_time"
}
QUESTION_COLUMNS = {
    "question_id", "bundle_id", "explanation_id", "correct_answer",
    "part", "tags", "deployed_at",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_id(value):
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def normalize_answer(value):
    token = normalize_id(value)
    if token is None or ANSWER_PATTERN.fullmatch(token) is None:
        return None
    return token.casefold()


def parse_timestamp(value):
    token = normalize_id(value)
    if token is None or not token.isdigit():
        return None
    return int(token)


def parse_solving_id(value):
    token = normalize_id(value)
    if token is None or not token.isdigit():
        return None
    return int(token)


def parse_tags(value):
    tokens = []
    seen = set()
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return tokens, 0
    duplicates = 0
    for raw in str(value).split(";"):
        token = normalize_id(raw)
        if token is None:
            continue
        token = replace_text(token)
        if token in seen:
            duplicates += 1
            continue
        seen.add(token)
        tokens.append(token)
    tokens.sort(key=lambda token: (token.casefold(), token))
    return tokens, duplicates


def load_question_metadata(path):
    """Load canonical question metadata and return its schema audit."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Official EdNet Contents metadata is required but missing: {path}. "
            "Place Contents/questions.csv at this path before freezing the subset."
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    missing = QUESTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"questions.csv missing columns: {sorted(missing)}")
    mapping = {}
    stats = Counter()
    for row in frame.to_dict("records"):
        stats["metadata_rows"] += 1
        question = normalize_id(row.get("question_id"))
        if question is None:
            stats["missing_question_id"] += 1
            continue
        tags, duplicate_tags = parse_tags(row.get("tags"))
        stats["duplicate_tags_within_question"] += duplicate_tags
        answer = normalize_answer(row.get("correct_answer"))
        if answer is None:
            stats["invalid_correct_answer"] += 1
        if not tags:
            stats["questions_with_empty_tags"] += 1
        canonical = {
            "question_id": question,
            "correct_answer": answer,
            "bundle_id": normalize_id(row.get("bundle_id")),
            "tags": tags,
        }
        if question in mapping:
            stats["duplicate_question_id"] += 1
            if mapping[question] != canonical:
                stats["conflicting_duplicate_question_id"] += 1
            continue
        mapping[question] = canonical
    stats["questions_metadata"] = len(mapping)
    stats["max_raw_tags_per_question"] = max(
        (len(value["tags"]) for value in mapping.values()), default=0
    )
    for key in (
        "missing_question_id", "invalid_correct_answer",
        "questions_with_empty_tags", "duplicate_question_id",
        "conflicting_duplicate_question_id",
    ):
        stats.setdefault(key, 0)
    report = {
        "format": "CSV", "columns": list(frame.columns),
        **dict(stats),
    }
    return mapping, report


def canonical_student_file(path):
    path = Path(path)
    match = STUDENT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid EdNet student filename: {path.name}")
    numeric_id = int(match.group(1))
    return f"u{numeric_id}", numeric_id


def selection_hash(student_id):
    text = f"{SELECTION_NAMESPACE}|{SELECTION_SEED}|{student_id}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inspect_student_file(path, question_metadata, collect_interactions=False):
    """Apply canonical validity without using response correctness statistics."""
    student_id, numeric_id = canonical_student_file(path)
    stats = Counter()
    interactions = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        missing = STUDENT_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for source_row, row in enumerate(reader):
            stats["raw_interaction_count"] += 1
            timestamp = parse_timestamp(row.get("timestamp"))
            question = normalize_id(row.get("question_id"))
            user_answer = normalize_answer(row.get("user_answer"))
            solving_id = parse_solving_id(row.get("solving_id"))
            if timestamp is None:
                stats["invalid_timestamp"] += 1
            if question is None:
                stats["missing_question_id"] += 1
            if user_answer is None:
                stats["invalid_user_answer"] += 1
            metadata = question_metadata.get(question)
            if question is not None and metadata is None:
                stats["unknown_question"] += 1
            if metadata is not None and metadata["correct_answer"] is None:
                stats["invalid_correct_answer"] += 1
            if metadata is not None and not metadata["tags"]:
                stats["question_without_tags"] += 1
            if solving_id is None:
                stats["invalid_solving_id"] += 1
            valid = (
                timestamp is not None and question is not None
                and user_answer is not None and metadata is not None
                and metadata["correct_answer"] is not None
                and bool(metadata["tags"])
            )
            if not valid:
                continue
            stats["valid_interaction_count"] += 1
            if collect_interactions:
                interactions.append({
                    "student_id": student_id,
                    "numeric_student_id": numeric_id,
                    "question_id": question,
                    "kc_set": metadata["tags"],
                    "response": int(user_answer == metadata["correct_answer"]),
                    "timestamp": timestamp,
                    "solving_id": solving_id,
                    "source_row": source_row,
                    "bundle_id": metadata["bundle_id"],
                    "user_answer": user_answer,
                })
    result = {
        "student_id": student_id,
        "numeric_student_id": numeric_id,
        "raw_filename": Path(path).name,
        "raw_interaction_count": stats["raw_interaction_count"],
        "valid_interaction_count": stats["valid_interaction_count"],
        "selection_hash": selection_hash(student_id),
        "audit": dict(stats),
    }
    if collect_interactions:
        interactions.sort(key=lambda row: (
            row["timestamp"],
            (0, row["solving_id"]) if row["solving_id"] is not None
            else (1, row["source_row"]),
            row["source_row"],
        ))
        result["interactions"] = interactions
    return result


def rank_candidates(candidate_records, target_students=TARGET_STUDENTS):
    eligible = [
        dict(record) for record in candidate_records
        if int(record["valid_interaction_count"]) >= 3
    ]
    eligible.sort(key=lambda row: (
        row["selection_hash"], int(row["numeric_student_id"])
    ))
    selected = eligible[:target_students]
    for rank, row in enumerate(selected, 1):
        row["rank"] = rank
    return selected, len(eligible)


def select_students(file_paths, question_metadata, target_students=TARGET_STUDENTS):
    records = [inspect_student_file(path, question_metadata) for path in file_paths]
    return rank_candidates(records, target_students)


def _relative_to_root(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_subset_manifest(
        selected, candidate_students, raw_dir, questions_csv, subset_dir,
        raw_student_file_count=None, target_students=TARGET_STUDENTS,
):
    subset_dir = Path(subset_dir)
    if len(selected) != target_students:
        raise AssertionError(
            f"Expected exactly {target_students} selected students, got {len(selected)}"
        )
    subset_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "rank", "student_id", "numeric_student_id", "raw_filename",
        "raw_interaction_count", "valid_interaction_count", "selection_hash",
    ]
    rows = [{key: row[key] for key in columns} for row in selected]
    csv_path = subset_dir / f"{SUBSET_NAME}_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    ids_text = "".join(f"{row['student_id']}\n" for row in rows)
    ids_path = subset_dir / "selected_student_ids.txt"
    ids_path.write_text(ids_text, encoding="utf-8", newline="\n")
    logical = {
        "subset_name": SUBSET_NAME,
        "raw_dataset": "EdNet-KT1",
        "target_students": target_students,
        "candidate_students": candidate_students,
        "selected_students": len(rows),
        "raw_student_file_count": raw_student_file_count,
        "eligibility_rule": "at least 3 canonical valid interactions",
        "hash_algorithm": "sha256",
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_seed": SELECTION_SEED,
        "response_agnostic_selection": True,
        "kc_density_agnostic": True,
        "kc_statistics_used_for_selection": False,
        "model_agnostic": True,
        "model_results_used_for_selection": False,
        "raw_student_dir": _relative_to_root(raw_dir),
        "questions_csv": _relative_to_root(questions_csv),
        "questions_csv_sha256": sha256_file(questions_csv),
        "selected_ids_sha256": hashlib.sha256(ids_text.encode("utf-8")).hexdigest(),
        "manifest_csv_sha256": sha256_file(csv_path),
        "selected_records": rows,
    }
    logical["manifest_sha256"] = canonical_json_sha256(logical)
    commit = _git_commit()
    if commit:
        logical["code_commit"] = commit
    json_path = subset_dir / f"{SUBSET_NAME}_manifest.json"
    json_path.write_text(
        json.dumps(logical, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return logical


def load_manifest(path):
    path = Path(path)
    if path.is_dir():
        path = path / "subset" / f"{SUBSET_NAME}_manifest.json"
    with path.open("r", encoding="utf-8") as fin:
        return json.load(fin), path


def resolve_manifest_inputs(manifest):
    raw_dir = Path(manifest["raw_student_dir"])
    questions = Path(manifest["questions_csv"])
    if not raw_dir.is_absolute():
        raw_dir = ROOT / raw_dir
    if not questions.is_absolute():
        questions = ROOT / questions
    return raw_dir, questions


def load_selected_sequences(manifest_path):
    manifest, _ = load_manifest(manifest_path)
    raw_dir, questions_path = resolve_manifest_inputs(manifest)
    metadata, metadata_audit = load_question_metadata(questions_path)
    sequences = []
    per_student_audit = []
    for selected in manifest["selected_records"]:
        result = inspect_student_file(
            raw_dir / selected["raw_filename"], metadata,
            collect_interactions=True,
        )
        assert result["student_id"] == selected["student_id"]
        assert result["valid_interaction_count"] == selected["valid_interaction_count"]
        sequences.append({
            "student_id": result["student_id"],
            "interactions": result["interactions"],
        })
        per_student_audit.append(result["audit"])
    return sequences, manifest, metadata, metadata_audit, per_student_audit


def read_data_from_csv(read_file, write_file):
    sequences, manifest, _, _, _ = load_selected_sequences(read_file)
    output = []
    for sequence in sequences:
        interactions = sequence["interactions"]
        output.append([
            [replace_text(sequence["student_id"]), str(len(interactions))],
            format_list2str([
                replace_text(row["question_id"]) for row in interactions
            ]),
            ["_".join(row["kc_set"]) for row in interactions],
            format_list2str([row["response"] for row in interactions]),
            format_list2str([row["timestamp"] for row in interactions]),
            ["NA"],
        ])
    write_txt(write_file, output)
    print(
        f"EdNet-KT1 adapter: subset={manifest['subset_name']}, "
        f"students={len(sequences)}, interactions="
        f"{sum(len(sequence['interactions']) for sequence in sequences)}"
    )
    return manifest
