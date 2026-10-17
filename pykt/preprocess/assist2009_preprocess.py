"""ASSISTments 2009 adapter for the shared RAPLKT multi-KC pipeline."""

import json
import os
from collections import Counter, defaultdict
from statistics import mean, median

import pandas as pd
from tqdm import tqdm

from .utils import replace_text


SOURCE_ENCODING = "cp1252"
INVALID_KCS = {"", "nan", "none", "noskill", "unknown", "null", "na", "-1"}
REQUIRED_NON_KC = ["user_id", "problem_id", "assistment_id", "correct", "order_id"]


def _ratio(count, total):
    return count / total if total else 0.0


def _distribution(counts):
    total = len(counts)
    result = {
        "k1": sum(value == 1 for value in counts),
        "k2": sum(value == 2 for value in counts),
        "k3plus": sum(value >= 3 for value in counts),
    }
    result.update({f"{key}_ratio": _ratio(value, total) for key, value in result.items()})
    return result


def parse_kc_set(value):
    """Parse the row-level complete skill_id set, preserving source order."""
    concepts = []
    if pd.isna(value):
        return concepts
    for concept in str(value).split("_"):
        concept = concept.strip()
        if concept.lower() in INVALID_KCS:
            continue
        concept = replace_text(concept)
        if concept not in concepts:
            concepts.append(concept)
    return concepts


def build_question_to_kcs(df):
    """Audit the explicit full KC sets; never infer them from responses."""
    observed = defaultdict(set)
    first_seen = {}
    for question, skill_id in zip(df["problem_id"], df["skill_id"]):
        concepts = tuple(parse_kc_set(skill_id))
        if not concepts:
            continue
        question = str(question)
        # KC ordering does not change the set. Retain first-seen metadata
        # order for deterministic output while auditing genuine set conflicts.
        observed[question].add(frozenset(concepts))
        first_seen.setdefault(question, list(concepts))
    conflicts = {question: sets for question, sets in observed.items() if len(sets) > 1}
    if conflicts:
        examples = list(conflicts.items())[:5]
        raise ValueError(
            "ASSIST2009 problem_id has inconsistent explicit full KC sets; "
            f"conflicting questions={len(conflicts)}, examples={examples}."
        )
    return first_seen


def _write_audit(path, audit):
    with open(path, "w", encoding="utf8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def read_data_from_csv(read_file, write_file):
    """Convert the corrected-collapsed CSV to canonical six-line data.txt."""
    df = pd.read_csv(
        read_file,
        encoding=SOURCE_ENCODING,
        dtype=str,
        low_memory=False,
    )
    missing_columns = set(REQUIRED_NON_KC + ["skill_id", "ms_first_response"]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"ASSIST2009 is missing columns: {sorted(missing_columns)}")

    raw_students = int(df["user_id"].nunique(dropna=True))
    raw_questions = int(df["problem_id"].nunique(dropna=True))
    for column in REQUIRED_NON_KC:
        df[column] = df[column].str.strip().replace("", pd.NA)
    missing_non_kc = df[REQUIRED_NON_KC].isna().any(axis=1)
    parsed_kcs = df["skill_id"].map(parse_kc_set)
    no_kc = parsed_kcs.map(len).eq(0)
    response_numeric = pd.to_numeric(df["correct"], errors="coerce")
    invalid_response = (~missing_non_kc) & (~response_numeric.isin([0, 1]))
    order_numeric = pd.to_numeric(df["order_id"], errors="coerce")
    # order_id is a sequence index, not a Unix timestamp. Reject fractions,
    # infinity and out-of-range values before casting to signed int64.
    valid_order = (
        order_numeric.notna() & order_numeric.ge(0)
        & order_numeric.lt(2 ** 63) & order_numeric.mod(1).eq(0)
    )
    invalid_order = (~missing_non_kc) & (~valid_order)
    valid_mask = ~(missing_non_kc | no_kc | invalid_response | invalid_order)
    valid = df.loc[valid_mask].copy()
    if valid.empty:
        raise ValueError("ASSIST2009 has no valid interactions after filtering.")
    valid["_response"] = response_numeric.loc[valid_mask].astype(int)
    valid["_order"] = order_numeric.loc[valid_mask].astype("int64")
    valid["_original_index"] = valid.index.astype("int64")

    duration_numeric = pd.to_numeric(valid["ms_first_response"], errors="coerce")
    duration_numeric = duration_numeric.where(
        duration_numeric.gt(-(2 ** 63)) & duration_numeric.lt(2 ** 63)
    )
    invalid_duration = int(duration_numeric.isna().sum())
    duration_numeric = duration_numeric.fillna(0).round().astype("int64")
    negative_duration = int((duration_numeric < 0).sum())
    valid["_duration"] = duration_numeric.clip(lower=0)

    question_to_kcs = build_question_to_kcs(valid)
    valid["_question"] = valid["problem_id"].astype(str).map(replace_text)
    valid["_student"] = valid["user_id"].astype(str).map(replace_text)
    valid["_concepts"] = valid["problem_id"].astype(str).map(question_to_kcs)
    if valid["_concepts"].isna().any():
        raise ValueError("A valid interaction has no audited question -> KC set.")

    identity_counts = valid.groupby("problem_id", sort=False)["assistment_id"].nunique(dropna=False)
    reused_question_ids = int((identity_counts > 1).sum())
    if reused_question_ids:
        raise ValueError(
            f"problem_id is reused across assistment_id for {reused_question_ids} questions."
        )

    valid = valid.sort_values(
        ["_student", "_order", "_original_index"], kind="stable"
    )
    rows_by_student = []
    same_question_repeats = 0
    consecutive_repeats = 0
    student_groups = valid.groupby("_student", sort=False)
    for student, rows in tqdm(student_groups, desc="Preprocessing ASSIST2009", unit="student"):
        rows = rows.copy()
        seen_questions = set()
        previous_question = None
        for question in rows["_question"]:
            same_question_repeats += question in seen_questions
            consecutive_repeats += question == previous_question
            seen_questions.add(question)
            previous_question = question
        rows_by_student.append((student, rows))

    os.makedirs(os.path.dirname(os.path.abspath(write_file)), exist_ok=True)
    with open(write_file, "w", encoding="utf8", newline="\n") as stream:
        for student, rows in rows_by_student:
            assert len(rows) == len(rows["_question"]) == len(rows["_concepts"]) == len(rows["_response"]) == len(rows["_duration"])
            # Six-line canonical item sequences: no synthetic timestamps,
            # no ID mapping and no manual multi-KC expansion.
            stream.write(f"{student},{len(rows)}\n")
            stream.write(",".join(rows["_question"]) + "\n")
            stream.write(",".join("_".join(value) for value in rows["_concepts"]) + "\n")
            stream.write(",".join(str(value) for value in rows["_response"]) + "\n")
            stream.write("NA\n")
            stream.write(",".join(str(value) for value in rows["_duration"]) + "\n")

    question_counts = [len(concepts) for concepts in question_to_kcs.values()]
    interaction_counts = [len(concepts) for concepts in valid["_concepts"]]
    incorrect_counts = [
        len(concepts)
        for concepts, response in zip(valid["_concepts"], valid["_response"])
        if response == 0
    ]
    question_distribution = _distribution(question_counts)
    interaction_distribution = _distribution(interaction_counts)
    incorrect_distribution = _distribution(incorrect_counts)
    sequence_lengths = [len(rows) for _, rows in rows_by_student]

    future_revisits = 0
    revisit_denominator = 0
    exposures = Counter()
    for _, rows in rows_by_student:
        future_concepts = set()
        records = list(zip(rows["_question"], rows["_concepts"]))
        for question, concepts in reversed(records):
            concepts = set(concepts)
            if future_concepts:
                revisit_denominator += 1
                future_revisits += bool(concepts & future_concepts)
            future_concepts.update(concepts)
            exposures[question] += 1

    exact_duplicate_columns = [column for column in df.columns if column != "Unnamed: 0"]
    exact_duplicates = int(df.duplicated(exact_duplicate_columns).sum())
    valid_interactions = len(valid)
    incorrect_total = len(incorrect_counts)
    incorrect_multi = incorrect_distribution["k2"] + incorrect_distribution["k3plus"]
    audit = {
        "dataset": "assist2009",
        "source_files": {"interactions": os.path.basename(read_file)},
        "source_encoding": SOURCE_ENCODING,
        "schema": {
            "student_field": "user_id",
            "question_field": "problem_id",
            "kc_field": "skill_id (explicit complete set delimited by underscore)",
            "response_field": "correct",
            "timestamp_field": None,
            "ordering_field": "order_id",
            "duration_field": "ms_first_response",
            "auxiliary_fields": [
                "assignment_id", "assistment_id", "attempt_count", "sequence_id",
                "opportunity", "original",
            ],
        },
        "question_identity": "problem_id; maps to exactly one assistment_id",
        "kc_set_rule": "interaction skill_id is the explicit complete KC set; no union",
        "raw_students": raw_students,
        "raw_questions": raw_questions,
        "raw_interactions": len(df),
        "valid_students": len(rows_by_student),
        "valid_questions": len(question_to_kcs),
        "valid_interactions": valid_interactions,
        "num_kcs": len({concept for concepts in question_to_kcs.values() for concept in concepts}),
        "missing_records": int((missing_non_kc | no_kc).sum()),
        "invalid_response_records": int(invalid_response.sum()),
        "invalid_order_records": int(invalid_order.sum()),
        "invalid_duration_records_replaced_with_zero": invalid_duration,
        "empty_kc_questions": raw_questions - len(question_to_kcs),
        "empty_kc_interactions": int(no_kc.sum()),
        "duplicate_records": exact_duplicates,
        "negative_duration_records_clamped_to_zero": negative_duration,
        "same_student_same_question_repeat_count": same_question_repeats,
        "same_student_same_question_repeat_rate": _ratio(same_question_repeats, valid_interactions),
        "same_student_same_question_consecutive_repeat_count": consecutive_repeats,
        "same_student_same_question_consecutive_repeat_rate": _ratio(consecutive_repeats, valid_interactions),
        "question_kc_conflicts": 0,
        "questions_with_one_kc_set": len(question_to_kcs),
        "questions_with_multiple_kc_sets": 0,
        "questions_fixed_by_union": 0,
        "questions_fixed_by_redefining_id": 0,
        "questions_removed": raw_questions - len(question_to_kcs),
        "question_k1": question_distribution["k1"],
        "question_k1_ratio": question_distribution["k1_ratio"],
        "question_k2": question_distribution["k2"],
        "question_k2_ratio": question_distribution["k2_ratio"],
        "question_k3plus": question_distribution["k3plus"],
        "question_k3plus_ratio": question_distribution["k3plus_ratio"],
        "question_max_k": max(question_counts, default=0),
        "question_mean_k": mean(question_counts) if question_counts else 0.0,
        "question_median_k": median(question_counts) if question_counts else 0.0,
        "interaction_k1": interaction_distribution["k1"],
        "interaction_k1_ratio": interaction_distribution["k1_ratio"],
        "interaction_k2": interaction_distribution["k2"],
        "interaction_k2_ratio": interaction_distribution["k2_ratio"],
        "interaction_k3plus": interaction_distribution["k3plus"],
        "interaction_k3plus_ratio": interaction_distribution["k3plus_ratio"],
        "multi_kc_interaction_ratio": _ratio(
            interaction_distribution["k2"] + interaction_distribution["k3plus"], valid_interactions
        ),
        "incorrect_total": incorrect_total,
        "incorrect_k1": incorrect_distribution["k1"],
        "incorrect_k2": incorrect_distribution["k2"],
        "incorrect_k3plus": incorrect_distribution["k3plus"],
        "incorrect_multi": incorrect_multi,
        "incorrect_multi_ratio_all": _ratio(incorrect_multi, valid_interactions),
        "incorrect_multi_ratio_incorrect": _ratio(incorrect_multi, incorrect_total),
        "student_sequence_length": {
            "min": min(sequence_lengths, default=0),
            "max": max(sequence_lengths, default=0),
            "mean": mean(sequence_lengths) if sequence_lengths else 0.0,
            "median": median(sequence_lengths) if sequence_lengths else 0.0,
        },
        "same_kc_future_revisit_rate": _ratio(future_revisits, revisit_denominator),
        "question_exposure_count": {
            "min": min(exposures.values(), default=0),
            "max": max(exposures.values(), default=0),
            "mean": mean(exposures.values()) if exposures else 0.0,
            "median": median(exposures.values()) if exposures else 0.0,
        },
        "canonical_checks": {
            "one_sequence_per_student": len(rows_by_student) == valid["_student"].nunique(),
            "responses_binary": bool(valid["_response"].isin([0, 1]).all()),
            "no_empty_kc_sets": bool(valid["_concepts"].map(bool).all()),
            "kc_sets_deduplicated": bool(valid["_concepts"].map(
                lambda concepts: len(concepts) == len(set(concepts))
            ).all()),
            "order_sorted": all(
                rows["_order"].is_monotonic_increasing for _, rows in rows_by_student
            ),
        },
        "metadata_problem_ids_reused_across_assistments": reused_question_ids,
    }
    audit_path = os.path.join(os.path.dirname(os.path.abspath(write_file)), "preprocess_audit.json")
    _write_audit(audit_path, audit)
    print(
        "ASSIST2009 canonical data: "
        f"students={audit['valid_students']}, questions={audit['valid_questions']}, "
        f"interactions={valid_interactions}, Kmax={audit['question_max_k']}"
    )
    return audit

