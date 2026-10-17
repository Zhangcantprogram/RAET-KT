"""MOOC-Radar adapter for the shared RAPLKT multi-KC pipeline.

The fine-grained interaction file contains one ``seq`` per student and the
problem file is JSON Lines.  Problem metadata is authoritative for the full
question -> KC-set relation; interaction responses are never used to build it.
"""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from statistics import mean, median

from .utils import replace_text


INVALID_KCS = {"", "nan", "none", "noskill", "unknown", "null"}
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def _percent(count, total):
    return count / total if total else 0.0


def _distribution(counts):
    total = len(counts)
    k1 = sum(value == 1 for value in counts)
    k2 = sum(value == 2 for value in counts)
    k3plus = sum(value >= 3 for value in counts)
    return {
        "k1": k1,
        "k1_ratio": _percent(k1, total),
        "k2": k2,
        "k2_ratio": _percent(k2, total),
        "k3plus": k3plus,
        "k3plus_ratio": _percent(k3plus, total),
    }


def load_question_kc_metadata(problem_file):
    """Load authoritative complete KC sets and audit question identity."""
    metadata_rows = []
    with open(problem_file, "r", encoding="utf8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            metadata_rows.append(row)

    rows_by_question = defaultdict(list)
    for row in metadata_rows:
        rows_by_question[row.get("problem_id")].append(row)

    duplicate_problem_records = 0
    concept_conflicts = 0
    course_reuse = 0
    question_to_kcs = {}
    empty_questions = set()
    for question, rows in rows_by_question.items():
        duplicate_problem_records += max(0, len(rows) - 1)
        raw_sets = {tuple(row.get("concepts") or []) for row in rows}
        if len(raw_sets) > 1:
            concept_conflicts += 1
        if len({row.get("course_id") for row in rows}) > 1:
            course_reuse += 1

        concepts = []
        for concept in rows[0].get("concepts") or []:
            value = str(concept).strip()
            if value.lower() in INVALID_KCS:
                continue
            value = replace_text(value)
            if value not in concepts:
                concepts.append(value)
        if concepts:
            question_to_kcs[str(question)] = concepts
        else:
            empty_questions.add(str(question))

    if duplicate_problem_records or concept_conflicts or course_reuse:
        raise ValueError(
            "problem_id is not a unique, fixed item identity: "
            f"duplicate records={duplicate_problem_records}, "
            f"KC conflicts={concept_conflicts}, course reuse={course_reuse}."
        )

    return question_to_kcs, {
        "metadata_records": len(metadata_rows),
        "metadata_questions": len(rows_by_question),
        "metadata_empty_kc_questions": len(empty_questions),
        "metadata_duplicate_problem_records": duplicate_problem_records,
        "metadata_question_kc_conflicts": concept_conflicts,
        "metadata_problem_ids_reused_across_courses": course_reuse,
    }


def _to_timestamp(value):
    parsed = datetime.strptime(value, TIMESTAMP_FORMAT)
    # A timezone-free integer is deterministic and preserves all ordering and
    # elapsed-time information in the source's local clock.
    epoch = datetime(1970, 1, 1)
    return int((parsed - epoch).total_seconds() * 1000)


def _load_interactions(interaction_file):
    with open(interaction_file, "r", encoding="utf8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError("student-problem-fine.json must contain a JSON array.")
    return data


def _write_audit(path, audit):
    with open(path, "w", encoding="utf8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def read_data_from_csv(read_file, write_file, question_metadata=None):
    """Convert MOOC-Radar fine interactions to canonical six-line data.txt."""
    problem_file = question_metadata or os.path.join(
        os.path.dirname(os.path.abspath(read_file)), "problem.json"
    )
    question_to_kcs, metadata_audit = load_question_kc_metadata(problem_file)
    student_groups = _load_interactions(read_file)

    raw_users = set()
    raw_questions = set()
    raw_interactions = 0
    removed_missing = 0
    removed_invalid_response = 0
    removed_invalid_timestamp = 0
    removed_no_kc = 0
    duplicate_counter = Counter()
    rows_by_student = defaultdict(list)
    original_index = 0

    for group in student_groups:
        sequence = group.get("seq") or []
        for row in sequence:
            raw_interactions += 1
            user = row.get("user_id")
            question = row.get("problem_id")
            response = row.get("is_correct")
            submit_time = row.get("submit_time")
            if user is not None:
                raw_users.add(str(user))
            if question is not None:
                raw_questions.add(str(question))

            if any(
                value is None or str(value).strip() == ""
                for value in (user, question, response, submit_time)
            ):
                removed_missing += 1
                original_index += 1
                continue
            if response not in (0, 1):
                removed_invalid_response += 1
                original_index += 1
                continue
            question = str(question)
            concepts = question_to_kcs.get(question)
            if not concepts:
                removed_no_kc += 1
                original_index += 1
                continue
            try:
                timestamp = _to_timestamp(str(submit_time))
            except (TypeError, ValueError):
                removed_invalid_timestamp += 1
                original_index += 1
                continue

            user = str(user)
            duplicate_counter[(user, question, str(submit_time), response)] += 1
            rows_by_student[user].append(
                {
                    "question": replace_text(question),
                    "concepts": concepts,
                    "response": int(response),
                    "timestamp": timestamp,
                    "original_index": original_index,
                }
            )
            original_index += 1

    exact_duplicate_records = sum(
        count - 1 for count in duplicate_counter.values() if count > 1
    )
    # Exact duplicates are retained unless the dataset semantics prove that
    # they are accidental duplicate logging.  MOOC-Radar has none currently.
    valid_rows = []
    for user, rows in rows_by_student.items():
        rows.sort(key=lambda item: (item["timestamp"], item["original_index"]))
        valid_rows.append((user, rows))

    same_question_repeats = 0
    consecutive_repeats = 0
    for _, rows in valid_rows:
        seen_questions = set()
        previous_question = None
        for row in rows:
            question = row["question"]
            if question in seen_questions:
                same_question_repeats += 1
            if question == previous_question:
                consecutive_repeats += 1
            seen_questions.add(question)
            previous_question = question

    os.makedirs(os.path.dirname(os.path.abspath(write_file)), exist_ok=True)
    with open(write_file, "w", encoding="utf8", newline="\n") as stream:
        for user, rows in valid_rows:
            stream.write(f"{replace_text(user)},{len(rows)}\n")
            stream.write(",".join(row["question"] for row in rows) + "\n")
            stream.write(",".join("_".join(row["concepts"]) for row in rows) + "\n")
            stream.write(",".join(str(row["response"]) for row in rows) + "\n")
            stream.write(",".join(str(row["timestamp"]) for row in rows) + "\n")
            stream.write("NA\n")

    interactions = [row for _, rows in valid_rows for row in rows]
    question_kcs = {}
    for row in interactions:
        question_kcs[row["question"]] = tuple(row["concepts"])
    question_counts = [len(concepts) for concepts in question_kcs.values()]
    interaction_counts = [len(row["concepts"]) for row in interactions]
    incorrect_counts = [
        len(row["concepts"]) for row in interactions if row["response"] == 0
    ]
    question_distribution = _distribution(question_counts)
    interaction_distribution = _distribution(interaction_counts)
    incorrect_distribution = _distribution(incorrect_counts)
    sequence_lengths = [len(rows) for _, rows in valid_rows]

    future_revisits = 0
    revisit_denominator = 0
    exposures = Counter()
    for _, rows in valid_rows:
        future_concepts = set()
        for row in reversed(rows):
            concepts = set(row["concepts"])
            if future_concepts:
                revisit_denominator += 1
                if concepts & future_concepts:
                    future_revisits += 1
            future_concepts.update(concepts)
            exposures[row["question"]] += 1

    valid_interactions = len(interactions)
    incorrect_total = len(incorrect_counts)
    incorrect_multi = incorrect_distribution["k2"] + incorrect_distribution["k3plus"]
    audit = {
        "dataset": "moocradar",
        "source_files": {
            "interactions": os.path.basename(read_file),
            "question_metadata": os.path.basename(problem_file),
        },
        "schema": {
            "student_field": "seq[].user_id",
            "question_field": "seq[].problem_id",
            "kc_field": "problem.json concepts (official complete KC set)",
            "response_field": "seq[].is_correct",
            "timestamp_field": "seq[].submit_time",
            "duration_field": None,
            "auxiliary_fields": ["log_id", "attempts", "score"],
        },
        "question_identity": "problem_id; globally unique in problem.json and not reused across courses",
        "kc_set_rule": "authoritative problem.json concepts; preserve order and de-duplicate",
        "raw_students": len(raw_users),
        "raw_questions": len(raw_questions),
        "raw_interactions": raw_interactions,
        "valid_students": len(valid_rows),
        "valid_questions": len(question_kcs),
        "valid_interactions": valid_interactions,
        "num_kcs": len({concept for row in interactions for concept in row["concepts"]}),
        "missing_records": removed_missing,
        "invalid_response_records": removed_invalid_response,
        "invalid_timestamp_records": removed_invalid_timestamp,
        "empty_kc_questions": metadata_audit["metadata_empty_kc_questions"],
        "empty_kc_interactions": removed_no_kc,
        "duplicate_records": exact_duplicate_records,
        "same_student_same_question_repeat_count": same_question_repeats,
        "same_student_same_question_repeat_rate": _percent(same_question_repeats, valid_interactions),
        "same_student_same_question_consecutive_repeat_count": consecutive_repeats,
        "same_student_same_question_consecutive_repeat_rate": _percent(consecutive_repeats, valid_interactions),
        "question_kc_conflicts": metadata_audit["metadata_question_kc_conflicts"],
        "questions_with_one_kc_set": len(question_kcs),
        "questions_with_multiple_kc_sets": 0,
        "questions_fixed_by_union": 0,
        "questions_fixed_by_redefining_id": 0,
        "questions_removed": metadata_audit["metadata_empty_kc_questions"],
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
        "multi_kc_interaction_ratio": _percent(
            interaction_distribution["k2"] + interaction_distribution["k3plus"],
            valid_interactions,
        ),
        "incorrect_total": incorrect_total,
        "incorrect_k1": incorrect_distribution["k1"],
        "incorrect_k2": incorrect_distribution["k2"],
        "incorrect_k3plus": incorrect_distribution["k3plus"],
        "incorrect_multi": incorrect_multi,
        "incorrect_multi_ratio_all": _percent(incorrect_multi, valid_interactions),
        "incorrect_multi_ratio_incorrect": _percent(incorrect_multi, incorrect_total),
        "student_sequence_length": {
            "min": min(sequence_lengths, default=0),
            "max": max(sequence_lengths, default=0),
            "mean": mean(sequence_lengths) if sequence_lengths else 0.0,
            "median": median(sequence_lengths) if sequence_lengths else 0.0,
        },
        "same_kc_future_revisit_rate": _percent(future_revisits, revisit_denominator),
        "question_exposure_count": {
            "min": min(exposures.values(), default=0),
            "max": max(exposures.values(), default=0),
            "mean": mean(exposures.values()) if exposures else 0.0,
            "median": median(exposures.values()) if exposures else 0.0,
        },
        "canonical_checks": {
            "one_sequence_per_student": len(valid_rows) == len(rows_by_student),
            "responses_binary": all(row["response"] in (0, 1) for row in interactions),
            "no_empty_kc_sets": all(row["concepts"] for row in interactions),
            "kc_sets_deduplicated": all(
                len(row["concepts"]) == len(set(row["concepts"])) for row in interactions
            ),
            "timestamps_sorted": all(
                all(rows[index]["timestamp"] <= rows[index + 1]["timestamp"]
                    for index in range(len(rows) - 1))
                for _, rows in valid_rows
            ),
        },
        **metadata_audit,
    }
    audit_path = os.path.join(os.path.dirname(os.path.abspath(write_file)), "preprocess_audit.json")
    _write_audit(audit_path, audit)
    print(
        "MOOC-Radar canonical data: "
        f"students={audit['valid_students']}, questions={audit['valid_questions']}, "
        f"interactions={valid_interactions}, Kmax={audit['question_max_k']}"
    )
    return audit
