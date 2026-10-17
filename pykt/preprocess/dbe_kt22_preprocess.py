"""DBE-KT22 adapter for the shared RAPLKT multi-KC pipeline."""

import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import mean, median

from .utils import replace_text


TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f %z"


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


def _read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_question_kc_metadata(data_dir):
    """Load the official complete question -> KC relation."""
    question_rows = _read_csv(os.path.join(data_dir, "Questions.csv"))
    kc_rows = _read_csv(os.path.join(data_dir, "KCs.csv"))
    relation_rows = _read_csv(os.path.join(data_dir, "Question_KC_Relationships.csv"))
    question_ids = [row.get("id", "").strip() for row in question_rows]
    kc_ids = [row.get("id", "").strip() for row in kc_rows]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Questions.csv contains duplicate question ids.")
    if len(kc_ids) != len(set(kc_ids)):
        raise ValueError("KCs.csv contains duplicate KC ids.")

    known_questions = set(question_ids)
    known_kcs = set(kc_ids)
    question_to_kcs = defaultdict(list)
    duplicate_relations = 0
    seen_relations = set()
    missing_question_references = 0
    missing_kc_references = 0
    for row in relation_rows:
        question = row.get("question_id", "").strip()
        concept = row.get("knowledgecomponent_id", "").strip()
        missing_question_references += question not in known_questions
        missing_kc_references += concept not in known_kcs
        pair = (question, concept)
        if pair in seen_relations:
            duplicate_relations += 1
            continue
        seen_relations.add(pair)
        if question in known_questions and concept in known_kcs:
            question_to_kcs[question].append(replace_text(concept))

    empty_questions = known_questions - set(question_to_kcs)
    if missing_question_references or missing_kc_references or duplicate_relations:
        raise ValueError(
            "Invalid official question-KC metadata: "
            f"missing question references={missing_question_references}, "
            f"missing KC references={missing_kc_references}, "
            f"duplicate relations={duplicate_relations}."
        )
    return dict(question_to_kcs), {
        "metadata_questions": len(question_rows),
        "metadata_kcs": len(kc_rows),
        "metadata_question_kc_relations": len(relation_rows),
        "metadata_referenced_kcs": len({kc for values in question_to_kcs.values() for kc in values}),
        "metadata_empty_kc_questions": len(empty_questions),
        "metadata_duplicate_question_kc_relations": duplicate_relations,
        "metadata_missing_question_references": missing_question_references,
        "metadata_missing_kc_references": missing_kc_references,
    }


def _parse_time(value):
    parsed = datetime.strptime(value, TIME_FORMAT)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def _write_audit(path, audit):
    with open(path, "w", encoding="utf8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def read_data_from_csv(read_file, write_file, question_metadata=None):
    """Convert Transaction.csv into canonical six-line student histories."""
    data_dir = question_metadata or os.path.dirname(os.path.abspath(read_file))
    question_to_kcs, metadata_audit = load_question_kc_metadata(data_dir)
    transactions = _read_csv(read_file)

    raw_students = set()
    raw_questions = set()
    rows_by_student = defaultdict(list)
    removed_missing = 0
    removed_invalid_response = 0
    removed_invalid_timestamp = 0
    removed_no_kc = 0
    hidden_interactions = 0
    negative_duration_records = 0
    exact_counter = Counter()
    candidate_duplicate_counter = Counter()

    for original_index, row in enumerate(transactions):
        student = row.get("student_id")
        question = row.get("question_id")
        response_text = str(row.get("answer_state", "")).strip().lower()
        start_text = str(row.get("start_time", "")).strip()
        end_text = str(row.get("end_time", "")).strip()
        if student is not None and str(student).strip():
            raw_students.add(str(student).strip())
        if question is not None and str(question).strip():
            raw_questions.add(str(question).strip())
        if any(
            value is None or str(value).strip() == ""
            for value in (student, question, row.get("answer_state"), start_text)
        ):
            removed_missing += 1
            continue
        if response_text not in {"true", "false"}:
            removed_invalid_response += 1
            continue
        student = str(student).strip()
        question = str(question).strip()
        concepts = question_to_kcs.get(question)
        if not concepts:
            removed_no_kc += 1
            continue
        try:
            timestamp = _parse_time(start_text)
            end_timestamp = _parse_time(end_text)
        except (TypeError, ValueError):
            removed_invalid_timestamp += 1
            continue
        duration = end_timestamp - timestamp
        if duration < 0:
            negative_duration_records += 1
            duration = 0
        response = 1 if response_text == "true" else 0
        hidden_interactions += str(row.get("is_hidden", "")).strip().lower() == "true"

        candidate_duplicate_counter[(student, question, start_text, response)] += 1
        # Transaction id is the stable attempt identity.  This semantic tuple
        # excludes only that id and detects truly duplicated logged attempts.
        semantic_tuple = tuple(row.get(key, "") for key in row if key != "id")
        exact_counter[semantic_tuple] += 1
        rows_by_student[student].append(
            {
                "question": replace_text(question),
                "concepts": concepts,
                "response": response,
                "timestamp": timestamp,
                "duration": duration,
                "original_index": original_index,
            }
        )

    valid_rows = []
    same_question_repeats = 0
    consecutive_repeats = 0
    for student, rows in rows_by_student.items():
        rows.sort(key=lambda item: (item["timestamp"], item["original_index"]))
        seen_questions = set()
        previous_question = None
        for row in rows:
            question = row["question"]
            same_question_repeats += question in seen_questions
            consecutive_repeats += question == previous_question
            seen_questions.add(question)
            previous_question = question
        valid_rows.append((student, rows))

    os.makedirs(os.path.dirname(os.path.abspath(write_file)), exist_ok=True)
    with open(write_file, "w", encoding="utf8", newline="\n") as stream:
        for student, rows in valid_rows:
            stream.write(f"{replace_text(student)},{len(rows)}\n")
            stream.write(",".join(row["question"] for row in rows) + "\n")
            stream.write(",".join("_".join(row["concepts"]) for row in rows) + "\n")
            stream.write(",".join(str(row["response"]) for row in rows) + "\n")
            stream.write(",".join(str(row["timestamp"]) for row in rows) + "\n")
            stream.write(",".join(str(row["duration"]) for row in rows) + "\n")

    interactions = [row for _, rows in valid_rows for row in rows]
    question_kcs = {row["question"]: tuple(row["concepts"]) for row in interactions}
    question_counts = [len(concepts) for concepts in question_kcs.values()]
    interaction_counts = [len(row["concepts"]) for row in interactions]
    incorrect_counts = [len(row["concepts"]) for row in interactions if row["response"] == 0]
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
                future_revisits += bool(concepts & future_concepts)
            future_concepts.update(concepts)
            exposures[row["question"]] += 1

    valid_interactions = len(interactions)
    incorrect_total = len(incorrect_counts)
    incorrect_multi = incorrect_distribution["k2"] + incorrect_distribution["k3plus"]
    exact_duplicates = sum(count - 1 for count in exact_counter.values() if count > 1)
    candidate_duplicates = sum(
        count - 1 for count in candidate_duplicate_counter.values() if count > 1
    )
    audit = {
        "dataset": "dbe_kt22",
        "source_files": {
            "interactions": os.path.basename(read_file),
            "questions": "Questions.csv",
            "kcs": "KCs.csv",
            "question_kc_metadata": "Question_KC_Relationships.csv",
        },
        "schema": {
            "student_field": "Transaction.csv student_id",
            "question_field": "Transaction.csv question_id",
            "kc_field": "Question_KC_Relationships.csv knowledgecomponent_id",
            "response_field": "Transaction.csv answer_state",
            "timestamp_field": "Transaction.csv start_time",
            "duration_fields": ["start_time", "end_time"],
            "attempt_identity": "Transaction.csv id",
            "auxiliary_fields": [
                "selection_change", "difficulty_feedback", "trust_feedback",
                "answer_choice_id", "hint_used", "is_hidden",
            ],
        },
        "question_identity": "question_id; unique in the single Questions.csv table",
        "kc_set_rule": "official Question_KC_Relationships.csv complete relation",
        "raw_students": len(raw_students),
        "raw_questions": len(raw_questions),
        "raw_interactions": len(transactions),
        "valid_students": len(valid_rows),
        "valid_questions": len(question_kcs),
        "valid_interactions": valid_interactions,
        "num_kcs": len({concept for row in interactions for concept in row["concepts"]}),
        "missing_records": removed_missing,
        "invalid_response_records": removed_invalid_response,
        "invalid_timestamp_records": removed_invalid_timestamp,
        "empty_kc_questions": metadata_audit["metadata_empty_kc_questions"],
        "empty_kc_interactions": removed_no_kc,
        "duplicate_records": exact_duplicates,
        "candidate_duplicate_key_collisions": candidate_duplicates,
        "hidden_interactions_retained": hidden_interactions,
        "negative_duration_records_clamped_to_zero": negative_duration_records,
        "same_student_same_question_repeat_count": same_question_repeats,
        "same_student_same_question_repeat_rate": _ratio(same_question_repeats, valid_interactions),
        "same_student_same_question_consecutive_repeat_count": consecutive_repeats,
        "same_student_same_question_consecutive_repeat_rate": _ratio(consecutive_repeats, valid_interactions),
        "question_kc_conflicts": 0,
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
        "DBE-KT22 canonical data: "
        f"students={audit['valid_students']}, questions={audit['valid_questions']}, "
        f"interactions={valid_interactions}, Kmax={audit['question_max_k']}"
    )
    return audit

