import hashlib
import json

import pandas as pd


BENCHMARK_PROTOCOL_VERSION = "item_multikc_window_v1"


def normalize_raw_q2cs_map(q2cs_map, max_concepts):
    """Normalize raw 0-based question/KC metadata for model-independent use."""
    normalized = {}
    if q2cs_map is None:
        return normalized
    for question, concepts in q2cs_map.items():
        question = int(question)
        clean = sorted({int(concept) for concept in concepts if int(concept) >= 0})
        clean = clean[:max_concepts]
        if clean:
            normalized[question] = clean
    return normalized


def q2cs_mapping_signature(q2cs_map, max_concepts, hash_length=10):
    normalized = normalize_raw_q2cs_map(q2cs_map, max_concepts)
    if not normalized:
        return "visible"
    payload = [[question, normalized[question]] for question in sorted(normalized)]
    serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:hash_length]
    return f"qmap_{digest}"


def _parse_int_list(value):
    if pd.isna(value):
        return []
    return [int(float(item)) for item in str(value).split(",") if str(item) != ""]


def build_raw_q2cs_map(file_path, folds=None, max_concepts=10**9, pad_val=-1):
    """Build raw 0-based question -> KC sets without reading response labels."""
    header = pd.read_csv(file_path, nrows=0).columns
    required = {"questions", "concepts", "is_repeat"}
    missing = required - set(header)
    if missing:
        raise ValueError(f"Missing q2cs metadata columns: {sorted(missing)}")
    usecols = [column for column in ["fold", "questions", "concepts", "is_repeat"] if column in header]
    df = pd.read_csv(file_path, usecols=usecols)
    if folds is not None:
        if "fold" not in df.columns:
            raise ValueError("fold filtering requested but the metadata file has no fold column.")
        df = df[df["fold"].isin(folds)]

    collected = {}
    for _, row in df.iterrows():
        questions = _parse_int_list(row["questions"])
        concepts = _parse_int_list(row["concepts"])
        repeats = _parse_int_list(row["is_repeat"])
        n = min(len(questions), len(concepts), len(repeats))
        index = 0
        while index < n:
            if questions[index] == pad_val or concepts[index] == pad_val:
                index += 1
                continue
            if repeats[index] != 0:
                raise ValueError(
                    "Original q2cs metadata cannot start inside a multi-KC item: "
                    f"row={row.name}, expanded_index={index}, is_repeat={repeats[index]}."
                )
            question = questions[index]
            item_concepts = []
            end = index
            while end < n:
                if end > index and repeats[end] != 1:
                    break
                if questions[end] != question:
                    break
                concept = concepts[end]
                if concept != pad_val:
                    item_concepts.append(concept)
                end += 1
            collected.setdefault(question, []).extend(item_concepts)
            index = max(end, index + 1)
    return normalize_raw_q2cs_map(collected, max_concepts)
