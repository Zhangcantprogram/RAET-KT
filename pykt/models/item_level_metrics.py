import csv

import numpy as np
import pandas as pd
from sklearn import metrics


KC_GROUPS = {
    "overall": lambda k: np.ones_like(k, dtype=bool),
    "k1": lambda k: k == 1,
    "k2": lambda k: k == 2,
    "k3plus": lambda k: k >= 3,
    "multi": lambda k: k >= 2,
}

MANIFEST_COLUMNS = [
    "target_cidx",
    "question_id",
    "response",
    "kc_ids",
    "k_count",
    "prediction",
]


def compute_item_level_metrics(y_true, y_score, k_count):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    k_count = np.asarray(k_count).reshape(-1)
    if not (len(y_true) == len(y_score) == len(k_count)):
        raise AssertionError(
            "Prediction, target, and KC-count lengths must match: "
            f"{len(y_score)}, {len(y_true)}, {len(k_count)}"
        )

    results = {}
    for group_name, selector in KC_GROUPS.items():
        selected = selector(k_count)
        group_true = y_true[selected]
        group_score = y_score[selected]
        n = int(group_true.size)
        pos = int((group_true == 1).sum())
        neg = int((group_true == 0).sum())
        if n == 0:
            auc = float("nan")
            acc = float("nan")
        else:
            predicted = (group_score >= 0.5).astype(np.int64)
            acc = float(metrics.accuracy_score(group_true, predicted))
            auc = (
                float("nan")
                if np.unique(group_true).size < 2
                else float(metrics.roc_auc_score(group_true, group_score))
            )
        results[group_name] = {
            "n": n,
            "pos": pos,
            "neg": neg,
            "auc": auc,
            "acc": acc,
        }
    return results


def print_item_level_metrics(results, split_name):
    is_window = str(split_name).lower() in {"window", "window_test"}
    suffix = "Window Test" if is_window else "Test"
    titles = {
        "overall": f"Overall {suffix}",
        "k1": f"Single-KC {suffix} (K=1)",
        "k2": f"2-KC {suffix}",
        "k3plus": f"3+-KC {suffix}",
        "multi": f"Multi-KC {suffix} (K>=2)",
    }
    for group_name in ["overall", "k1", "k2", "k3plus", "multi"]:
        values = results[group_name]
        print(f"{titles[group_name]}:")
        print(
            f"    N={values['n']} Pos={values['pos']} Neg={values['neg']} "
            f"AUC={values['auc']:.6f} ACC={values['acc']:.6f}"
        )
    compact_name = "WINDOW" if is_window else "TEST"
    print(
        f"{compact_name}: overall_auc={results['overall']['auc']:.6f} "
        f"k1_auc={results['k1']['auc']:.6f} "
        f"k2_auc={results['k2']['auc']:.6f} "
        f"k3p_auc={results['k3plus']['auc']:.6f} "
        f"multi_auc={results['multi']['auc']:.6f} "
        f"n_k1={results['k1']['n']} n_k2={results['k2']['n']} "
        f"n_k3p={results['k3plus']['n']} n_multi={results['multi']['n']}"
    )


def write_benchmark_manifest(path, records):
    with open(path, "w", encoding="utf8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


def _manifest_frame(value):
    if isinstance(value, (str, bytes)):
        return pd.read_csv(value, sep="\t")
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame(value, columns=MANIFEST_COLUMNS)


def _canonical_manifest(value, side):
    frame = _manifest_frame(value)
    identity_columns = [
        "target_cidx", "question_id", "response", "kc_ids", "k_count"
    ]
    for column in identity_columns:
        if column not in frame:
            raise AssertionError(
                f"{side} benchmark manifest is missing column: {column}."
            )

    frame = frame.copy()
    for column in ["target_cidx", "question_id", "response", "k_count"]:
        try:
            numeric = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise AssertionError(
                f"{side} benchmark manifest column {column} must be integer-valued."
            ) from error
        if not np.all(np.equal(numeric, np.floor(numeric))):
            raise AssertionError(
                f"{side} benchmark manifest column {column} must be integer-valued."
            )
        frame[column] = numeric.astype(np.int64)

    if (frame["target_cidx"] < 0).any():
        raise AssertionError(
            f"{side} benchmark manifest target_cidx must be non-negative."
        )
    duplicated_targets = frame.loc[
        frame["target_cidx"].duplicated(keep=False), "target_cidx"
    ].tolist()
    if duplicated_targets:
        raise AssertionError(
            f"{side} benchmark manifest target_cidx must be unique; "
            f"duplicates={sorted(set(duplicated_targets))[:5]}."
        )
    if not frame["response"].isin([0, 1]).all():
        raise AssertionError(
            f"{side} benchmark manifest response must be in {{0, 1}}."
        )

    canonical_kc_ids = []
    for row_index, (kc_text, k_count) in enumerate(
            zip(frame["kc_ids"], frame["k_count"])):
        if pd.isna(kc_text):
            raise AssertionError(
                f"{side} benchmark manifest kc_ids is empty at row {row_index}."
            )
        try:
            kc_ids = [int(value) for value in str(kc_text).split(",") if value != ""]
        except ValueError as error:
            raise AssertionError(
                f"{side} benchmark manifest kc_ids must be comma-separated integers "
                f"at row {row_index}."
            ) from error
        unique_kc_ids = set(kc_ids)
        if len(kc_ids) != len(unique_kc_ids):
            raise AssertionError(
                f"{side} benchmark manifest kc_ids contains duplicates at row "
                f"{row_index}: {kc_text}."
            )
        if int(k_count) != len(unique_kc_ids):
            raise AssertionError(
                f"{side} benchmark manifest k_count disagrees with unique KC count "
                f"at row {row_index}: k_count={k_count}, kc_ids={kc_text}."
            )
        canonical_kc_ids.append(",".join(str(value) for value in sorted(unique_kc_ids)))
    frame["kc_ids"] = canonical_kc_ids
    return frame.sort_values("target_cidx", kind="stable").reset_index(drop=True)


def assert_same_benchmark_targets(left, right):
    left = _canonical_manifest(left, "left")
    right = _canonical_manifest(right, "right")
    identity_columns = [
        "target_cidx", "question_id", "response", "kc_ids", "k_count"
    ]
    if len(left) != len(right):
        raise AssertionError(
            f"Benchmark target count differs: left={len(left)}, right={len(right)}."
        )
    for column in identity_columns:
        left_values = left[column].tolist()
        right_values = right[column].tolist()
        if left_values != right_values:
            mismatch = next(
                index for index, pair in enumerate(zip(left_values, right_values))
                if pair[0] != pair[1]
            )
            raise AssertionError(
                f"Benchmark target mismatch in {column} at row {mismatch}: "
                f"left={left_values[mismatch]}, right={right_values[mismatch]}."
            )
    return True
