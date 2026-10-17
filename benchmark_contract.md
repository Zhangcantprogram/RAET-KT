# Fair Benchmark Contract

This document freezes only the experimental protocol needed for fair comparison across KT models. It does not summarize or require any RAPLKT model innovation.

## Identity and fixed parameters

- Protocol version: `item_multikc_window_v1`
- Repository/branch: `Zhangcantprogram/RAPLKT`, `V1.4`
- V1.4 source commit at contract generation: `679c3bfd54d5785a2a472af621b165bfac09bbe9`
- RAPLKT preprocessing cache: `v7`; AKT preprocessing cache: `v5`
- `min_seq_len=3`, `maxlen=200`, `kfold=5`, `random_state=1024`
- Student split: shuffle student rows with `DataFrame.sample(frac=1.0, random_state=1024)`, use the first `N-floor(0.2N)` students for train+validation and the final `floor(0.2N)` for Test.
- Fold assignment: reshuffle train+validation student rows with the same random state, divide them into folds `0..4`, and give remainder students to the earliest folds. For fold `f`, validation is exactly fold `f`; training is the other four folds. Test rows have `fold=-1`.

Rebuild either dataset from `examples/`:

```bash
python data_preprocess.py -d algebra2005 -m 3 -l 200 -k 5
python data_preprocess.py -d assist2017 -m 3 -l 200 -k 5
```

## Shared preprocessing and file contract

Responses must be binary `{0,1}`. After dataset-specific filtering, the shared pipeline computes the dataset maximum KC count, expands multi-KC items, then maps user/question/KC strings to deterministic raw 0-based IDs in first-occurrence order. `keyid2idx.json` stores those mappings and `max_concepts`.

For a K-KC item, expansion creates K consecutive positions: the first has `is_repeat=0`; the remaining K-1 positions have `is_repeat=1`. Question, response, timestamp, and use-time values are repeated across the block. `train_valid_sequences.csv` and `test_sequences.csv` use item-boundary-safe segmentation: no block may cross a `maxlen=200` boundary, an item longer than 200 is an error, and a final segment shorter than 3 expanded positions is dropped.

The required files are:

- `train_valid_sequences.csv`: non-overlapping item-boundary-safe train/validation segments, with `fold`, `selectmasks`, and `is_repeat`.
- `test.csv`: complete held-out expanded-position student histories before segmentation, with `fold=-1` and globally unique expanded-position `cidxs`.
- `test_sequences.csv`: non-overlapping item-boundary-safe Test segments; short tails may be absent.
- `test_window_sequences.csv`: expanded-position windows. For a history longer than 200, the first 200-position window selects its eligible positions; each subsequent one-position slide contains the last 200 expanded positions and selects only the final position. This is explicitly not an item-level window.
- `keyid2idx.json`: raw string-to-0-based-ID maps and `max_concepts`.

## Multi-KC loading and target contract

Evaluation uses `sequence_mode=collapse`. One `rep=0` plus its following `rep=1` positions becomes one item with one unique candidate KC set. Window q2cs metadata must be built from complete `test.csv`, using only `fold`, `questions`, `concepts`, and `is_repeat`; the builder is forbidden to read `responses`. If q2cs has already supplied the complete KC set, visible repeats must not append a KC already present.

A window that starts at `is_repeat=1` contains an orphan partial item. It may update history but must never be a target. Raw q2cs and manifest IDs are 0-based. AKT uses raw IDs directly; RAPLKT adds 1 internally to reserve 0 for padding and subtracts 1 when exporting. This internal representation difference must not change manifest identity.

Prediction uses next-item shift `1`: positions `0..S-2` predict positions `1..S-1`. `smasks` are shifted target-selection masks intersected with valid adjacent positions. Padding, collapsed repeat positions, and orphan repeats are excluded. Every eligible item contributes exactly one scalar probability; KC-level outputs must never be counted as separate examples. K is the number of valid entries in the complete shifted candidate-KC mask—not visible-window KCs and not selected responsibility support.

## Metrics, loaders, and model selection

Concatenate all eligible item targets globally before calculating metrics. ROC-AUC is one global item-level `sklearn.metrics.roc_auc_score(response, probability)`. ACC predicts 1 exactly when `probability >= 0.5`.

The reporting groups are:

- Overall: all eligible items
- K1: `K == 1`
- K2: `K == 2`
- K3+: `K >= 3`
- Multi: `K >= 2`

Training loaders use `shuffle=True`. Validation, Test, and Window Test loaders use `shuffle=False`.

Checkpoint selection uses validation AUC only. Every strict improvement `valid_auc > true_best_auc` must be saved even when smaller than `early_stop_min_delta`. That delta controls only the independent patience reference. Test and Window Test labels or metrics are forbidden in checkpointing, early stopping, hyperparameter tuning, and model selection.

## Window manifest contract

Every model writes `window_test_manifest.tsv`, tab-delimited, with columns:

| Column | Contract |
|---|---|
| `target_cidx` | Required non-negative, globally unique expanded-position ID of the `rep=0` target. `shft_cidxs` must exist; fallback `-1` is forbidden. |
| `question_id` | Raw 0-based question ID. |
| `response` | Integer 0 or 1. |
| `kc_ids` | Comma-separated raw 0-based unique KC IDs; writer output is sorted. |
| `k_count` | Exactly the number of unique IDs in `kc_ids`. |
| `prediction` | One model-specific item-correctness probability; excluded from target-identity comparison. |

`assert_same_benchmark_targets` must enforce all of the following before comparing models:

1. Equal row counts and all five identity columns present.
2. Integer `target_cidx`, non-negative and unique within each manifest.
3. Canonical sort by `target_cidx` before comparison.
4. `response` in `{0,1}`.
5. Integer `kc_ids` with no duplicates; canonicalize each set by sorting.
6. `k_count == number of unique KC IDs`.
7. Exact row-wise equality of `target_cidx`, `question_id`, `response`, `kc_ids`, and `k_count` after canonicalization.

## External FAKT/KT adapter contract

An external model that lacks native multi-KC input may add only a minimal item adapter. It must retain every KC in the complete candidate set and combine them through a response-free, permutation-symmetric operation. Acceptable examples include a masked mean, a normalized masked sum, or permutation-invariant attention whose query and weights do not use the current response.

It is forbidden to keep only the first KC, use the current response/label or future interactions to construct the input, alter splits or target identities, or emit one evaluated probability per KC.

## Complete dataset contract: Algebra2005

Dataset name is `algebra2005`; raw input is `data/algebra2005/algebra_2005_2006_train.txt`. Resolved values are `min_seq_len=3`, `maxlen=200`, `kfold=5`, `max_concepts=7`, `num_questions=173113`, `num_concepts=112`, student split `80/20`, folds `0..4`, `sequence_mode=collapse`, and shift `1`.

Dataset-specific preprocessing is fixed as follows:

1. Read the UTF-8 tab-separated source with `low_memory=False`.
2. Sanitize `Problem Name` and `Step Name`; define the question as `<Problem Name>----<Step Name>`.
3. Drop rows missing `Anon Student Id`, derived `Questions`, `KC(Default)`, `First Transaction Time`, or `Correct First Attempt`.
4. Retain only `Correct First Attempt` in `{0,1}`.
5. Sanitize `KC(Default)` and convert its `~~` multi-KC separator to `_` before shared expansion.
6. Group students with `sort=False`; within each student sort by converted `First Transaction Time`, then original source-row index.
7. Apply the complete shared split, fold, expansion, item-safe segmentation, expanded Window, q2cs, shift/smask, metric, loader, checkpoint, manifest, and adapter rules above without modification.

Artifact SHA256 values refer to the current generated files. `sha256_bytes` checks the exact Windows CRLF bytes. `sha256_lf_canonical` first converts CRLF to LF and is the cross-platform content check.

| File | Rows | Bytes | SHA256 bytes | SHA256 LF-canonical |
|---|---:|---:|---|---|
| `train_valid_sequences.csv` | 3,980 | 22,023,391 | `b1824ce7751f36a6d98b969f1d2dac74f3151afc54448b8423d86e826611ea00` | `06c88114094366023951bb21f4b3bf9c66900bf70083701fccac97b385434da0` |
| `test.csv` | 114 | 4,387,818 | `f67d2eee8e388a168bdab469a7adaeef067d2fe826e6ca48e4fd9d9e328c9858` | `c9b6a040a3391e3a539d5e1ae9debbd3edb463d9fcc365d39609dd5db19badcf` |
| `test_sequences.csv` | 732 | 4,903,087 | `c6c6f95dd3cf260488e12d116681dd08034dabbed93d4487504cbee9ac224253` | `8e902c7bc18e089c01c01a325f78ba8e91d6392144c43701ef9709f89085f1a6` |
| `test_window_sequences.csv` | 114,220 | 810,463,802 | `5c48203675f4d5e365a925940d528ba9e79d447d7c1a866a67ccc5487cd5f1cb` | `4a6d2d257f7f9126baa2280963fe6af2ac86bc02fcf014cb891cf71f166d8a96` |
| `keyid2idx.json` | — | 5,954,281 | `9a9dcf9c81f3811fb4243953fb9fd506446e3247924df8320dec3ad0cdd43069` | same |

## Complete dataset contract: ASSIST2017

Dataset name is `assist2017`; raw input is `data/assist2017/anonymized_full_release_competition_dataset.csv`. Resolved values are `min_seq_len=3`, `maxlen=200`, `kfold=5`, `max_concepts=2`, `num_questions=2210`, `num_concepts=101`, student split `80/20`, folds `0..4`, `sequence_mode=collapse`, and shift `1`.

Dataset-specific preprocessing is fixed as follows:

1. Read the UTF-8 CSV with `low_memory=False`.
2. Drop rows missing `studentId`, `problemId`, `correct`, `skill`, or `startTime`; retain only `correct` in `{0,1}`.
3. For every `problemId`, build the deterministic first-appearance-order union of sanitized real skill values across the filtered data.
4. Exclude `noskill` case-insensitively. If a problem has no real skill after exclusion, remove all its interactions.
5. Attach the same complete underscore-joined union set to every remaining interaction of that problem before shared expansion.
6. Convert `timeTaken` to rounded milliseconds and `startTime` to integer milliseconds.
7. Group students with `sort=False`; within each student sort by `startTime`, then original source-row index.
8. Apply the complete shared split, fold, expansion, item-safe segmentation, expanded Window, q2cs, shift/smask, metric, loader, checkpoint, manifest, and adapter rules above without modification.

| File | Rows | Bytes | SHA256 bytes | SHA256 LF-canonical |
|---|---:|---:|---|---|
| `train_valid_sequences.csv` | 5,793 | 38,236,388 | `f5c6b4e1851dbc549729b5a26696a87b373b860eb3a29812a8bf26d0204e65b0` | `555e62de7d20b95bca4e91b32bed646e3b2897416cf62a2822b7e8d3c5cb2387` |
| `test.csv` | 341 | 9,352,634 | `5ad15f4aebe0ffebda0760e504a7cb224f787de3820b45a12fc4efb6c729583f` | `c40034e8e2123b041fa716f5787b70ea28fd0ab903a51ede054ba3a40569bb4d` |
| `test_sequences.csv` | 1,372 | 10,691,594 | `0a1e40eda51849607f36761685c7b09c90e6a32ad12f19eec1e852b76e8fe4d4` | `37dc880b80ea7f1407d549cea60f846f67816d1ed52246c31caa98d01d9034fa` |
| `test_window_sequences.csv` | 175,066 | 1,470,120,034 | `23c9bfe1a839385c8e827a7343ac4f577c90235d79b463369be58a81b79c475c` | `8d9dc3687759ecb5d170720a28d4cb8be668fce69fbe9c4966ebd77a78c94ec8` |
| `keyid2idx.json` | — | 54,145 | `5def52ea84dfde4c9c5b9be60a0203ed997a1ffde68f0457533d3b1786cb80e4` | same |

The machine-readable form of this complete contract is `benchmark_contract.json`.
