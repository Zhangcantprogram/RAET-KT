# RAPLKT cleanup inventory

Inventory taken before cleanup on 2026-09-07 from branch `V1.5`, HEAD
`495718b444d5957710b6e8944f626f49779f6b3d`.

## KEEP

- `pykt/datasets/init_raplkt_dataset.py`: formal multi-KC train/test loader entrypoints.
- `pykt/datasets/raplkt_data_loader.py`: collapse/preserve protocol, multi-KC masks,
  evidence update mask, benchmark identity and compatible `*_raplkt_*_v7_*.pkl` cache names.
- `pykt/models/akt.py`: shared AKT implementation used by other models.
- `pykt/models/item_level_metrics.py`: generic item/K-group metrics and benchmark manifests.
- `pykt/preprocess/assist2017_preprocess.py` and shared preprocessing: required dataset logic.
- `tests/test_assist2017_preprocess.py`, `tests/test_benchmark_protocol.py`: shared protocol tests.
- Checkpoints, `results/`, WandB/log/CSV/XLSX research outputs and archives including
  `RAPLKT.zip`, `RAPLKT_ablation.zip`, and `raplkt_algebra2005_5fold.xlsx`.

## RENAME / REPLACE FROM SOURCE OF TRUTH

- `pykt/models/raplkt_core_set.py` -> replace `pykt/models/raplkt.py`;
  rename class `RAPLKTCoreSet` to `RAPLKT`, model name to `raplkt`.
- `pykt/models/raplkt_core_set_train_model.py` -> replace
  `pykt/models/raplkt_train_model.py`; rename public functions without changing loss/evaluation.
- `examples/wandb_raplkt_core_set_train.py` -> replace
  `examples/wandb_raplkt_train.py`; remove the migration-only `--full_config`, retain
  `set_loss_weight`, data protocol and standard KT hyperparameters.
- `examples/wandb_raplkt_core_set_predict.py` -> replace
  `examples/wandb_raplkt_predict.py`; accept only the final `raplkt` identity and strict-load.
- Formal Full-Set correctness/equivalence tests -> consolidate into `tests/test_raplkt.py`.
- Replace experimental audits with one `RAPLKT_AUDIT.md`.

## DELETE

Model/trainer variants:

- `pykt/models/raplkt_core.py`, `raplkt_core_train_model.py`
- `pykt/models/raplkt_core_t.py`, `raplkt_core_t_train_model.py`
- `pykt/models/raplkt_core_set.py`, `raplkt_core_set_train_model.py` after promotion
- `pykt/models/raplkt_core_uniform.py`, `raplkt_core_uniform_train_model.py`
- `pykt/models/raplkt_core_uniform_set.py`, `raplkt_core_uniform_set_train_model.py`
- `pykt/models/raplkt_core_uniform_evidence.py`,
  `raplkt_core_uniform_evidence_train_model.py`

Experimental entrypoints and comparison helpers:

- all `examples/wandb_raplkt_core_*.py`
- `examples/compare_raplkt_core_uniform.py`
- `examples/compare_core_set_vs_uniform_set.py`
- `examples/compare_core_set_three_way.py`

Experimental/old RAPLKT tests:

- all `tests/test_raplkt_core*.py`
- old V1.5 design tests: `tests/test_raplkt_engineering.py`,
  `test_raplkt_evaluation.py`, `test_raplkt_evidential.py`,
  `test_raplkt_relation.py`, `test_raplkt_training.py`
- `tests/benchmark_raplkt.py` (old experimental benchmark script)

Experimental documents:

- `RAPLKT_CORE_AUDIT.md`, `RAPLKT_CORE_SET_AUDIT.md`, `RAPLKT_CORE_T2_AUDIT.md`
- `RAPLKT_CORE_UNIFORM_AUDIT.md`, `RAPLKT_CORE_UNIFORM_SET_AUDIT.md`
- `RAPLKT_CORE_UNIFORM_EVIDENCE_AUDIT.md`

## UPDATE REFERENCES

- `README.md`: remove relation/router/temperature/evidential-ablation documentation and
  replace with the sole formal model, training and prediction commands.
- `pykt/models/__init__.py`: add only `from .raplkt import RAPLKT`; no variant exports.
- Formal scripts/config JSON written at run time: identity `raplkt`, set-level partial-label
  enabled, set loss weight 0.1, responsibility rule `normalized_failure_evidence`.
- `tests/test_raplkt.py`: import only final model/trainer and retain mathematical, gradient,
  causal, checkpoint, smoke and benchmark correctness checks.

## UNKNOWN / PRESERVE

- `RAPLKT.zip`, `RAPLKT_ablation.zip`: archives may contain historical source but are not
  active Python source; preserve as research records.
- `raplkt_algebra2005_5fold.xlsx`, `results/`, any saved model/log/WandB directories:
  experimental evidence; preserve.
- No RAPLKT-specific entries were found in `pykt/models/init_model.py`,
  `train_model.py`, `evaluate_model.py`, or static JSON configs during inventory.
