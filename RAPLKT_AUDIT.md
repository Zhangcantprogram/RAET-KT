# RAPLKT Formal Implementation Audit

Audit date: 2026-09-08

## Formal implementation

The active implementation has one public identity: `pykt.models.raplkt.RAPLKT` with `model_name="raplkt"` and architecture version `responsibility_partial_label_fixed_relation_mix_unit_evidence_v4`.

Its behavior extends the former final Full model with a backward-compatible responsibility temperature:

- multi-KC collapse protocol and fixed-lambda mean/Relation question representation;
- two calls to one shared causal AKT module;
- KC-local contextual success evidence `p_k`;
- stable set-level partial-label likelihood;
- temperature-scaled incorrect responsibility `rho = softmax(log(1 - p) / T)` and correct uniform routing; `T=1` recovers `normalize(1-p)`;
- correct credit `1/K` per required KC and incorrect credit `rho`, so both outcomes have total evidence budget 1 per item;
- parallel `scatter_add` plus exclusive cumulative sum Beta history;
- responsibility-aware historical interaction, Rasch difficulty and final KT prediction;
- end-to-end gradients through KC evidence, responsibility and Beta history.

Relation weights affect only the current item representation and current item difficulty. Student-side responsibility uses the fixed positive hyperparameter `responsibility_temperature` and exclusively controls failure credit and historical routing. The active model contains no prior responsibility, copied KC-label BCE, effective-K credit, experimental mode switch, or sequence-wise Python recurrence.

The checkpoint protocol remains `responsibility_partial_label_fixed_relation_mix_unit_evidence_v4`: the temperature adds no trainable parameter, old configs default to `1.0`, and that default executes the original normalization path exactly.

## Files consolidated

- `pykt/models/raplkt_core_set.py` became `pykt/models/raplkt.py` with class `RAPLKT`.
- `pykt/models/raplkt_core_set_train_model.py` became `pykt/models/raplkt_train_model.py`.
- The corresponding train and predict entrypoints became `examples/wandb_raplkt_train.py` and `examples/wandb_raplkt_predict.py`.
- Formal correctness coverage was consolidated into `tests/test_raplkt.py`.

All superseded model, trainer, entrypoint, comparison, test and experimental audit files were deleted from the active tree. Their last complete state is recoverable from local Git commit `a9452c4`.

## Registry and configuration

`pykt/models/__init__.py` exports only `RAPLKT` for this model family. No experimental dispatch branches or experimental configuration entries were present after consolidation. The formal scripts persist:

```json
{
  "model_name": "raplkt",
  "set_level_partial_label": true,
  "set_loss_weight": 0.1,
  "responsibility_rule": "normalized_failure_evidence",
  "question_aggregation": "fixed_mean_relation_mix"
}
```

## Verification

- Active-source residual search in `pykt/`, `examples/`, `configs/` and `tests/`: 0 matches.
- Formal import: passed. Removed experimental module lookup: absent as expected.
- Old mean-pooling checkpoints are rejected by the Relation architecture version and strict state-dict load, as intended.
- Causality, exclusive prefix, evidence budget, gradient and set-likelihood tests: passed.
- Smoke train: two epochs, backward pass, validation and checkpoint save passed.
- Smoke predict: strict checkpoint load, test/window evaluation, responsibility and set-level diagnostics passed.
- Relation-specific verification and current test results are recorded in `RAPLKT_RELATION_AUDIT.md`.
- Base trainable parameters for Assist-style `n_question=101`, `n_pid=3162`, `d_model=256`, `d_ff=256`, 4 heads and 2 blocks: 2,694,517; Relation adds 104,480 embedding parameters. The fixed mixing lambda adds no parameter.

Only the final full RAPLKT implementation remains in active source code. All RAPLKT ablation implementations and experimental mode switches have been removed from `pykt/`, `examples/`, `configs/`, and `tests/`. The responsibility and Beta backbone remains equivalent to the former final Full model; current-item aggregation now uses the formal item-side Relation module documented above.
