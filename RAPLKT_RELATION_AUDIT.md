# RAPLKT Item-side Relation Audit

Base: V1.6 formal RAPLKT.

Imported conceptually from V1.5:

- item relation embedding;
- KC relation embedding;
- cosine relevance score;
- masked softmax.

`alpha_relation` is item-side KC relevance derived from the unchanged cosine Relation score. The final item-side weight is `(1 - relation_mix_lambda) * alpha_mean + relation_mix_lambda * alpha_relation`. The lambda is a fixed `[0, 1]` hyperparameter and is not trainable. It affects the current item KC representation and current item KC-difficulty representation only.

`rho` is student-specific failure responsibility: it answers which candidate KC most likely contributed to this student's observed failure. For incorrect responses it is `softmax(log(1-p) / responsibility_temperature)`; temperature `1.0` recovers `normalize(1-p)`. Correct responses remain uniform. It alone routes Beta failure credit, historical interaction and historical interaction difficulty.

Relation does not affect:

- the responsibility formula;
- Beta evidence credit;
- historical routing;
- historical interaction difficulty routing;
- set-level partial-label loss;
- the incorrect-response failure budget.

The evidence budget is unchanged; only the sharpness of incorrect responsibility is configurable:

```text
wrong responsibility: rho = softmax(log(1 - p) / responsibility_temperature)
correct responsibility: uniform
correct update: +1/K for every required KC (total success budget 1)
wrong update: rho, total budget = 1
```

There is no responsibility prior, pi, effective-K credit, copied KC BCE, sequential Beta update, Relation auxiliary loss, dynamic gate, or runtime aggregation switch. Responsibility temperature is a fixed positive hyperparameter and is not trainable.

The checkpoint protocol is `responsibility_partial_label_fixed_relation_mix_unit_evidence_v4`. Training must start from scratch; older gated Relation checkpoints are intentionally incompatible with this protocol.

Tested invariants include lambda endpoints and interpolation, exact K=1 weights, padding, normalization, response and history independence, unchanged cosine scoring, mixed question/current-difficulty representation, rho-routed history and history difficulty, unchanged responsibility/Beta updates, set-level likelihood, causality, vectorization, and the absence of a trainable lambda parameter.

Verification on 2026-09-08:

- Relation and benchmark suite: 27 passed, 5 subtests passed.
- Full project suite: 29 passed, 5 subtests passed.
- Smoke training: two epochs, backward, validation and checkpoint save passed.
- Smoke prediction: strict load, test/window metrics, Relation diagnostics and responsibility diagnostics passed.
- Forbidden Relation/responsibility coupling search in active model and entrypoints: zero matches.
