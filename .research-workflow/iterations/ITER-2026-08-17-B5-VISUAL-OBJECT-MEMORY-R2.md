### ITER-2026-08-17-B5-VISUAL-OBJECT-MEMORY-R2 — Competitive cross-replan object memory

- 时间：2026-08-17T07:01:23+08:00
- 类型：Code Config Experiment
- 动机：R1 completed finite eight-B200 optimization but collapsed all eight slots and failed to distinguish aligned history from zeroed or shuffled history.
- 变更：Replaced independent query pooling with slot-competitive assignment and fixed spatial anchors; added a masked slot-diversity objective and configurable memory gate; allowed the existing strong action expert to adapt during the 500-step probe while the video expert remained frozen. Acceptance required finite eight-B200 optimization, a complete checkpoint, slot mean absolute off-diagonal cosine below 0.95, and aligned-history action loss at least 1 percent lower than both matched interventions.
- Git：207246a5e06c7c3219c78c5eb61fd5266bde3a36 on exp/cosmos3-b5-visual-object-memory-20260817, clean and pushed.
- 实验：Experiment CWAM-COSMOS3-B5-VOM-PROBE-R2-20260817; Job cwam-b5-vom-probe-r2; train Run CWAM-COSMOS3-B5-VOM-PROBE-R2-B4WARM-500S-20260817; validation Runs CWAM-COSMOS3-B5-VOM-PROBE-R2-VAL-NONE-20260817, CWAM-COSMOS3-B5-VOM-PROBE-R2-VAL-ZERO-20260817, and CWAM-COSMOS3-B5-VOM-PROBE-R2-VAL-SHUFFLE-20260817.
- 结果：The exact container passed 72 focused tests and a B200 BF16 production-shape forward/backward smoke test. The eight-B200 probe completed 500 finite optimizer steps and saved iter_000000500. Slot mean absolute off-diagonal cosine fell from 0.9986472 to 0.0073938 and diversity loss fell from 0.9972963 to 0.0000940. On the identical 2048-sample held-out stream, action loss was aligned 0.0082494505, zero 0.0082474817, and shuffle 0.0082493899. Aligned memory was 0.02387 percent worse than zero and 0.00073 percent worse than shuffle.
- 决策：Failed the predeclared representation-use gate. R2 proves that competitive assignment and the diversity objective prevent slot collapse, but it does not prove that the action expert uses sample-specific cross-replan memory. The formal 20k eight-GPU run was not launched.
- 下一步：Before another full run, test a short R3 probe that makes memory usage structurally necessary or explicitly supervised, for example memory-conditioned residual FiLM/cross-attention with a learned relevance gate plus temporal correspondence or future-action/object-state prediction. Keep the same none/zero/shuffle causal gate.
- 记录人：Codex
