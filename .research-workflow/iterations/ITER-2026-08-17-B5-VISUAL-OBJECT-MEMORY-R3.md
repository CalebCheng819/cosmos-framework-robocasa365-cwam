### ITER-2026-08-17-B5-VISUAL-OBJECT-MEMORY-R3 - structurally conditioned visual memory

- 时间：2026-08-17T08:36:23+08:00
- 类型：Code Config Experiment
- 动机：R2 prevented slot collapse but aligned visual history did not improve held-out action loss over zero or shuffled history, showing that the complete B4 action path could ignore the added memory context tokens.
- 计划变更：Starting from c762c793b1fccd7f3d43dc2220d4b0f6685b8a44, add a fixed-scale memory-conditioned cross-attention/FiLM branch to every strong-action layer; add a memory-only future-action auxiliary decoder driven by the pre-transformer noisy-action seed; add adjacent-history temporal correspondence prediction over object slots. Preserve the R2 competitive slots, B4 warm start, frozen video expert, 500-step eight-B200 budget, batch 64 per rank, and identical none/zero/shuffle held-out stream.
- 假设：A non-bypassable conditioning branch plus sample-specific future-action and temporal correspondence supervision will make aligned cross-replan visual memory causally useful to the action expert instead of merely diverse.
- 验收：Focused tests and BF16 GPU smoke pass; eight-B200 training completes 500 finite optimizer steps and saves a complete checkpoint; slot mean absolute off-diagonal cosine remains below 0.95; temporal and memory-only action auxiliary losses are finite; on the identical 2048-sample held-out stream, aligned-history main action loss is at least 1 percent lower than both zero-history and batch-shuffled-history interventions. Formal training remains blocked unless every gate passes.
- 基线：Experiment CWAM-COSMOS3-B5-VOM-PROBE-R2-20260817; commit c762c793b1fccd7f3d43dc2220d4b0f6685b8a44.
- 实验：CWAM-COSMOS3-B5-VOM-PROBE-R3-20260817; source commit 1fa9d1c2b14f544f5b8e4e9ff4a968197ab7e8a4; Notion page 3bf21e77-89cc-81d9-bc09-cbe844e6f9e5.
- 实现：在每个 ActionExpertBlock 后加入固定强度 0.25 的 memory cross-attention + FiLM；加入只接 noisy-action seed/progress 与 memory 的 future-action 辅助分支（权重 0.25，50-step warmup）；加入相邻历史帧同 slot 的 stop-gradient temporal correspondence（权重 0.05）。
- 验证：26 个聚焦测试通过；B200 BF16 生产维度 forward/backward 有限，conditioner、auxiliary head 和 temporal predictor 均获得有限非零梯度，峰值显存 907.28 MiB。
- 记录人：Codex
