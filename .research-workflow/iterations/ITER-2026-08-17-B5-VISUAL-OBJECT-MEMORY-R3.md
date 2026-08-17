### ITER-2026-08-17-B5-VISUAL-OBJECT-MEMORY-R3 — structurally conditioned visual memory

- 时间：2026-08-17T08:36:23+08:00
- 类型：Code Config Experiment
- 动机：R2 prevented slot collapse but aligned visual history did not improve held-out action loss over zero or shuffled history, showing that the complete B4 action path could ignore the added memory context tokens.
- 变更：Starting from c762c793b1fccd7f3d43dc2220d4b0f6685b8a44, add a fixed-scale memory-conditioned cross-attention/FiLM branch to every strong-action layer; add a memory-only future-action auxiliary decoder driven by the pre-transformer noisy-action seed; add adjacent-history temporal correspondence prediction over object slots. Preserve the R2 competitive slots, B4 warm start, frozen video expert, 500-step eight-B200 budget, batch 64 per rank, and identical none/zero/shuffle held-out stream.
- 假设：A non-bypassable conditioning branch plus sample-specific future-action and temporal correspondence supervision will make aligned cross-replan visual memory causally useful to the action expert instead of merely diverse.
- 验收：Focused tests and BF16 GPU smoke pass; eight-B200 training completes 500 finite optimizer steps and saves a complete checkpoint; slot mean absolute off-diagonal cosine remains below 0.95; temporal and memory-only action auxiliary losses are finite; on the identical 2048-sample held-out stream, aligned-history main action loss is at least 1 percent lower than both zero-history and batch-shuffled-history interventions. Formal training remains blocked unless every gate passes.
- 基线：Experiment CWAM-COSMOS3-B5-VOM-PROBE-R2-20260817; commit c762c793b1fccd7f3d43dc2220d4b0f6685b8a44.
- Git：Source commit 1fa9d1c2b14f544f5b8e4e9ff4a968197ab7e8a4 on exp/cosmos3-b5-visual-object-memory-r3-20260817; clean and pushed before launch.
- 实验：CWAM-COSMOS3-B5-VOM-PROBE-R3-20260817; source commit 1fa9d1c2b14f544f5b8e4e9ff4a968197ab7e8a4; Notion page 3bf21e77-89cc-81d9-bc09-cbe844e6f9e5.
- 实现：在每个 ActionExpertBlock 后加入固定强度 0.25 的 memory cross-attention + FiLM；加入只接 noisy-action seed/progress 与 memory 的 future-action 辅助分支（权重 0.25，50-step warmup）；加入相邻历史帧同 slot 的 stop-gradient temporal correspondence（权重 0.05）。
- 验证：26 个聚焦测试通过；B200 BF16 生产维度 forward/backward 有限，conditioner、auxiliary head 和 temporal predictor 均获得有限非零梯度，峰值显存 907.28 MiB。
- 运行：八张 B200 完成 500 个有限 optimizer steps，step 500 的 191571798 个可训练参数梯度中 non-finite 数为 0；保存完整 iter_000000500（model/optim/scheduler/trainer 均为 8 个 rank shard 加 metadata，共 93812338333 bytes）。训练期 GPU 活跃样本平均利用率 74.97%，平均显存 56079 MiB，峰值显存 58654 MiB。
- 结果：相邻帧 correspondence loss 从 0.8570142 降至 0.0064946，memory-only future-action loss 从 1.6276376 降至 0.3089735，说明两项辅助任务确实被优化；但 slot mean absolute off-diagonal cosine 从 0.9989161 升至 0.9996318，八个 slot 严重同质化。相同 2048 样本下主 action loss 为 none 0.0082432544、zero 0.0082443468、shuffle 0.0082437079；none 相对两项干预仅改善 0.01325% 和 0.00550%，均未达到 1%。三项验证均 exit code 0。
- 决策：实验执行完成，但研究验收失败。固定的 per-layer memory conditioning 使分支结构上活跃，辅助损失却可以被共享的 episode-level 表征解决，未迫使模型保留对象级差异或使用样本特定 memory。anti-collapse 和 none/zero/shuffle 两类门槛均失败，因此不启动 20k 正式训练。
- 下一步：保留 R2 的竞争分配/多样性约束，再加入对象局部 temporal correspondence、跨对象/跨 episode negatives 或 action-gradient localization；避免仅靠全局 predictive auxiliary 重新制造 slot collapse，并继续沿用相同三组因果干预。
- 记录人：Codex
