### ITER-2026-08-17-B5-VISUAL-OBJECT-MEMORY-R6 - R5 contribution decomposition and dynamic-exclusive memory

- 时间：2026-08-17T18:10:00+08:00
- 类型：Code Config Experiment
- 动机：R5 在保持 object slots 可分离的同时，使 aligned history 相对 zero/shuffle 的主 action loss 分别改善 0.35283%/0.09025%，但未达到 1% 门槛且 aligned loss 比 R4 差 0.19041%。需要先分离 multiplicative gate 与 directional residual 的贡献，再去掉主策略对 absolute visual-memory tokens 的旁路。
- R5 贡献分解：复用 R5 iter_000000500，不训练；在相同 2048 个 held-out samples 上分别执行 base、gate-only、residual-only、full 四种最终 residual 组合，并记录 applied delta RMS、delta/base RMS ratio、delta-base cosine 与 action loss。none/zero/shuffle 使用相同 seed 和数据顺序。
- R6-A：从 R4 iter_000000500 model-only warm start训练新的 500-step probe。主 ActionDiT context 不再拼接 absolute visual-memory tokens，也不运行 absolute-memory per-layer conditioner；对象 memory 只通过最终 object-local delta gate/residual 进入主 action 输出。当前视觉、task、proprio、action history/control memory 保持不变。
- R6-B：与 R6-A 完全相同，但在 ActionDiT 的 4 个 action layers 后额外加入 exact-zero-masked object-delta cross-attention plus FiLM。最终 delta gate/residual 与 R6-A 相同。
- 公平性：R6-A/R6-B 使用相同 R4 checkpoint、数据、batch 64/rank、DP8、500 optimizer steps、seed、optimizer scope、训练损失和 2048-sample none/zero/shuffle validation；新增模块允许 model-only 非严格加载，optimizer/scheduler/trainer state 不恢复。
- 验收：聚焦测试和生产维度 BF16 forward/backward 通过；zero intervention 下所有主策略 delta 路径严格为零；每个 probe 完成 500 个有限 optimizer steps并保存完整 8-rank checkpoint；slot cosine 低于 0.95，两个 contrastive margins 高于 0.05；aligned main action loss 至少比 zero 和 shuffle 各低 1%。任一门槛失败均不启动 20k。
- Git：从 7fd7885f621b9c6b705d02118350daa83b929198 创建 exp/cosmos3-b5-visual-object-memory-r6-20260817；实现、测试、push 后冻结远端 source snapshot。
- 实验：CWAM-COSMOS3-B5-VOM-R5-DECOMP-20260817；CWAM-COSMOS3-B5-VOM-PROBE-R6A-20260817；CWAM-COSMOS3-B5-VOM-PROBE-R6B-20260817。
- 实现验证：远端 B200 容器聚焦单测 22 passed；R6-A/R6-B 均通过 8-rank BF16 forward/backward smoke。两者 action-gradient valid fraction 均为 1.0，absolute-memory main context tokens 均为 0，zero intervention 下新增主路径严格归零。R6-A delta-conditioned layers/RMS 为 0/0，final applied delta RMS 为 0.22210；R6-B 为 4/0.11197，final applied delta RMS 为 0.22414。
- 修复记录：首轮 R6 smoke 暴露 action-gradient supervision 仍对 absolute slots 求导，导致 dynamic-exclusive 主路径下 valid fraction 为 0；已改为 exclusive 模式对真实进入 action path 的 object delta 求导，修复后两组均为 1.0。
- R5 分解结果：12 个 matched validation cells 均完成且每个覆盖相同 2048 个 held-out samples。aligned base/gate-only/residual-only/full action loss 分别为 0.008303430/0.008301578/0.008344762/0.008263416；gate-only 相对 base 改善 0.02231%，residual-only 退化 0.49777%，full 改善 0.48190%，推导出的 gate-residual interaction 为 base loss 的 0.95736%。full none/zero/shuffle 为 0.008263416/0.008293392/0.008271596，仅改善 0.36145%/0.09890%。因此拒绝“单一路径贡献主导”假设：两个分支需要协同，但因果效应仍低于 1% 门槛。
- R6-A attempt0：Run `CWAM-COSMOS3-B5-VOM-PROBE-R6A-R4WARM-500S-20260817` 完成 initial validation 后，在第一个 optimizer step 前由 fail-closed action-gradient gate 终止。缺失梯度的 1,025/199,968,598 个元素精确对应 dynamic-exclusive 模式不可达的 legacy `visual_memory_context_embed`（1,024）与 `visual_memory_gate_logit`（1），不是数值非有限问题；失败日志、GPU 监控和首尾容器 inspect 均保留。
- R6 修复：commit `a8681df43d240cd56d86627b7677d7b9dbc78f1c` 冻结上述两个不可达 legacy 参数，并允许梯度证据 callback 显式排除它们；所有其他 active action parameters 仍执行 fail-closed 检查。修复后 22 个聚焦测试及 R6-A/R6-B 8-rank BF16 smoke 均通过。正式重试使用全新 Run IDs `CWAM-COSMOS3-B5-VOM-PROBE-R6A-R4WARM-500S-RETRY1-20260817` 与 `CWAM-COSMOS3-B5-VOM-PROBE-R6B-R4WARM-500S-FIX1-20260817`。
- 记录人：Codex
