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
- 记录人：Codex
