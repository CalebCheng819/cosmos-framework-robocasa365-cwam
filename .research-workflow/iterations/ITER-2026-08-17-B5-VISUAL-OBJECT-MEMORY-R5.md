### ITER-2026-08-17-B5-VISUAL-OBJECT-MEMORY-R5 - delta-gated main action residual

- 时间：2026-08-17T14:32:59+08:00
- 类型：Code Config Experiment
- 动机：R4 已把 slot absolute cosine 从 0.9989161 降到 0.2091563，并使 memory-only action loss 对 aligned history 明显优于 zero；但主 action loss 的 none/zero/shuffle 分别为 0.0082484251/0.0082450183/0.0082476446，说明完整 ActionDiT 仍能绕过中间层 memory conditioner。
- 变更：从最近两个有效历史时刻的同索引 object slots 构造 object-local memory delta；最终主 action residual 通过 action-to-object-delta attention 获得局部 delta，并同时施加不可学习关闭的 raw-delta multiplicative gate 与可学习 directional residual。zero intervention 必须让 delta、gate deviation 和 directional residual 严格为零；shuffle 保留动态幅值但破坏 episode 对齐。保留 R4 的局部竞争、多样性、跨对象/episode negatives、action-gradient supervision 和 memory-only auxiliary。
- 假设：如果对象局部动态被结构性写入最终 hidden-state residual，而不再只作为可旁路的中间 cross-attention 条件，主 action objective 将对 aligned object delta 产生可测的因果依赖。
- 验收：聚焦测试和生产维度 BF16 forward/backward 通过，新增 delta 模块参数及 visual memory 获得有限非零梯度；zero intervention 下 object delta RMS、main gate deviation、directional residual RMS 均为零；八张 B200 完成 500 个有限 optimizer steps并保存完整 checkpoint；相同 2048 样本和 seed 下，aligned-history 主 action loss 至少比 zero-history 和 batch-shuffled-history各低 1%；slot absolute cosine 继续低于 0.95，同对象正样本相似度比跨对象及跨 episode negatives 各高至少 0.05。任一门槛失败均不启动 20k 正式训练。
- 基线：Experiment CWAM-COSMOS3-B5-VOM-PROBE-R4-20260817；checkpoint iter_000000500；主 action none/zero/shuffle 0.0082484251/0.0082450183/0.0082476446；memory-only none/zero/shuffle 0.3021545/0.3670223/0.3114415。
- Git：分支 exp/cosmos3-b5-visual-object-memory-r5-20260817；实现提交在测试与 GPU smoke 通过后生成并推送。
- 实验：CWAM-COSMOS3-B5-VOM-PROBE-R5-20260817；Run CWAM-COSMOS3-B5-VOM-PROBE-R5-R4WARM-500S-20260817；正式页面在启动前创建。
- 记录人：Codex
