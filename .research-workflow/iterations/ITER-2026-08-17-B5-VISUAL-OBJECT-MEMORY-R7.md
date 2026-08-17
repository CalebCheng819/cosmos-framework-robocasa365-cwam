### ITER-2026-08-17-B5-VISUAL-OBJECT-MEMORY-R7 - aligned-memory policy intervention ranking

- 时间：2026-08-17T20:10:27+08:00
- 类型：Code Config Experiment
- 动机：R6-A/R6-B 的 compact dynamic memory 表示未坍缩且 exact-zero intervention 生效，但 aligned history 相对 zero/shuffle 的 main action-loss 改善仅为 0.50%/0.15% 与 0.46%/0.10%。增加注入深度只改善一般拟合，没有增强策略对正确历史的依赖。
- 假设：在相同样本、action noise、diffusion timestep 与非 memory 条件下，直接要求 aligned-memory 动作去噪误差低于 zero/shuffle 反事实误差，可增强策略对时间对齐历史的因果依赖。
- 设计：5B joint transformer 每个 batch 只运行一次；从同一 pre-action hidden state 运行 aligned、zero、shuffle 三个 Action Expert 分支。zero/shuffle 分支停止梯度，只作为 target-aware 难度基线，避免模型通过主动恶化负分支取巧。对每个样本优化 `relu(L_aligned - (1-rho) * stopgrad(L_intervention))`，zero 与 shuffle 等权平均。
- 公平性：以 R6-A dynamic-exclusive final-delta 架构为主体，从相同 R4 step-500 checkpoint model-only warm start；当前视觉、task、proprio、action/control history、数据、batch、seed、optimizer scope 和既有 auxiliary losses 保持不变。新增 ranking warmup 50 steps，预设相对 margin rho=1%，weight=1.0。
- 验收：聚焦单测及 8-rank BF16 forward/backward/optimizer/EMA 门禁通过；反事实分支与 aligned 共用 noise/timestep/target 且反事实输出无梯度；zero intervention 主 memory 路径严格为零；500-step probe 完成有限训练并保存完整 checkpoint；matched 2048-sample validation 上 aligned action loss 至少比 zero 与 shuffle 各低 1%，否则不扩展到 20k。
- Git：从 `14becbd2dcd995014d6dcb05537dafe4257ec4b2` 创建 `exp/cosmos3-b5-visual-object-memory-r7-20260817`；实现提交 `324ace63f3e665ed7226371db67863aee2cde503` 已推送。远端 clean detached source 为 `/data0/wam/cwam-b5/source/CWAM-COSMOS3-B5-VOM-R7-GIT-20260817`。
- 验证：远端 CPU targeted regression `38 passed`；8-rank B200 BF16 smoke 全 ranks 通过，确认 ranking loss 有限且为正、aligned memory-delta 参数获得非零梯度、zero/shuffle counterfactual 无梯度、zero memory delta 严格为零、per-call intervention 不修改模块默认状态。
- 计划实验：`CWAM-COSMOS3-B5-VOM-PROBE-R7-POLICY-RANK-20260817`。
- 状态：Planned；正式实验页面将在代码冻结后、八卡提交前创建。
- 记录人：Codex
