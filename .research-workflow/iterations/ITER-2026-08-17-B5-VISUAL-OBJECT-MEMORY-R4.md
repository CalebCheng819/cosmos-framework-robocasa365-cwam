### ITER-2026-08-17-B5-VISUAL-OBJECT-MEMORY-R4 - object-local contrastive memory

- 时间：2026-08-17T11:02:11+08:00
- 类型：Code Config Experiment
- 动机：R3 的 temporal correspondence 和 memory-only future-action auxiliary 都快速下降，但八个 object slots 的平均绝对非对角 cosine 上升到 0.9996318，说明共享的 episode-level representation 可以绕过对象级建模并解决 predictive auxiliary。
- 变更：恢复并增强 token-to-slot 局部竞争与 slot 多样性约束；增加低熵 assignment、slot usage balance 和局部空间 compactness；对相邻历史帧加入同 episode 跨对象 negatives 与同 slot 跨 episode negatives；从主 action loss 对 memory slots 的一阶梯度构造 stop-gradient saliency target，监督每层 memory-conditioned cross-attention 的对象选择。保留 R3 的固定强度 memory cross-attention/FiLM、memory-only action auxiliary、B4 warm start、冻结 video expert、500-step 八卡短探针和相同 none/zero/shuffle 验证流。
- 假设：只有当每个 slot 必须竞争局部 token、区分其他对象与其他 episode，并且 policy attention 对齐到主 action loss 的局部 slot sensitivity 时，predictive auxiliary 才不能继续由共享的 collapsed representation 解决，aligned visual memory 才会产生可测的因果作用。
- 验收：聚焦测试和单卡 B200 BF16 生产维度 forward/backward 通过，所有新增损失及参数梯度有限；八张 B200 完成 500 个有限 optimizer steps并保存完整 checkpoint；最终 slot 平均绝对非对角 cosine 低于 0.95；同对象正样本相似度分别比跨对象和跨 episode negatives 高至少 0.05；action-gradient target 有效样本比例大于 0 且 alignment loss 获得有限非零梯度；相同 2048 样本和 seed 下，aligned-history 主 action loss 至少比 zero-history 和 batch-shuffled-history 各低 1%。任一门槛失败均不启动 20k 正式训练。
- 基线：Experiment CWAM-COSMOS3-B5-VOM-PROBE-R3-20260817；主 action loss none 0.0082432544、zero 0.0082443468、shuffle 0.0082437079；最终 slot absolute cosine 0.9996318。
- Git：分支 exp/cosmos3-b5-visual-object-memory-r4-20260817；实现提交在测试与 GPU smoke 通过后生成并推送。
- 实验：CWAM-COSMOS3-B5-VOM-PROBE-R4-20260817；Run CWAM-COSMOS3-B5-VOM-PROBE-R4-B4WARM-500S-20260817；正式页面在启动前创建。
- 记录人：Codex
- 实现：提交 0169f11f719b4abb5b4b3580e0131406872d5b04；生产容器内 22 个聚焦测试通过；两 rank B200 BF16 生产维度 smoke 通过，action-gradient valid fraction 为 1.0，六组新增参数均获得有限非零梯度。
- 运行：八张 B200 完成 500 个有限 optimizer steps，step 500 的 action expert 梯度 L2 norm 为 4.1605312，187,299,210 个元素非零且无 non-finite。完整 iter_000000500 含 model/optim/scheduler/trainer 各 9 个文件，总大小 93,812,338,333 bytes。三组 matched validation 均 exit code 0。
- 结果：slot absolute cosine 从 0.9989161 降至 0.2091563，diversity loss 从 0.9978336 降至 0.0985463；同对象正样本相似度为 0.7254457，跨对象和跨 episode 负样本分别为 0.0084049 和 0.0618669，margin 为 0.7170408 和 0.6635788。局部竞争、多样性和 negatives 成功阻止 R3 的共享 slot 坍缩。memory-only action loss 为 none 0.3021545、zero 0.3670223、shuffle 0.3114415，说明 compact memory 对独立辅助头有预测信息。
- 因果验证：相同 2048 样本下主 action loss 为 none 0.0082484251、zero 0.0082450183、shuffle 0.0082476446；none 分别比 zero 和 shuffle 差 0.04132% 和 0.00946%，没有达到至少改善 1% 的门槛。
- 决策：实验执行完成，但研究验收失败。R4 证明了对象局部 anti-collapse 与跨对象/episode negatives 可以得到非共享、可预测的 memory slots，却也表明完整 ActionDiT 仍能绕过或淹没这些信息。保持 20k 正式训练阻塞。下一次短探针应让主 action residual/gate 显式依赖对象局部 memory delta，并继续使用相同 none/zero/shuffle 门槛；不要再仅增加可由独立辅助头解决的 predictive objective。
