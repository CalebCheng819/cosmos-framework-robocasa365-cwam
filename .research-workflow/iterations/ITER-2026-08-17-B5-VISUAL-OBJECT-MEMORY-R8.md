### ITER-2026-08-17-B5-VISUAL-OBJECT-MEMORY-R8 - high-utilization formal training

- 时间：2026-08-17T22:41:24+08:00
- 类型：Config Experiment
- 动机：R7 正式训练在 8 张 B200 上以 batch 64/rank、selective activation checkpointing 运行时，独立监控的平均 GPU utilization 约为 60%-65%，显存仅约 54-56 GiB/卡，未充分利用 183359 MiB/卡的容量。
- 假设：关闭 activation checkpointing 可减少重算，提高每步速度；同时提高每 rank micro-batch 和 token packing 上限，可提高持续样本吞吐与显存利用率。最终选择必须以 samples/s 为主，而不是单独追求显存占用。
- 设计：从与 R7 相同的 R4 step-500 model-only checkpoint 启动独立 100-step 八卡探针。首选 batch 128/rank、global batch 1024、token cap 296000、activation checkpointing none；若峰值显存仍低于约 80% 且稳定，再测试 batch 192 或 256。所有候选保持 R6-A dynamic-exclusive full final-delta 架构和 R7 policy-ranking 目标不变。
- 公平性：探针和后续正式任务都不恢复 R7 optimizer/scheduler/trainer state；R7 作为低利用率运行独立终止并保留。数据、seed、LR、train scope、memory intervention、policy-ranking weight/margin 保持不变。提高 global batch 会改变每个 optimizer step 的样本数，正式结果必须单独命名并明确记录。
- 探针门槛：8 ranks 完成有限 forward/backward/optimizer；无 OOM、Traceback、NCCL 或非有限梯度；峰值显存低于 90%；稳定窗口 samples/s 高于 R7 基线约 160 samples/s；最终 matched validation 中 aligned action loss 至少比 zero 与 shuffle 各低 1%。
- 正式门槛：选定配置后创建新的 20,000-step confirmatory run；每 500 steps matched validation，每 1,000 steps 完整 DCP checkpoint；20k 完成和 closed-loop evaluation 分别作为独立终态证据。
- 已完成探针：`CWAM-COSMOS3-B5-VOM-PROBE-R8-HIGH-UTIL-B128-20260817`。100 步有限训练、完整 step-100 checkpoint、aligned 相对 zero/shuffle 改善 10.90046%/8.71514%；峰值显存 96,500 MiB/卡，mean throughput 163.535 samples/s，仅比 R7 约 160 samples/s 高 2.2%。
- 最后探针：`CWAM-COSMOS3-B5-VOM-PROBE-R8-HIGH-UTIL-B192-20260817`。若 B192 的 samples/s 没有实质超过 B128，则正式 20k 回选 B128，不能仅以显存占用作为选择依据。
- 状态：B128 Succeeded；B192 Planned；20k formal Pending probe selection。
- 记录人：Codex
