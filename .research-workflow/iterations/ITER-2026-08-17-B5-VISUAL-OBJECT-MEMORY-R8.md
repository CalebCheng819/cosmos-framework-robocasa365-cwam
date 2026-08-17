### ITER-2026-08-17-B5-VISUAL-OBJECT-MEMORY-R8 — high-utilization formal training

- 时间：2026-08-17T22:41:24+08:00
- 类型：Config Experiment Decision
- 动机：R7 正式训练使用 batch 64/rank 和 selective activation checkpointing 时，8 张 B200 的平均 GPU utilization 约为 60%-65%，显存仅约 54-56 GiB/卡，需要在保留 aligned-memory intervention dependence 的前提下提高有效算力利用率。
- 变更：从相同 R4 step-500 model-only checkpoint 分别运行 B128 与 B192 八卡 100-step 探针，关闭 activation checkpointing，并按 batch 将 token cap 提高到 296000/444000。新增 B128 20k launcher，保持 R6-A dynamic-exclusive full final-delta 架构、R7 policy-ranking、数据、seed、LR、train scope 和 memory intervention 不变。
- Git：训练源固定为已推送 commit `88f3c95c18a3f3195b66e35b2cd8f6ecfe595b81`；R8 探针 launcher 与前序记录已推送至分支 `exp/cosmos3-b5-visual-object-memory-r7-20260817`；本次终态与正式 launcher 将以新的普通 Git commit 推送，不计算文件 hash。
- 实验：B128 page `3bf21e77-89cc-81a2-8e39-fae2251c171a`，job `cwam-b5-vom-probe-r8-b128`，W&B offline `q44f94ks`；B192 page `3bf21e77-89cc-81c3-a2b0-c2823a1cbad2`，job `cwam-b5-vom-probe-r8-b192`，W&B offline `2jvkmqy9`；正式实验 `CWAM-COSMOS3-B5-VOM-FORMAL-R8-HIGH-UTIL-B128-20K-20260817` 待创建页面和提交。
- 结果：B128 完成 100 步，mean throughput 163.535 samples/s，峰值显存 96,500 MiB，aligned 相对 zero/shuffle 改善 10.90046%/8.71514%。B192 完成 100 步，mean throughput 164.371 samples/s，峰值显存 155,996 MiB（85.08%），平均 GPU utilization 80.34%-85.68%，aligned 相对 zero/shuffle 改善 9.71371%/7.78553%。两者均为有限梯度、exit 0、完整 8-rank checkpoint。
- 决策：B192 的 mean throughput 仅比 B128 提高 0.511%，不构成实质提升。固定 20,000 optimizer steps 时，B192 约需 51.9 小时，B128 约需 34.8 小时，因此正式配置选择 B128/rank、global batch 1024、token cap 296000、activation checkpointing none。
- 下一步：创建正式 Notion 页面与 prelaunch manifest，验证远端 clean source、R4 checkpoint、空闲 8 卡、存储和独立 run root 后，从 optimizer step 0 启动 20k；每 500 steps matched validation，每 1,000 steps 完整 DCP checkpoint，并在出现真实有限 optimizer step 后更新为 Running。
- 记录人：Codex
