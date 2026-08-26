### ITER-2026-08-26-LIBERO-WAN-CACHE - Precompute strict LIBERO Wan VAE latents

- 时间：2026-08-26T20:15:00+08:00
- 类型：Code, Config, Data, Experiment
- 动机：当前 LIBERO S1 在线训练每步约 58-60 秒，视频解码、resize 和 Wan VAE encode 是主要可消除开销。预计算固定样本索引对应的 Wan latent，用严格缓存读取模式实测训练吞吐提升。
- 计划变更：增加 LIBERO latent 固定记录格式与只读 reader；为 LIBERO 数据集增加 exact-index 视频窗口读取和 required cache 模式；增加三卡 H299 分布式构建、OSS 分片续传、完整性清点和训练侧缓存 staging；保持在线路径默认行为不变。
- Git：base c4509dddd909063d9752694f285fe6b68e8f83ff；branch exp/libero-s1-4suite-20260826；目标提交待生成。
- 实验：CWAM-COSMOS3-LIBERO-WAN-CACHE-20260826-001；Notion page 待创建；计划使用 H299 三张 H200，缓存只持久化到 eailab-hdd2:eailab/chengjuntao/cwam/libero-s1-4suite-20260826/wan-latent-cache-v1。
- 验收：四套 train split 共 247835 个索引均有且仅有一个 BF16 latent；shape 为 [48,5,10,20]；严格 reader 首尾样本可读；真实样本在线/缓存等价预检通过；随后用相同配置完成短程在线与缓存 A/B，报告实测速率。
- 记录人：juntao-cheng@sjtu.edu.cn
