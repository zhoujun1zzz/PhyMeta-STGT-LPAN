# 两轮修改摘要

本文档只记录最近两轮修改，不重复服务器环境、数据上传或完整实验命令。

## 第一轮：实验可信度与协议约束

- 指标累积改为 fail-fast：预测、目标或指标中出现 NaN/Inf 时立即报错，不再静默丢弃样本。
- 数据语义验证改为证据推断：`verify_data_semantics.py` 分别尝试 grouped/interleaved 解码，根据 Yd 与对应 Hd pilot 位置的复数相关性、领先幅度和一致性给出 verified、ambiguous 或 inconsistent 结论。
- 复杂度口径统一为 batch size 1、FP32、一次前向和 `1 MAC = 2 FLOPs`；所有空间/时间插值路径均不计入 GMAC/GFLOP。
- 正式消融必须显式继承 Stage B `best_result.json` 中的 hidden size、图层数、head 数、dropout、学习率和 weight decay。
- Ridge 的正则化只由训练/验证数据选择，独立 test 评估后移至协议冻结后的 Stage F。

## 第二轮：模型语义、恢复边界与可审计性

- 空间插值由 flattened-index 一维插值改为逐行 grid-aware interpolation；index 15 不会再使用下一行 index 16 的观测。interpolation baseline、EDSR-lite、CNN-GRU 和空间注意力消融均使用该实现。
- 正式名称统一为 **LPAN-L-Direct**，删除歧义 `lpan_l` alias；模型仅接受官方 `(0,8,...,248)` 输入顺序。
- adaptation 参数集合改为严格嵌套：`adapter_only < selective < frozen_spatial < full`，并在 run metadata 中保存可训练参数名、模块名、数量和比例。
- observation mask 同时作用于空间扩展、attention 和观测一致性损失；全 padding 或某行无有效观测时明确报错。
- 新增 `official_lpan/custom` semantic profile。默认 official 模式锁定 RIS 索引、pilot 时间和 grouped 复数排列，避免只修改标签而不改变原始列。
- audit 新增文件大小、key、shape 和完整样本数记录，并强制 Mobility train/validation/test 为 `20000/1800/9000`。
- checkpoint 改为临时文件写入后 `os.replace()` 原子替换；history 超前 checkpoint 时自动安全截断并写入 `recovery.log`，不可重建时仍拒绝 resume。
- 消融结果新增 `variant_id`、`display_name` 和 `replacement_mechanism`，明确空间/时间注意力移除后使用的确定性插值机制。
- README 明确当前 PhyMeta-STGT 不是 MAML 或 episodic meta-learning；本轮不进行破坏 checkpoint 的全局重命名。

## 对既有实验的影响

以下结果必须重跑：interpolation、EDSR-lite、CNN-GRU、`no_spatial_cross_attention`，以及参数集合发生改变的 `selective`/`adapter_only` 迁移实验。PhyMeta-STGT 完整模型与 LPAN-L-Direct 的参数结构未改变，旧权重通常仍可加载；但 `build_model("lpan_l")` 必须改为 `build_model("lpan_l_direct")`。

本地 41 项回归测试运行到 100%，未出现断言失败，另已通过语法和 JSON 完整性检查。本机 pytest 在显示 100% 后存在 Windows 解释器退出挂起，因而人工终止空转进程。正式实验前仍需在服务器完成真实数据 audit、语义验证和 CUDA smoke。

joint training 的完整 resume/RNG/recovery 机制属于条件项：只有在 joint training 确认进入论文正式结果后再补齐。

## 服务器 CUDA resume 补丁

真实服务器验证发现：checkpoint 使用 `map_location=cuda` 加载时，CPU RNG 和 DataLoader generator 的状态 tensor 也会被搬到 GPU，旧实现随后把 CUDA tensor 直接交给只接受 CPU ByteTensor 的 PyTorch RNG 接口。现在 `restore_rng_state()` 会对 CPU RNG、CUDA RNG 状态列表和 DataLoader generator 状态逐项执行类型检查，并显式恢复为 contiguous CPU `uint8` tensor。该补丁只影响 resume 的 RNG 恢复路径，不改变模型权重或 optimizer state 的加载策略。

新增 CUDA-only 回归测试覆盖真实 `map_location=cuda` 场景，并增加可在 CPU CI 中执行的 dtype 归一化与错误类型测试。此前已通过的 audit、数据语义验证和四个 CUDA smoke 无需重跑；服务器只需基于新 commit 重新验证 history recovery 后能够继续训练到下一 epoch，并确认 `resume_commands.log` 正常生成。
