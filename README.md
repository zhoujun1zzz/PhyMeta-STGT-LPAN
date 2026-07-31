# LPAN Two-Dataset RIS Channel Completion

统一处理 LPAN 准静态和时变数据集的 RIS 级联信道补全工程，包含可复现 baseline、
LPAN-Compatible PhyMeta-STGT、独立训练、平衡联合训练和准静态到时变少样本迁移。

> 数据集不包含在本仓库中。使用者需要从 IEEE DataPort 单独下载，并遵守数据集
> 页面给出的许可和引用要求。

## 已实现

- Windows worker-safe HDF5 懒加载；
- 统一复数布局：
  - 准静态 `[B,1,32,64,2] -> [B,1,256,64,2]`
  - 时变 `[B,2,32,64,2] -> [B,6,256,64,2]`
- LS coarse input + nearest/linear 空间与时间插值；
- Empirical Ridge，正则系数只用验证集选择；
- EDSR-lite、Spatial GCN、CNN-GRU、GCN-GRU；
- CNN-GRU/GCN-GRU 使用自回归 GRU 时间解码器生成 6 个不同目标块；
- Spatial GCN 故意保留线性插值/末端保持的时间策略，作为纯空间消融；
- PhyMeta-STGT：observed-to-all cross-attention、四邻域稀疏图注意力、
  可变长度时间查询、domain FiLM adapter、逐 RIS 节点解码；
- 样本级复数 NMSE、Charbonnier、粗信道观测一致性、时间变化匹配损失；
- 总体、逐时间块、导频/非导频块、已观测/未观测 RIS 的 NMSE；
- full、frozen-spatial、adapter-only、selective 四种适配策略；
- checkpoint、训练历史 CSV、最终 JSON、完整命令和可选按 SNR CSV。

CNN-GRU/GCN-GRU 在编码两个导频块后，从最终上下文递归解码位置 `0..5`，属于
sequence-to-sequence 全帧重建（包括两个导频位置），不是只从位置 2 开始的严格
future-only rollout。

## 导频块位置

时变数据的两个导频块位于六个目标块中的前两个位置，因此零基索引为：

```powershell
--obs-times 0,1
```

这是所有命令的默认值；一般无需重复传入。参数仍保留，便于后续处理其他数据设置。
32 个 RIS 观测列默认映射到 `0,8,...,248`，复数通道默认采用 grouped 排列。
两项均可显式覆盖：

```bash
--obs-ris-indices 0,8,16,...,248
--complex-layout grouped  # 或 interleaved
```

MAT 文件本身不包含 RIS 物理索引或 UPA flatten 元数据。正式实验仍应以权威数据
生成设置确认这些语义；`audit` 只报告当前配置和可检查的证据，不会把假设标成事实。

## 快速开始

要求 Python 3.10 或更高版本。GPU 不是运行必需项，但正式训练建议使用 CUDA。

```bash
git clone https://github.com/zhoujun1zzz/LPAN-two-dataset-channel-completion.git
cd LPAN-two-dataset-channel-completion
python -m pip install -e .
```

开发和测试依赖：

```bash
python -m pip install -e ".[dev]"
pytest
```

## 下载数据集

- [LPAN 准静态数据集](https://doi.org/10.21227/3c2t-dz81)
- [LPAN 时变数据集](https://doi.org/10.21227/pz7h-q132)
- [LPAN 官方代码与论文说明](https://github.com/WiCi-Lab/LPAN)

下载并解压后，推荐目录如下：

```text
data/
├── quasi/
│   ├── indoorH_LS_Data6users_1B32pilot/
│   │   └── indoorH_LS_Data6users_1B32pilot.mat
│   ├── indoorH_LSval_Data6users_1B32pilot/
│   │   └── indoorH_LSval_Data6users_1B32pilot.mat
│   └── indoorH_LStest_Data6users_1B32pilot/
│       └── indoorH_LStest_Data6users_1B32pilot.mat
└── mobility/
    ├── OutdoorH_LS_Data6users_60B32pilot/
    │   └── OutdoorH_LS_Data6users_60B32pilot.mat
    ├── OutdoorH_LSval_Data6users_60B32pilot/
    │   └── OutdoorH_LSval_Data6users_60B32pilot.mat
    └── OutdoorH_LStest_Data6users_60B32pilot/
        └── OutdoorH_LStest_Data6users_60B32pilot.mat
```

程序也能识别 MAT 文件直接位于 `data/quasi`、`data/mobility` 或数据根目录的
布局。数据位置有三种配置方式，优先级从高到低为：

1. 单个命令的 `--train-path`、`--val-path` 或 `--data-path`；
2. `--data-root /path/to/data`；
3. 环境变量 `LPAN_DATA_ROOT`；
4. 默认的仓库内 `data/`。

PowerShell 示例：

```powershell
$env:LPAN_DATA_ROOT = "D:\datasets\LPAN"
python main.py audit
```

Bash 示例：

```bash
export LPAN_DATA_ROOT=/mnt/datasets/LPAN
python main.py audit
```

审计会报告六个 split 的文件位置、HDF5 keys、原始 shape、dtype 和统一接口 shape；
只读取元数据及每个 split 的一个样本。

## 无学习 baseline

```bash
python main.py interpolate --domain quasi --split validation
python main.py interpolate --domain mobility --split validation \
  --spatial linear --temporal linear

python main.py ridge --domain quasi --max-train 512 --max-val 128
python main.py ridge --domain mobility --max-train 512 --max-val 128
```

去掉 `--max-*` 才是完整实验。Ridge 加 `--test` 后才会读取独立测试集。

## 学习模型 smoke test

```bash
python main.py train --domain quasi --model edsr_lite --mode smoke
python main.py train --domain quasi --model spatial_gcn --mode smoke
python main.py train --domain mobility --model cnn_gru --mode smoke
python main.py train --domain mobility --model gcn_gru --mode smoke
python main.py train --domain mobility --model phymeta_stgt --mode smoke
```

`smoke` 固定只训练 64、验证 16 个样本且只跑 1 epoch，结果不能作为正式性能。

## 完整独立训练与测试

```bash
python main.py train --domain quasi --model phymeta_stgt --mode full \
  --epochs 100 --batch-size 8 --run-name quasi_stgt_seed123

python main.py train --domain mobility --model phymeta_stgt --mode full \
  --epochs 100 --batch-size 2 \
  --run-name mobility_stgt_seed123

python main.py evaluate \
  --checkpoint runs/mobility_stgt_seed123/checkpoints/best_checkpoint.pth \
  --domain mobility --split test --per-snr \
  --output runs/mobility_stgt_seed123/results/independent_test.json
```

训练只使用 train 和 validation。`evaluate --split test` 是单独入口。

## 平衡联合训练

```bash
python main.py joint --mode smoke
python main.py joint --mode full --epochs 100 \
  --run-name joint_stgt_seed123
```

每个 batch 只含一个任务，并按准静态/时变交替更新；不会复制准静态标签到 6 块。

## 准静态到时变少样本迁移

先完成准静态预训练，然后对每个比例和随机种子复用完全相同的目标域抽样规则：

```bash
python main.py train --domain mobility --model phymeta_stgt --mode full \
  --pretrained runs/quasi_stgt_seed123/checkpoints/best_checkpoint.pth \
  --fraction 0.05 --adaptation selective --seed 123 \
  --run-name transfer_5pct_selective_seed123
```

对应对照只需改变：

- scratch：去掉 `--pretrained`，并使用 `--adaptation full`；
- full fine-tuning：`--adaptation full`；
- frozen spatial：`--adaptation frozen_spatial`；
- adapter-only：`--adaptation adapter_only`；
- proposed selective：`--adaptation selective`。

`--pretrained` 只接受结构兼容的 PhyMeta-STGT checkpoint，并使用严格权重键检查；
传入其他模型、不同 hidden/图层/heads 配置或缺失关键权重都会立即报错。

建议比例为 `0.01/0.05/0.10/0.20/1.0`，每个比例至少 3 个 seed。
同一 seed 使用一个随机排列的前缀，因此 1% 子集严格包含于 5%，5% 包含于 10%。

## 恢复训练

```bash
python main.py train ... \
  --resume runs/<run>/checkpoints/last_checkpoint.pth \
  --run-name <run>
```

恢复时命令中的模型、数据、比例和主要训练配置必须与原实验一致。
checkpoint 会保存 Python、NumPy、PyTorch CPU/CUDA 和 DataLoader 随机状态；恢复时
代码会校验配置、接续原训练历史，并把恢复命令追加到 `resume_commands.log`。

独立 `evaluate` 默认从 checkpoint 继承 domain、导频时间、RIS 映射和复数布局。
若命令行显式给出的语义与 checkpoint 不同，程序会拒绝运行；只有确实需要跨语义
测试时才使用 `--allow-semantic-override`。

## 按 SNR 评估

数据文件没有逐样本 SNR 字段，因此分组必须显式声明并通过总样本数校验：

```bash
python main.py evaluate \
  --checkpoint runs/<run>/checkpoints/best_checkpoint.pth \
  --domain mobility --split test --per-snr \
  --snr-values=-10,-5,0,5,10,15,20,25,30 \
  --samples-per-snr 1000
```

若测试集样本数不等于 `SNR 数量 × 每组样本数`，或样本已被抽样/重排，程序会拒绝
生成可能错误的标签。

## 输出与公平性

- 训练仅使用官方 train/validation split 选择最佳 checkpoint；
- test split 只通过独立 `evaluate` 命令读取；
- 每个 run 保存完整命令、best/last checkpoint、训练历史 CSV 和结果 JSON；
- Charbonnier、观测一致性和时间差分辅助项均按样本功率归一化；
- 低 SNR 下观测一致性可能鼓励保留 LS 噪声，正式报告应包含 `--obs-weight 0`
  及不同辅助权重的消融；
- `.gitignore` 排除数据集、checkpoint 和实验输出，避免意外提交大文件；
- 时变任务只做单帧内 `2 -> 6` 重建，不跨 MAT 样本拼接轨迹。

## 引用

使用本工程进行研究时，请引用 LPAN 数据集页面以及对应论文：

J. Xiao, J. Wang, Z. Wang, W. Xie, and Y. Liu, “Multi-Scale Attention Based
Channel Estimation for RIS-Aided Massive MIMO Systems,” *IEEE Transactions on
Wireless Communications*, vol. 23, no. 6, pp. 5969–5984, 2024.

## License

本仓库代码采用 [MIT License](LICENSE)。LPAN 数据集和官方代码不属于本许可证
覆盖范围，请分别遵守其来源页面的条款。
