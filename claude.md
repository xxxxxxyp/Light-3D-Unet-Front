# FL-70% Lightweight 3D U-Net Implementation

## 执行助手任务说明文档

本项目实现了 FL-70% 数据上的轻量级 3D-UNet 前端召回器的完整训练与验证流水线。

### 项目核心要求

#### 一、总体目标
- 构建轻量级 3D-UNet 用于 PET-only 病灶候选检测
- 仅使用 FL-70% 数据进行训练与验证
- 实现高召回率的粗掩码与 BBox 输出

#### 二、数据隔离规则（严格遵守）
✅ **允许使用**：
- FL 训练集 (70%, 86例)
- FL 验证集 (15%, 19例)

❌ **禁止使用**：
- DLBCL 数据
- FL 测试集 (15%, 18例 - 黑盒测试)
- 任何外部数据源

#### 三、处理路径 B（强制）
- **空间处理**：保留原始 4×4×4mm spacing，不做重采样
- **强度处理**：0.5%-99.5% 百分位裁剪，线性归一化到 [0,1]
- **SUV 值**：预计算，不得重新计算
- **Patch 大小**：48×48×48 voxels（物理覆盖约 192mm）

### 项目结构

```
Light-3D-Unet-Front/
├── configs/
│   └── unet_fl70.yaml          # 主配置文件
├── data/
│   ├── raw/                    # 原始数据（用户提供）
│   ├── processed/              # 预处理后数据
│   └── splits/                 # 数据划分文件
├── models/
│   ├── unet3d.py              # 模型架构
│   ├── losses.py              # 损失函数
│   ├── dataset.py             # 数据加载器
│   └── metrics.py             # 评估指标
├── scripts/
│   ├── split_dataset.py       # 数据划分
│   ├── preprocess_data.py     # 数据预处理
│   ├── train.py               # 训练脚本
│   ├── inference.py           # 推理脚本
│   └── evaluate.py            # 评估脚本
└── main.py                     # 主执行脚本
```

### 快速开始

#### 1. 环境设置

```bash
# 运行设置脚本
bash setup.sh

# 或手动设置
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. 数据准备

将 FL 数据放置在 `data/raw/` 目录下：

```
data/raw/
├── FL_001/
│   ├── images/
│   │   └── FL_001_pet.nii.gz
│   └── labels/
│       └── FL_001_label.nii.gz
├── FL_002/
...
```

#### 3. 运行完整流水线

```bash
python main.py --mode all
```

或分步执行：

```bash
# 步骤 1: 数据划分
python main.py --mode split

# 步骤 2: 数据预处理
python main.py --mode preprocess

# 步骤 3: 模型训练
python main.py --mode train

# 步骤 4: 推理
python main.py --mode inference

# 步骤 5: 评估
python main.py --mode evaluate
```

### 模型设计

#### 架构特点
- **轻量级设计**：起始通道 16 → 32 → 64 → 128
- **高效卷积**：分组卷积 + 深度可分离卷积
- **残差连接**：提升训练稳定性
- **归一化**：InstanceNorm3d
- **激活函数**：LeakyReLU

#### 损失函数
- **主要**：Focal Tversky Loss (α=0.7, β=0.3, γ=0.75)
  - α=0.7: 高权重假阴性（优先召回率）
  - β=0.3: 低权重假阳性
  - γ=0.75: 聚焦难例
- **备选**：FTL 0.8 + BCE 0.2（训练不稳定时）

### 训练配置

#### 核心超参数
- **Epochs**: 200
- **Batch Size**: 2（显存允许可增至 4）
- **优化器**: AdamW (lr=1e-4, weight_decay=1e-5)
- **调度器**: CosineAnnealingLR
- **Warmup**: 前 5 epochs
- **早停**: 20 epochs 无改善

#### 数据增强
- 随机翻转
- 随机旋转 (±15°)
- 随机缩放 (±10%)
- 强度扰动 (±10%)
- 高斯噪声 (σ≤0.01)

#### Class-Balanced Sampling
- 每 batch ≥50% 含病灶 patch
- 或每 batch 至少 1 个含病灶 patch

### 评估指标

#### 主指标（模型选择）
- **Lesion-wise Recall@Lesion**: 病灶级召回率

#### 次要指标
- **Voxel-wise DSC**: 体素级 Dice 系数
- **Lesion-wise Precision**: 病灶级精确率
- **FP per case**: 每例假阳性数

#### 命中规则
预测与 GT 匹配标准：
- IoU ≥ 0.1，或
- 中心点距离 ≤ 10mm

### 后处理

#### 连通域过滤
- **训练样本生成**：过滤 < 0.1 cc
- **候选生成**：过滤 < 0.5 cc

#### BBox 扩张
- **物理扩张**：10 mm
- **体素扩张**：3 voxels (at 4mm spacing)

#### 输出格式
每个候选 BBox JSON 包含：
- case_id
- mask_id
- bbox_voxel: [zmin, zmax, ymin, ymax, xmin, xmax]
- bbox_mm: [zmin_mm, zmax_mm, ymin_mm, ymax_mm, xmin_mm, xmax_mm]
- volume_cc
- confidence
- orig_spacing
- processing_path

### 验收标准

#### 必须满足
1. ✅ 数据合规：所有样本保留 4×4×4 spacing，metadata.json 完整
2. ✅ 可复现性：提供训练日志、seed、环境信息、checkpoint
3. ✅ 模型选择：best_model.pth 基于验证 Recall 选择
4. ✅ 后处理产物：每例输出 BBox JSON

#### 性能目标（讨论性）
- 验证集 Lesion-wise Recall ≥ 80%
- 若未达成，需在报告中分析原因并给出改进建议

### 交付物清单

#### Milestone 1: 数据准备 (2-4天)
- [ ] data/splits/ with train/val/test lists
- [ ] data/split_manifest.json
- [ ] data/processed/ with metadata.json
- [ ] 预处理验证报告

#### Milestone 2: 初次训练 (4-7天)
- [ ] 训练日志
- [ ] 多个 checkpoint
- [ ] TensorBoard 记录

#### Milestone 3: 超参调优与最终训练 (3-7天)
- [ ] models/best_model.pth
- [ ] 验证报告与指标
- [ ] 训练历史

#### Milestone 4: 推理与最终报告 (2-3天)
- [ ] 概率图 (NIfTI)
- [ ] 候选 BBox (JSON)
- [ ] inference/metrics.csv
- [ ] experiment_report.pdf

### 监控与日志

#### TensorBoard
```bash
tensorboard --logdir logs/tensorboard
```

#### 训练历史
```bash
cat logs/training_history.json
```

#### 评估结果
```bash
cat inference/metrics.csv
cat inference/detailed_results.json
```

### 故障排除

#### 显存不足 (OOM)
```yaml
training:
  batch_size: 1  # 从 2 降至 1
```

或

```yaml
data:
  patch_size: [32, 32, 32]  # 从 48 降至 32
```

#### 训练不稳定
```yaml
loss:
  use_combined_loss: true
  combined_loss_weights:
    focal_tversky: 0.8
    bce: 0.2
```

#### 召回率低
可能原因：
1. 阈值过高 → 做敏感性分析
2. 病灶 patch 不足 → 增加 lesion_patch_ratio
3. 类别不平衡 → 调整 Focal Tversky α/β

### 不可变约束

❌ **禁止**：
- 使用 DLBCL 数据
- 使用 FL 黑盒测试集
- 重新计算 SUV

✅ **必须**：
- 保留 4×4×4mm spacing
- 记录所有变更
- 保存所有中间文件

### 实验报告要求

最终 experiment_report.pdf 必须包含：

1. **数据准备说明**
   - Metadata JSON 示例
   - 裁剪值与归一化参数
   - Patch size 与物理覆盖

2. **训练过程**
   - 训练曲线（loss, recall, DSC）
   - 模型选择依据
   - Best model epoch 与指标

3. **验证结果**
   - 指标表格
   - 阈值敏感性分析
   - 每例结果

4. **分析与建议**
   - 成功/失败案例分析
   - 遇到的问题与解决方案
   - 下一步改进建议

### 随机种子

所有随机操作使用 seed=42：
- NumPy random
- PyTorch random
- 数据划分
- 数据增强

### 环境信息

保存环境快照：
```bash
pip freeze > environment.txt
git log -1 > git_commit.txt
```

### 联系与支持

如有问题，请参考：
1. README.md - 完整英文文档
2. configs/unet_fl70.yaml - 配置说明
3. 各脚本的 --help 选项

---

**重要提示**：
- 严格遵守数据隔离规则
- 所有变更需记录在配置文件中
- 定期保存 checkpoint
- 做好实验笔记以便撰写报告
