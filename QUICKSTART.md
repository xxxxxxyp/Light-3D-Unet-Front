# Quick Start Guide - FL-70% Lightweight 3D U-Net

## 快速开始指南

本指南帮助您在 5 分钟内开始使用 FL-70% 轻量级 3D U-Net 系统。

### 前置要求

- Python 3.8+
- GPU (推荐，用于训练)
- 至少 16GB RAM
- 至少 50GB 磁盘空间

### 安装步骤

#### 选项 1: 使用安装脚本（推荐）

```bash
# 克隆仓库
git clone <repository-url>
cd Light-3D-Unet-Front

# 运行安装脚本
bash setup.sh

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

#### 选项 2: 手动安装

```bash
# 克隆仓库
git clone <repository-url>
cd Light-3D-Unet-Front

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

1. **组织您的数据**

   将 FL 数据放置在 `data/raw/` 目录下：

   ```
   data/raw/
   ├── images/
   │   ├── 0001_0000.nii.gz  # PET 图像
   │   ├── 0002_0000.nii.gz
   │   └── ...
   └── labels/
       ├── 0001.nii.gz  # 病灶标注
       ├── 0002.nii.gz
       └── ...
   ```

   **数据要求**：
   - 格式: NIfTI (.nii 或 .nii.gz)
   - 空间分辨率: 4×4×4mm
   - SUV 值已预计算
   - 标签: 二值 (0=背景, 1=病灶)
   - 文件命名: 图像文件为 `{case_id}_*.nii.gz`，标签文件为 `{case_id}.nii.gz`

2. **数据划分**

   如果您的数据已就位，可以跳过此步骤（系统已创建示例划分）：

   ```bash
   python scripts/split_dataset.py \
       --data_root data/raw \
       --output_dir data/splits \
       --seed 42
   ```

   这将创建：
   - `data/splits/train_list.txt` (86例, 70%)
   - `data/splits/val_list.txt` (18例, 15%)
   - `data/splits/test_list.txt` (19例, 15%, 黑盒测试)
   - `data/split_manifest.json` (元数据)

### 运行完整流水线

#### 一键运行（最简单）

```bash
python main.py --mode all
```

这将依次执行：
1. 数据预处理
2. 模型训练
3. 验证集推理
4. 结果评估

#### 分步运行（推荐用于调试）

```bash
# 步骤 1: 数据预处理
python main.py --mode preprocess

# 步骤 2: 模型训练 (可能需要数小时)
python main.py --mode train

# 步骤 3: 推理
python main.py --mode inference

# 步骤 4: 评估
python main.py --mode evaluate
```

### 监控训练

#### 使用 TensorBoard

```bash
# 在另一个终端窗口
tensorboard --logdir logs/tensorboard

# 在浏览器打开
http://localhost:6006
```

#### 查看训练日志

```bash
# 查看训练历史
cat logs/training_history.json

# 实时查看日志
tail -f logs/*.log
```

### 查看结果

#### 评估指标

```bash
# 查看汇总指标
cat inference/metrics.csv

# 查看详细结果
cat inference/detailed_results.json
```

#### 候选边界框

```bash
# 查看某个病例的候选框
cat inference/bboxes/FL_001_bboxes.json
```

输出示例：
```json
{
  "case_id": "FL_001",
  "processing_path": "B",
  "orig_spacing": [4.0, 4.0, 4.0],
  "threshold": 0.3,
  "num_candidates": 5,
  "candidates": [
    {
      "mask_id": 1,
      "bbox_voxel": [10, 25, 30, 50, 40, 60],
      "bbox_mm": [40.0, 100.0, 120.0, 200.0, 160.0, 240.0],
      "volume_cc": 1.5,
      "confidence": 0.85
    }
  ]
}
```

### 自定义配置

修改 `configs/unet_fl70.yaml` 以调整：

- **批次大小** (如果 OOM):
  ```yaml
  training:
    batch_size: 1  # 从 2 降至 1
  ```

- **学习率**:
  ```yaml
  training:
    learning_rate: 5.0e-5  # 从 1e-4 降低
  ```

- **数据增强**:
  ```yaml
  augmentation:
    random_rotation:
      enabled: false  # 禁用旋转
  ```

### 常见问题

#### Q: 显存不足 (CUDA out of memory)

**A**: 减少批次大小或 patch 大小
```yaml
training:
  batch_size: 1
data:
  patch_size: [32, 32, 32]
```

#### Q: 训练很慢

**A**: 
1. 检查是否使用 GPU: `python -c "import torch; print(torch.cuda.is_available())"`
2. 减少数据增强
3. 增加批次大小（如果显存允许）

#### Q: 召回率低

**A**:
1. 运行阈值敏感性分析（自动完成）
2. 增加病灶 patch 比例
3. 调整 Focal Tversky Loss 参数

#### Q: 如何继续训练？

**A**: 当前不支持断点续训，需要从头训练。可以修改代码添加 checkpoint 加载功能。

### 输出文件说明

```
Light-3D-Unet-Front/
├── models/
│   ├── best_model.pth          # 最佳模型 (基于验证召回率)
│   └── checkpoints/            # 训练检查点
├── logs/
│   ├── tensorboard/            # TensorBoard 日志
│   └── training_history.json   # 训练历史
├── inference/
│   ├── prob_maps/              # 概率图 (NIfTI)
│   ├── bboxes/                 # 边界框 (JSON)
│   ├── metrics.csv             # 评估指标
│   └── detailed_results.json   # 详细结果
└── data/
    ├── processed/              # 预处理后数据
    └── splits/                 # 数据划分
```

### 下一步

1. **验证结果**: 查看 `inference/metrics.csv`
2. **调整参数**: 如果性能不理想，修改 `configs/unet_fl70.yaml`
3. **撰写报告**: 使用模板准备实验报告
4. **分析失败案例**: 找出模型的不足之处

### 获取帮助

查看详细文档：
- 英文: `README.md`
- 中文: `claude.md`

每个脚本都有帮助信息：
```bash
python scripts/train.py --help
python scripts/inference.py --help
python scripts/evaluate.py --help
```

### 性能基准

在配置正确的情况下（NVIDIA GPU，16GB+ RAM）：

- **预处理**: ~2-10 分钟（取决于数据量）
- **训练**: ~2-6 小时（200 epochs，可能提前停止）
- **推理**: ~5-15 分钟（验证集）
- **评估**: ~1-2 分钟

### 重要提示

⚠️ **数据隔离规则**：
- ✅ 仅使用 FL 训练集和验证集
- ❌ 不使用 DLBCL 数据
- ❌ 不使用 FL 测试集（黑盒）
- ❌ 不重新计算 SUV

⚠️ **可复现性**：
- 所有随机操作使用 seed=42
- 保存环境信息: `pip freeze > environment.txt`
- 记录 Git commit: `git log -1 > git_commit.txt`

### 联系支持

如有问题，请查阅：
1. `README.md` - 完整文档
2. `claude.md` - 中文详细说明
3. GitHub Issues - 报告问题

---

**祝您训练顺利！** 🚀
