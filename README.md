# Lightweight 3D U-Net for FL Lesion Recall

This repository implements a lightweight 3D U-Net for PET-only lesion candidate detection using Follicular Lymphoma (FL) data. The system is designed for high recall with coarse masks and bounding boxes, featuring a modular architecture and mixed-domain training capabilities.

## Project Overview

**Core Objective**: Build and validate a lightweight 3D U-Net for lesion candidate recall using FL-70% data.

**Key Features**:
- **Lightweight Architecture**: 217K parameters (16→32→64→128 channels).
- **Advanced Training**: Step-based mixed training (FL + DLBCL) with FL-only validation.
- **Robust Preprocessing**: Path B (4×4×4mm preservation) + Body Mask generation.
- **Loss Function**: Focal Tversky Loss for imbalanced segmentation.
- **Modular Design**: Separated core logic (`light_unet`) from execution scripts.

## Project Structure

```text
Light-3D-Unet-Front/
├── configs/                  # Configuration files
│   ├── unet_fl70.yaml        # Main FL experiment config
│   └── unet_mixed_fl_dlbcl.yaml # Mixed training config
├── light_unet/               # Core source package
│   ├── core/                 # Trainer and Inferencer logic
│   ├── models/               # U-Net architecture and losses
│   ├── data/                 # Dataset and DataLoader
│   └── utils/                # Metrics and helper functions
├── scripts/                  # Execution scripts (internal use, called by main.py)
├── docs/                     # Documentation and templates
│   ├── archive/              # Archived logs and legacy docs
│   └── templates/            # Report templates
├── data/                     # Data storage
│   ├── raw/                  # Raw input images/labels (User provided)
│   ├── processed/            # Preprocessed data (Auto-generated)
│   └── splits/               # Train/val/test split lists
├── main.py                   # Single entry point for all stages
└── requirements.txt          # Python dependencies

```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Light-3D-Unet-Front

```


2. **Setup Environment**:
```bash
bash setup.sh
source venv/bin/activate

```



## Data Preparation

Place your FL data in the `data/raw/` directory following this structure:

```text
data/raw/
├── images/
│   ├── 0001_0000.nii.gz
│   └── ...
└── labels/
    ├── 0001.nii.gz
    └── ...

```

## Usage

The pipeline is orchestrated via `main.py`.

### 1. Run Full Pipeline

```bash
python main.py --mode all

```

### 2. Run Individual Stages

**Preprocessing**:
Generates body masks, normalizes intensity, and prepares metadata.

```bash
python main.py --mode preprocess

```

**Training**:
Starts training using `configs/unet_fl70.yaml` by default.

```bash
python main.py --mode train

```

**Inference & Evaluation**:

```bash
python main.py --mode inference
python main.py --mode evaluate

```

### 3. Mixed Training (Optional)

To enable mixed FL + DLBCL training (using step-based strategy):

```bash
python main.py --mode train --config configs/unet_mixed_fl_dlbcl.yaml

```

## Configuration

Key hyperparameters are defined in `configs/unet_fl70.yaml`:

* **Data**: Spacing (4mm), patch size (48x48x48), body mask settings.
* **Model**: Channels, dropout, normalization.
* **Training**: Learning rate, batch size, mixed training strategies.

## Documentation

* **[Chinese Guide (中文文档)](claude.md)**: Detailed implementation details and rules.
* **[Report Template](https://www.google.com/search?q=docs/templates/EXPERIMENT_REPORT_TEMPLATE.md)**: Template for experiment reporting.
* **[Mixed Training Guide](https://www.google.com/search?q=docs/MIXED_TRAINING_GUIDE.md)**: Details on step-based training strategy.

## License

[To be specified]

