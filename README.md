# CoPAS-STMRI (Co-Plane Attention for Sequential MRI)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://www.apache.org/licenses/LICENSE-2.0)

## Project Overview

**Full name:** CoPAS-STMRI (Co-Plane Attention for Sequential MRI)  
**Abbreviation:** STMRI  
**One-line description:** STMRI is a multi-phase MRI modeling framework for medical image representation learning and lesion-related prediction/reconstruction tasks.

This repository is a cleaned open-source package derived from the CoPAS research codebase.  
The core model is `STMRI` in `main/model/model_my.py`.

## News / Updates

- **2026-03-16**: Initial open-source release with cleaned STMRI training/inference pipeline.
- **2026-03-16**: Added dual dataset format support (`npy_table` and CoPAS-like `long_table`).

## Installation

### 1) Clone

```bash
git clone <YOUR_REPO_URL>
cd CoPAS_OpenSource
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The loader supports two formats.

### Format A: `npy_table` (recommended for portability)

Required columns:
- `patient_id`
- `label`
- `ph0`, `ph1`, `ph2`, `ph3` (relative `.npy` volume paths)

Example:

```csv
patient_id,label,ph0,ph1,ph2,ph3
case001,1,volumes/case001_ph0.npy,volumes/case001_ph1.npy,volumes/case001_ph2.npy,volumes/case001_ph3.npy
```

### Format B: `long_table` (closer to original CoPAS table)

Required columns (default names):
- `姓名` (patient id)
- `label`
- `view` (e.g. `ph0`~`ph3`)
- `子序列`
- `图序列`

When `--data_format long_table` is enabled, these Chinese field names are used by default automatically.

Path rule (default):
`<data_root>/<姓名>/<子序列>/<图序列>.png`

You can override field names in CLI:
`--patient_col --view_col --sub_seq_col --image_col --image_ext`

Template files are provided in `data/`:
- `sample_train_npy.csv`
- `sample_valid_npy.csv`
- `sample_long_table.csv`

## Quick Start

All runtime arguments are managed in `main/run/Args.py` and overridable from CLI.

### Train (`npy_table`)

```bash
python run.py   --data_format npy_table   --train_csv data/sample_train_npy.csv   --valid_csv data/sample_valid_npy.csv   --data_root data   --experiment_name stmri_exp1   --class_num 2   --epochs 50   --batch_size 2   --lr 5e-5
```

### Train (`long_table`, CoPAS-like)

```bash
python run.py   --data_format long_table   --train_csv data/train_long.csv   --valid_csv data/valid_long.csv   --data_root data
```

### Test / Inference

```bash
python run.py --test   --data_format npy_table   --valid_csv data/sample_valid_npy.csv   --data_root data   --class_num 2   --weight_path outputs/stmri_exp1/best_model.pth
```

Testing will export predictions to:
`outputs/<experiment_name>/test_predictions.csv`

### Optional: Data Sanity Check

Before training, you can validate CSV schema and basic path availability:

```bash
python -m main.run.check_data \
  --data_format long_table \
  --train_csv data/train_long.csv \
  --valid_csv data/valid_long.csv \
  --data_root data
```

## Directory Structure

```text
CoPAS_OpenSource/
├── run.py
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── sample_train_npy.csv
│   ├── sample_valid_npy.csv
│   └── sample_long_table.csv
└── main/
    ├── data/
    │   └── datasets.py
    ├── model/
    │   ├── model_my.py          # STMRI core model
    │   ├── ResNet3D.py
    │   ├── kan.py
    │   └── otk/
    │       ├── layers.py
    │       ├── sinkhorn.py
    │       └── utils.py
    └── run/
        ├── Args.py
        ├── check_data.py
        └── train.py
```

## Citation

If you find this project useful, please cite your corresponding paper version:

```bibtex
@article{copas_stmri_2026,
  title   = {<TO_BE_FILLED>},
  author  = {<TO_BE_FILLED>},
  journal = {<TO_BE_FILLED>},
  year    = {2026},
  doi     = {<TO_BE_FILLED>}
}
```

## License

This project is released under the **Apache License 2.0**.
