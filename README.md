# AD_RF_Analysis

Code and tutorial notebooks for **"Deep Learning Analysis of Retinal Structures and Risk Factors of Alzheimer’s Disease"**.

This README is a step-by-step guide for running the tutorial code with explicit path requirements.

## Publication

Deep Learning Analysis of Retinal Structures and Risk Factors of Alzheimer’s Disease  
Conference  
Seowung Leem, Yunchao Yang, Adam J. Woods, Ruogu Fang  
The 46th International Conference of the IEEE Engineering in Medicine and Biology Society, Orlando, FL  
Publication Date: July 15-19, 2024

Risk Factor Prediction & Analysis using fundus image. Funded by NSF

## Abstract

Using artificial intelligence, we trained computer models to analyze standard retinal images commonly taken during eye exams. These models could accurately estimate several AD-related risk factors, including age, blood pressure, diabetes, smoking, sleep problems, and depression. The AI consistently focused on meaningful regions of the eye, especially blood vessels and the optic nerve, which are closely linked to brain and vascular health. Notably, some eye-based patterns differed in people who later developed AD, nearly nine years before their diagnosis. While this approach does not diagnose Alzheimer’s disease, it suggests that routine eye images may capture early biological changes associated with future risk. This work highlights the potential of regular eye exams as a low-cost, non-invasive way to study and monitor brain health long before symptoms begin.

## Repository layout

- `code/` – training scripts, Grad-CAM scripts, notebooks, and the `environment.yml`.
- `AutoMorph/` – external segmentation pipeline referenced by the tutorials.

## Step-by-step: set up the environment

1. **Create and activate the conda environment** (from the repo root):
   ```bash
   conda env create -f code/environment.yml
   conda activate <env-name>
   ```
   > Replace `<env-name>` with the `name:` defined in `code/environment.yml`.

2. **Install any extra dependencies required by Grad-CAM** (if not already in the environment):
   ```bash
   pip install pytorch-grad-cam
   ```

## Step-by-step: prepare data (path requirements)

The training scripts expect **UK Biobank** fundus images and a CSV file with an `eid` column plus the label columns you want to predict.

1. **Fundus image directory**
   - **Single-eye training scripts** (`train_classification_multi.py`, `train_regression_multi.py`) read images using:
     ```text
     <image_dir>/<eid><eye_code>
     ```
     Example (default): `1234567_21015_0_0.png`.

   - **Multi-eye training scripts** (`*_mlflow.py`) expect *two* directories:
     ```text
     <left_image_dir>/<eid><left_eye_code>
     <right_image_dir>/<eid><right_eye_code>
     ```

2. **CSV label file**
   - Must include `eid` as a string column.
   - Must include label columns referenced by `--label_code` (single label) or `--label` (multi-label).

3. **Path requirements checklist**
   - Use absolute paths or repo-relative paths.
   - Keep **trailing slashes** in `--image_dir`, `--left_image_dir`, and `--right_image_dir` to match the code’s string concatenation.
   - Ensure the `eye_code` suffixes match the filenames in the directory.

Example directory layout:
```
/data/ukb/
  fundus_left/
    1234567_21015_0_0.png
    7654321_21015_0_0.png
  fundus_right/
    1234567_21016_0_0.png
    7654321_21016_0_0.png
  labels.csv
```

## Step-by-step: run training

> These scripts use `torch.distributed` and should be launched with `torchrun`, even on a single GPU.

### 1) Classification (single-eye)
```bash
torchrun --nproc_per_node=1 code/train_classification_multi.py \
  --image_dir /data/ukb/fundus_left/ \
  --csv_dir /data/ukb/labels.csv \
  --eye_code _21015_0_0.png \
  --label_code 31-0.0 \
  --working_dir ViT_sex \
  --model_name ViT_sex
```

### 2) Regression (single-eye)
```bash
torchrun --nproc_per_node=1 code/train_regression_multi.py \
  --image_dir /data/ukb/fundus_left/ \
  --csv_dir /data/ukb/labels.csv \
  --eye_code _21015_0_0.png \
  --label_code 21003-0.0 \
  --working_dir ViT_age \
  --model_name ViT_age
```

### 3) Classification (multi-eye + multi-GPU)
```bash
torchrun --nproc_per_node=2 code/train_classification_multi_mlflow.py \
  --left_image_dir /data/ukb/fundus_left/ \
  --right_image_dir /data/ukb/fundus_right/ \
  --csv_dir /data/ukb/labels.csv \
  --left_eye_code _21015_0_0.png \
  --right_eye_code _21016_0_0.png \
  --label 31-0.0 20116-0.0 \
  --working_dir Swin_classification \
  --model_name Swin_classification
```

### 4) Regression (multi-eye + multi-GPU)
```bash
torchrun --nproc_per_node=2 code/train_regression_multi_mlflow.py \
  --left_image_dir /data/ukb/fundus_left/ \
  --right_image_dir /data/ukb/fundus_right/ \
  --csv_dir /data/ukb/labels.csv \
  --left_eye_code _21015_0_0.png \
  --right_eye_code _21016_0_0.png \
  --label 21003-0.0 \
  --working_dir Swin_regression \
  --model_name Swin_regression
```

### Output locations
- Single-eye scripts save to `./savedmodel/<working_dir>/` by default.
- MLflow scripts write to `--logdir` (default: `./log`) and include their saved model checkpoints.

## Step-by-step: Grad-CAM visualization

The Grad-CAM scripts are **standalone** and have internal paths that you must update before running.

1. Edit the following inside the script you plan to run:
   - `get_saliency_map_classification()` in `code/GradCam_Visualization_Classification.py`
   - `get_saliency_map_regression()` in `code/GradCam_Visualization_Regression.py`

   Update:
   - `img_path` (output directory)
   - `model_path` (trained checkpoint)
   - `label_df` CSV path

2. Run the script:
   ```bash
   python code/GradCam_Visualization_Classification.py --feat 0
   python code/GradCam_Visualization_Regression.py --feat 0
   ```

## Step-by-step: segmentation & inference (AutoMorph + notebooks)

1. **AutoMorph segmentation**
   - Follow the instructions in `AutoMorph/` to generate segmentation maps and quality scores.
   - Point any outputs you generate to paths used in the notebooks below.

2. **Run notebooks**
   - `code/Tutorial.ipynb` – end-to-end examples, including overlap calculations.
   - `code/Correlation_Analysis.ipynb` – correlation analysis workflows.

   Launch Jupyter from the repo root:
   ```bash
   jupyter lab
   ```
   Then update any data paths inside the notebook cells to match your local directories.

## Notes

- The UK Biobank dataset requires approved access. See: https://www.ukbiobank.ac.uk/.
- Multi-GPU runs require a CUDA-enabled environment with NCCL configured.
- If you see path errors, confirm your directory **trailing slashes** and filename suffixes.
