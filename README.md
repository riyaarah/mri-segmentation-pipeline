# 🧠 MRI Brain Tumor Segmentation

End-to-end deep learning pipeline for automated brain tumor segmentation using 3D U-Net architecture trained on the BraTS 2021 dataset.

## Results

| Metric | Score |
|--------|-------|
| Mean Dice Score | **0.52** |
| Dataset | BraTS 2021 (1,251 patients) |
| Architecture | 3D U-Net |
| Input | 4-modality MRI (FLAIR, T1, T1ce, T2) |

## Tumor Classes

| Class | Description | Color |
|-------|-------------|-------|
| Necrotic Core | Dead tumor tissue | 🔴 Red |
| Edema | Swelling around tumor | 🟢 Green |
| Enhancing Tumor | Active growing region | 🔵 Blue |

## Architecture

The model uses a 3D U-Net with:
- Encoder path: 4 levels with DoubleConv3D blocks (32 → 64 → 128 → 256 channels)
- Bottleneck: 512 channels
- Decoder path: 4 upsampling levels with skip connections
- Dropout (0.1) for regularization
- Total parameters: 22.5M

## Pipeline

```
Raw MRI (.nii.gz)
    ↓
Preprocessing (normalize + crop to 96³)
    ↓
3D U-Net Inference
    ↓
Tumor Mask (4 classes)
    ↓
Streamlit Visualization
```

## Tech Stack

- **Deep Learning:** PyTorch, MONAI
- **MRI Processing:** Nibabel, SimpleITK
- **Demo:** Streamlit
- **Dataset:** BraTS 2021 (Kaggle)
- **Training:** Google Colab T4 GPU

## Project Structure

```
mri-segmentation-pipeline/
├── app.py                  ← Streamlit demo app
├── download_data.py        ← Dataset download script
├── notebooks/
│   ├── Untitled.ipynb      ← Full training notebook
│   └── results/
│       └── training_curves.png
└── src/
    └── __init__.py
```

## Setup & Run

```bash
# Clone repo
git clone https://github.com/riyaarah/mri-segmentation-pipeline.git
cd mri-segmentation-pipeline

# Install dependencies
pip install torch torchvision
pip install monai nibabel SimpleITK streamlit

# Run demo
streamlit run app.py
```

## Demo

The Streamlit app allows you to:
- Upload 4 MRI modalities (FLAIR, T1, T1ce, T2) in .nii.gz format
- Run automated 3D segmentation
- Navigate brain slices interactively
- View tumor regions color-coded by class
- See percentage statistics for each tumor region

## Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-4) |
| Loss | Dice Loss + Cross Entropy |
| Batch Size | 2 |
| Input Size | 96 × 96 × 96 |
| Augmentation | Random flips (x, y, z axes) |
| Scheduler | ReduceLROnPlateau |

## Dataset

BraTS 2021 Task 1 — Brain Tumor Segmentation Challenge
- 1,251 multi-institutional MRI scans
- 4 MRI modalities per patient: T1, T1ce, T2, FLAIR
- Expert-annotated tumor regions
- Available on Kaggle: `dschettler8845/brats-2021-task1`

---
*Built with PyTorch · 3D U-Net · BraTS 2021*
