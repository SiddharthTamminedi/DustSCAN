# 🌪️ DustSCAN: Semantic Segmentation of Dust Plumes

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**DustSCAN** is an advanced deep learning pipeline designed to automatically detect and segment airborne dust plumes from SEVIRI (Spinning Enhanced Visible and InfraRed Imager) satellite imagery. Utilizing a highly optimized **UNet++** architecture with an **EfficientNet-B4** encoder, **SCSE attention**, and Automatic Mixed Precision (AMP), the model accurately classifies dust presence on a pixel-by-pixel basis across complex meteorological backgrounds.

---

## 📁 Project Structure

The repository is modularized for scalable research and production:

```text
DustSCAN/
│
├── src/                    # Core logic and modules
│   ├── __init__.py         # Package initializer
│   ├── dataset.py          # PyTorch Dataset for parsing NetCDF satellite data
│   ├── models.py           # UNet++ model architecture and custom loss functions
│   ├── train.py            # Training loop with AMP, gradient accumulation, and checkpointing
│   └── utils.py            # Normalization stats, constants, and visualization helpers
│
├── scripts/                # Executable pipelines
│   ├── run_training.py             # Entry point to launch model training
│   ├── test_model.py               # Inference on unseen off-season data with visualization
│   ├── ablation_study.py           # Feature importance analysis via input ablation
│   └── generate_visualizations.py  # Extracts raw dust storm RGBs from the dataset
│
├── notebooks/              # Analysis and Reporting
│   ├── dust_analysis.ipynb          # Comprehensive multi-year EDA on dust distributions
│   └── visualize_performance.ipynb  # TensorBoard log parsing & training metrics analysis
│
└── outputs/                # Generated Artifacts
    ├── models/             # Saved PyTorch weights (.pth)
    ├── runs/               # TensorBoard event logs
    ├── predictions/        # Exported inference plots (RGB / GT / Pred triplets)
    └── visualizations/     # Dataset samples and ablation study plots
```

---

## 📊 1. Dataset & Exploratory Data Analysis (EDA)

The project utilizes multi-year NetCDF satellite datasets (e.g., `DustSCAN_2021.nc`, `DustSCAN_2022.nc`).

### Input Features (5-Channel)

Each sample fed to the model is a **5-channel tensor** composed of:

| Channel | Source Variable                        | Normalization             |
| ------- | -------------------------------------- | ------------------------- |
| 0–2     | `dust_rgb` (SEVIRI Dust RGB composite) | ImageNet mean/std         |
| 3       | `sun_zenith` (Solar Zenith Angle)      | Z-score (global mean/std) |
| 4       | `pdi` (Pink Dust Index)                | Z-score (global mean/std) |

### Exploratory Analysis

Before training, thorough analysis of the physical characteristics of dust is conducted via **`notebooks/dust_analysis.ipynb`**. This includes:
- **Inter-Annual & Seasonal Trends:** Analyzing dust frequency variations across multiple years.
- **Spatial Heatmaps:** Identifying the geographical zones most prone to dust storms.
- **Diurnal Cycles:** Tracking dust presence across the 24-hour UTC cycle.
- **Physical Feature Distributions:** Comparing Pink Dust Index (PDI) and Sun Zenith angles for dusty vs. clear skies.
- **Cloud Mask Interaction:** Evaluating how often dust detection intersects with clear skies versus cloudy conditions (accounting for EUMETSAT cloud mask encodings).

---

## 🚀 2. Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/DustSCAN.git
   cd DustSCAN
   ```

2. **Install dependencies:**
   Ensure you have PyTorch installed for your specific CUDA version, then install the remaining requirements:
   ```bash
   pip install -r requirements.txt
   ```
   *(Requires: `torch`, `segmentation_models_pytorch`, `xarray`, `netCDF4`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `tensorboard`, `tqdm`, `nbformat`)*

3. **Data Placement:**
   Ensure the `DustSCAN_2021.nc` and `DustSCAN_2022.nc` datasets are downloaded. By default, scripts look for these in `~/Downloads/`. Update the file paths in the scripts if stored elsewhere.

---

## 🧠 3. Model Architecture & Training

### Architecture

The model is built on **UNet++** from `segmentation_models_pytorch`:

- **Encoder:** EfficientNet-B4 (ImageNet-pretrained)
- **Decoder Attention:** Spatial and Channel Squeeze & Excitation (SCSE)
- **Input Channels:** 5 (Dust RGB + Sun Zenith + PDI)
- **Output:** 1-channel binary segmentation logits
- **Padding:** Input is padded to match encoder stride requirements and cropped back to 148×357 after decoding.

### Training

To initiate training:

```bash
python scripts/run_training.py
```

**Training Highlights:**
- **Loss Function:** Compound **Focal + Dice + BCE** loss (`FocalDiceBCELoss`) with configurable focal parameters (`α=0.90`, `γ=2.0`) and positive class weighting (`pos_weight=15.0`) to handle extreme class imbalance.
- **Optimization:** AdamW optimizer with weight decay (`1e-4`) and CosineAnnealingWarmRestarts learning rate scheduler.
- **Gradient Accumulation:** Configurable accumulation steps for effective larger batch sizes on limited VRAM.
- **Gradient Clipping:** Max norm clipping (`1.0`) for stable training.
- **Efficiency:** PyTorch Automatic Mixed Precision (AMP) with `GradScaler` for faster iteration and reduced VRAM usage. TF32 and cuDNN benchmarking enabled on CUDA.
- **Data Filtering:** Training data is filtered to active dust season months (March–August) and only timesteps with significant plume activity (>100 dust pixels).
- **Augmentation:** Random horizontal and vertical flips during training.
- **Checkpointing:** Best model weights (by validation F1 score) are automatically saved to `outputs/models/best_dustscan_model.pth`. Training can be resumed from an existing checkpoint.

---

## 🌐 4. Web Application (Streamlit)

A local web interface is provided to quickly run inference on new, custom Dust RGB images downloaded from the internet or EUMETSAT viewers.

```bash
streamlit run app.py
```

**Features:**
- **Zero-Config Inference:** Upload a standard `.png` or `.jpg` Dust RGB image; the app automatically resizes it to the expected 148x357 spatial grid.
- **Physical PDI Reconstruction:** The app mathematically reconstructs the true Pink Dust Index (PDI) directly from the RGB pixels using the Euclidean distance from magenta (`PDI = 1 - (P_dist / MaxDist)`), identical to the original authors' physical methodology.
- **Dynamic Context:** Use the sidebar slider to set the approximate Sun Zenith Angle to match the time of day the image was captured, ensuring accurate probability scaling.
- **Threshold Control:** Interactively adjust the binarization threshold (default `0.5`) to tighten or loosen the strictness of the final dust mask.

---

## 📈 5. Evaluation & Performance Tracking

### Training Metrics

Monitor the model's training and validation metrics via TensorBoard:

```bash
tensorboard --logdir outputs/runs
```

Or use the dedicated notebook `notebooks/visualize_performance.ipynb` to parse TensorBoard logs and visualize learning curves for **Train/Val Loss**, **IoU**, and **F1 Scores**.

### Inference on Unseen Off-Season Data

To evaluate model generalization, predictions are generated on **off-season months** (Jan, Feb, Sep–Dec) — data the model has never seen during training:

```bash
python scripts/test_model.py
```

**How it works:**
1. Filters the dataset to off-season months only.
2. Identifies timesteps with significant dust activity (>100 dust pixels).
3. Randomly selects 20 active dust storms for evaluation.
4. Generates side-by-side plots of **Input RGB → Ground Truth → Predicted Mask** for each sample.
5. Saves all plots to `outputs/predictions/`.

---

## 🔬 6. Ablation Study

To quantify the contribution of each input feature, run the ablation study:

```bash
python scripts/ablation_study.py
```

This script zeroes out individual input channels and measures the resulting drop in mean IoU relative to the full-model baseline. The analysis covers three ablation scenarios:

| Scenario            | Channels Zeroed | What It Tests                               |
| ------------------- | --------------- | ------------------------------------------- |
| No RGB              | Channels 0–2    | Importance of the SEVIRI Dust RGB composite |
| No Sun Zenith Angle | Channel 3       | Importance of solar geometry                |
| No PDI              | Channel 4       | Importance of the Pink Dust Index           |

Results are saved as a horizontal bar chart to `outputs/visualizations/feature_importance.png`.

---

## 📝 License
This project is licensed under the MIT License.