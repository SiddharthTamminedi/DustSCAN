# 🌪️ DustSCAN: Semantic Segmentation of Dust Plumes

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**DustSCAN** is an advanced deep learning pipeline designed to automatically detect and segment airborne dust plumes from SEVIRI (Spinning Enhanced Visible and InfraRed Imager) satellite imagery. Utilizing a highly optimized **UNet++** architecture with Automatic Mixed Precision (AMP), the model accurately classifies dust presence on a pixel-by-pixel basis across complex meteorological backgrounds.

---

## 📁 Project Structure

The repository is modularized for scalable research and production:

```text
DustSCAN/
│
├── src/                    # Core logic and modules
│   ├── dataset.py          # PyTorch Dataset for parsing NetCDF satellite data
│   ├── models.py           # UNet++ model architecture definitions
│   └── utils.py            # Normalization stats, constants, and helper functions
│
├── scripts/                # Executable pipelines
│   ├── run_training.py          # Main training loop with AMP and checkpointing
│   ├── test_model.py            # Inference script that saves predictions
│   └── generate_visualizations.py # Extracts raw dust storm RGBs from the dataset
│
├── notebooks/              # Analysis and Reporting
│   ├── dust_analysis.ipynb          # Comprehensive multi-year EDA on dust distributions
│   └── visualize_performance.ipynb  # TensorBoard log parsing & training metrics analysis
│
└── outputs/                # Generated Artifacts
    ├── models/             # Saved PyTorch weights (.pth)
    ├── runs/               # TensorBoard event logs
    └── visuals/            # Exported inference plots and heatmaps
```

---

## 📊 1. Dataset & Exploratory Data Analysis (EDA)

The project utilizes multi-year NetCDF satellite datasets (e.g., `DustSCAN_2021.nc`, `DustSCAN_2022.nc`). 

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
   *(Requires: `torch`, `xarray`, `netCDF4`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `tensorboard`, `nbformat`)*

3. **Data Placement:**
   Ensure the `DustSCAN_2021.nc` and `DustSCAN_2022.nc` datasets are downloaded. By default, scripts look for these in `~/Downloads/`. Update the file paths in the scripts if stored elsewhere.

---

## 🧠 3. Model Training

The pipeline leverages a robust UNet++ model. To initiate training:

```bash
python scripts/run_training.py
```

**Training Highlights:**
- **Loss Function:** Combined Binary Cross Entropy (BCE) and Dice Loss to handle extreme class imbalance (as dust pixels are sparse).
- **Optimization:** AdamW optimizer coupled with a learning rate scheduler.
- **Efficiency:** Utilizes PyTorch Automatic Mixed Precision (AMP) for faster iteration and reduced VRAM usage.
- **Checkpointing:** The best model weights are automatically saved to `outputs/models/best_dustscan_model.pth`.

---

## 📈 4. Evaluation & Performance Tracking

Monitor the model's training and validation metrics seamlessly.

### Training Metrics
Run the dedicated performance notebook to parse TensorBoard logs and visualize learning curves:
- Open `notebooks/visualize_performance.ipynb`
- Features graphs for **Train/Val Loss**, **IoU**, and **F1 Scores**.

### Inference & Visualizing Predictions
To evaluate the model against the validation set and visualize the predicted masks versus the ground truth, run:

```bash
python scripts/test_model.py
```

**Features of the Inference Script:**
- Automatically computes Intersection over Union (IoU) for all predictions.
- Intelligently filters and saves predictions of high interest:
  - **High Quality:** Saves up to 15 predictions with $\ge$ 0.80 IoU.
  - **Average Quality:** Saves up to 5 representative predictions with IoU between 0.65 and 0.75.
- Exports cleanly named plots (e.g., `pred_1.png`) to the `outputs/visuals/` directory.

---

## 📝 License
This project is licensed under the MIT License.