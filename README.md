# DustSCAN Segmentation Project

This repository contains the training pipeline for a semantic segmentation model designed to detect dust plumes from SEVIRI satellite imagery using an advanced UNet++ architecture.

## Project Structure

The project has been organized into a modular structure:

-   **`src/`**: Contains the core logic (dataset parsing, model definitions, training loops, and utilities).
-   **`scripts/`**: Executable scripts. Run the files in here to train models or generate visuals.
-   **`outputs/`**: All generated artifacts (model weights, tensorboard logs, visualization images) are saved here.
-   **`notebooks/`**: Contains exploratory data analysis and Jupyter notebooks.

## Quick Start

### 1\. Download Dataset

Ensure the `DustSCAN_2022.nc` dataset is downloaded to `~/Downloads/`. If it is located somewhere else, you will need to update the path in `scripts/run_training.py`.

### 2\. Train the Model

To start training with the pipeline, run:

powershell

Copy

```powershell
python scripts/run_training.py
```

This will automatically save your best model weights to `outputs/models/` and tensorboard metrics to `outputs/runs/`.

### 3\. Generate Visualizations

To run the visualization script and extract the dust storm images from the raw dataset, run:

powershell

Copy

```powershell
python scripts/generate_visualizations.py
```

The output images will be saved directly to `outputs/visualizations/`.