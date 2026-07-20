# Scaling & Fine-Tuning Guide 

This document explains how to scale the **DustSCAN** model for superior segmentation performance on satellite imagery, and provides a comprehensive guide on how to fine-tune the current architecture (`UNet++` with `EfficientNet-B4` encoder) on localized or new datasets.

---

## Part 1: Scaling the Model for Better Results

To achieve higher IoU (Intersection over Union) and F1 scores, scaling can be approached from three axes: **Model Capacity**, **Data & Input Features**, and **Compute & Training Hyperparameters**.

---

### 1. Scaling Model Capacity (Encoder & Decoder)

The current model uses `efficientnet-b4` as the backbone encoder. You can scale the backbone capacity within the `segmentation_models_pytorch` (SMP) framework.

#### A. Selecting Larger Encoders
You can swap the encoder name in [src/models.py] to a larger variant:

| Encoder Name                  | Parameters (Approx) | VRAM Consumption   | Recommended Batch Size | Primary Use Case                   |
| :---------------------------- | :------------------ | :----------------- | :--------------------- | :--------------------------------- |
| `efficientnet-b4` *(Current)* | ~19M                | Low (4-8 GB)       | 32 (with AMP)          | Baseline & rapid iteration         |
| `efficientnet-b7`             | ~66M                | High (12-16 GB)    | 8 - 16                 | High-capacity spatial extraction   |
| `resnet101`                   | ~44M                | Medium (8-12 GB)   | 16 - 24                | Strong residual gradients          |
| `mit_b5` (SegFormer)          | ~82M                | Very High (>16 GB) | 8                      | Transformer-based global attention |

#### B. Swapping the Architecture
If boundary refinement is your primary bottleneck, consider swapping `smp.UnetPlusPlus` to other structures:
* **`smp.MAnet`**: Highly recommended for multi-scale attention in remote sensing.
* **`smp.DeepLabV3Plus`**: Excellent at retaining sharp object boundaries using Atrous Spatial Pyramid Pooling (ASPP).

#### C. Adjusting Dimension Adaptation (Padding)
Larger encoders may have different downsampling steps (e.g., up to $32\times$ or $64\times$). The padding/cropping code in [src/models.py] must be updated accordingly. For example, if an encoder requires dimensions to be multiples of 64:
* Raw size: `(148, 357)`
* Padded size: Next multiples of 64 are 192 and 384.
* Horizontal padding: $384 - 357 = 27$
* Vertical padding: $192 - 148 = 44$
* Pad code: `F.pad(x, (0, 27, 0, 44))`
* Crop code: `out[:, :, :148, :357]`

---

### 2. Data & Input Feature Scaling

In remote sensing, scaling the feature space is often more effective than simply increasing model size.

#### A. Temporal & Dataset Expansion
* **Multi-Year Saturation**: Train on a multi-year archive (e.g., 2018–2026) instead of just two years to cover solar cycle variations and regional dust patterns.


#### B. Scaling Input Channels
You can scale the inputs from **5 channels** to **8+ channels** by adding raw thermal infrared (TIR) brightness temperature differences (BTDs) which are physically sensitive to silicate dust:

$$BTD_{\text{dust}} = BT_{12.0} - BT_{10.8}$$
$$BTD_{\text{moisture}} = BT_{10.8} - BT_{8.7}$$

To incorporate this, modify the dataset assembly in [src/dataset.py] to read these variables from the NetCDF, normalize them, and expand `in_channels` in `smp.UnetPlusPlus(..., in_channels=8)`.

---

### 3. Compute & Hyperparameter Scaling

When scaling model size, adjust training hyperparameters to avoid overfitting or instability:

* **Gradient Accumulation**: If VRAM limits your batch size to 4 or 8 with larger encoders, set `accumulation_steps=8` or `16` in [scripts/run_training.py] to simulate a stable batch size of 64.
* **Mixed Precision (AMP)**: Always keep `GradScaler` active to save VRAM and increase throughput.
* **Distributed Data Parallel (DDP)**: For multi-GPU environments, launch training using:
  ```bash
  torchrun --nproc_per_node=NUM_GPUS scripts/run_training.py
  ```
  *(Note: You will need to wrap the model in `torch.nn.parallel.DistributedDataParallel` and use `DistributedSampler` in the loaders).*

---
## Part 2: Fine-Tuning the Current Model

Fine-tuning allows you to adapt the pretrained weights of `best_dustscan_model.pth` to new regions, different satellite instruments (e.g., Himawari-8/9 AHI, GOES-R ABI), or specific high-difficulty seasons without starting training from scratch.

---

### 1. Key Principles of Fine-Tuning

When adapting the existing model, follow these three rules to prevent **catastrophic forgetting**:

1. **Lower the Learning Rate**: Reduce the learning rate by a factor of 10 to 100 relative to the initial training rate. Use a rate of $1 \times 10^{-5}$ or $3 \times 10^{-5}$ (instead of $1 \times 10^{-4}$).
2. **Freeze the Encoder**: The early layers of the encoder extract basic features (edges, textures, color gradients) that generalize well. Freeze them and only train the attention blocks, decoder, and segmentation head.
3. **Calibrate class weights**: If the target fine-tuning region has different dust occurrences, adjust the `pos_weight` and focal loss `alpha` in the compound loss.

---

### 2. Implementation: Code Walkthrough for Fine-Tuning

To fine-tune, you can create a dedicated script `scripts/run_finetuning.py` or modify the training loop parameters. Below is the Python code setup for loading, freezing, and setting up differential learning rates:

```python
import torch
import torch.nn as nn
from src.models import build_advanced_unet_model, FocalDiceBCELoss
from src.train import train_model

def setup_finetuning(model_path, device):
    # 1. Instantiate the model architecture
    model = build_advanced_unet_model().to(device)
    
    # 2. Load the best-saved checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded pretrained weights from {model_path}")
    
    # 3. Freeze the encoder layers
    # This prevents the pretrained EfficientNet-B4 features from being distorted
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    print("Encoder layers successfully frozen.")
    
    # 4. Set up differential learning rates (optional but recommended)
    # Give the decoder and head a slightly larger learning rate than the encoder (if unfrozen)
    # Or optimize only parameters that require gradients
    optimizer_params = [
        {"params": [p for p in model.model.decoder.parameters() if p.requires_grad], "lr": 1e-4},
        {"params": [p for p in model.model.segmentation_head.parameters() if p.requires_grad], "lr": 1e-4}
    ]
    
    return model, optimizer_params

# Example execution within training pipeline:
# optimizer = torch.optim.AdamW(optimizer_params, weight_decay=1e-4)
```

---

### 3. Step-by-Step Fine-Tuning Guide

#### Step 1: Prepare the Fine-Tuning Dataset
Create a separate NetCDF file (e.g., `DustSCAN_Finetune.nc`) containing your target images. Place it in your data directory.

#### Step 2: Create the Fine-Tuning Script
Create a new file [scripts/run_finetuning.py] with the following content:

```python
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model
from src.models import build_advanced_unet_model

if __name__ == "__main__":
    # Paths to your fine-tuning data
    FINETUNE_DATA = os.path.expanduser("~/Downloads/DustSCAN_Finetune.nc")
    PRETRAINED_CHECKPOINT = "outputs/models/best_dustscan_model.pth"
    
    if not os.path.exists(FINETUNE_DATA):
        print(f"Fine-tuning dataset not found at {FINETUNE_DATA}. Using default files.")
        # Fallback to existing files for demonstration
        FINETUNE_DATA = os.path.expanduser("~/Downloads/DustSCAN_2022.nc")
        
    if os.path.exists(FINETUNE_DATA):
        print(f"Starting Fine-Tuning Pipeline using: {FINETUNE_DATA}")
        
        # We call the training logic with a low learning rate
        # train_model automatically checks if resume_path exists and loads it
        trained_model = train_model(
            nc_file_paths=[FINETUNE_DATA], 
            epochs=10,                      # Lower number of epochs is sufficient for fine-tuning
            batch_size=16,                  # Adjust based on VRAM
            accumulation_steps=2, 
            lr=3e-5,                        # Low learning rate for fine-tuning
            resume_path=PRETRAINED_CHECKPOINT
        )
    else:
        print("No fine-tuning datasets found. Exiting.")
```

#### Step 3: Run the Fine-Tuning Script
Execute the fine-tuning pipeline from your terminal:
```bash
python scripts/run_finetuning.py
```


