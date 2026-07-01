"""
DustSCAN U-Net Semantic Segmentation Pipeline (Optimized for 2022 Dataset)

Optimizations Included:
- Custom integration of ACTUAL 2022 dataset variables (dust_rgb, sun_zenith, pdi)

- ImageNet Normalization for dust_rgb (pretrained encoder compatibility)
- Z-Score Normalization for auxiliary channels (sun_zenith, pdi)

- Random Cropping & Data Augmentation (Flips, 90° Rotations)
- UnetPlusPlus Architecture with efficientnet-b3 backbone (5 Input Channels)
- Cloud-Masked Focal + Dice Loss
- Automatic Mixed Precision (AMP) & Gradient Accumulation
- Chunked NetCDF access for faster random I/O
- Learning Rate Scheduler & Best Model Checkpointing
- TensorBoard Logging
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

# =============================================================================
# 1. Global Normalization Statistics
# =============================================================================
# ImageNet stats for the pretrained encoder's first 3 channels (dust_rgb)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Z-score stats for auxiliary channels (computed from DustSCAN_2022.nc)
# NOTE: If applying to a new year or dataset, recompute these statistics!
GLOBAL_STATS = {
    'sun_zenith_mean': 89.064937,
    'sun_zenith_std': 43.909445,
    'pdi_mean': 0.5071591,
    'pdi_std': 0.1111518
}

# =============================================================================
# 2. Data Pipeline & Custom PyTorch Dataset
# =============================================================================

class DustSCANDataset(Dataset):
    """
    Custom PyTorch Dataset for the DustSCAN 2022 NetCDF file.
    Extracts dust_rgb (3 channels), sun_zenith (1 channel), and pdi (1 channel).
    Uses all timesteps (day and night) and relies solely on plume_id as ground truth.
    """
    def __init__(self, nc_file_path, patch_size=128, time_indices=None, mode='train',
                 samples_per_time_step=10):
        self.nc_file_path = nc_file_path
        self.patch_size = patch_size
        self.mode = mode
        self.ds = None  # Lazy loading in __getitem__ to support multi-worker safely
        
        try:
            with xr.open_dataset(self.nc_file_path, engine='netcdf4') as ds_temp:
                self.time_dim = 'time'
                self.lat_dim = 'lat'
                self.lon_dim = 'lon'
                
                self.total_time_steps = len(ds_temp[self.time_dim])
                self.height = len(ds_temp[self.lat_dim])
                self.width = len(ds_temp[self.lon_dim])
                
                if time_indices is None:
                    time_indices = list(range(self.total_time_steps))
                
        except Exception as e:
            raise FileNotFoundError(f"Error opening NetCDF file {self.nc_file_path}: {e}")

        # Use all timesteps (day + night)
        self.time_indices = time_indices
        
        if self.mode == 'train':
            self.samples_per_time_step = samples_per_time_step
            self.total_patches = len(self.time_indices) * self.samples_per_time_step
        else:
            # Precompute unique patch coordinates for validation
            self.val_coords = []
            seen = set()
            for r in range(0, self.height, self.patch_size):
                for c in range(0, self.width, self.patch_size):
                    r_start = min(r, max(0, self.height - self.patch_size))
                    c_start = min(c, max(0, self.width - self.patch_size))
                    if (r_start, c_start) not in seen:
                        seen.add((r_start, c_start))
                        self.val_coords.append((r_start, c_start))
            self.patches_per_time = len(self.val_coords)
            self.total_patches = len(self.time_indices) * self.patches_per_time
            
    def __len__(self):
        return self.total_patches
        
    @staticmethod
    def _normalize_zscore(data, mean, std):
        return (data - mean) / (std + 1e-8)
    
    @staticmethod
    def _normalize_imagenet(rgb):
        """Normalize RGB channels with ImageNet mean/std for pretrained encoder."""
        return (rgb - IMAGENET_MEAN) / IMAGENET_STD

    def _extract_patch(self, time_idx, row_start, col_start):
        """Extract a single patch from the dataset and return numpy arrays."""
        row_end = row_start + self.patch_size
        col_end = col_start + self.patch_size
        
        ds_slice = self.ds.isel({
            self.time_dim: time_idx,
            self.lat_dim: slice(row_start, row_end),
            self.lon_dim: slice(col_start, col_end)
        })
        
        dust_rgb = ds_slice['dust_rgb'].values  # already [0, 1] float32
        if dust_rgb.ndim == 3 and dust_rgb.shape[-1] == 3:  # (H, W, C)
            dust_rgb = dust_rgb.transpose(2, 0, 1)
        
        sun_zenith = ds_slice['sun_zenith'].values
        pdi = ds_slice['pdi'].values
        
        # ImageNet normalization for pretrained encoder (dust_rgb is already 0-1)
        dust_rgb = self._normalize_imagenet(dust_rgb.astype(np.float32))
        # Z-score only for auxiliary channels
        sun_zenith = self._normalize_zscore(sun_zenith, GLOBAL_STATS['sun_zenith_mean'], GLOBAL_STATS['sun_zenith_std'])
        pdi = self._normalize_zscore(pdi, GLOBAL_STATS['pdi_mean'], GLOBAL_STATS['pdi_std'])
        
        X = np.concatenate([
            dust_rgb,
            np.expand_dims(sun_zenith, axis=0),
            np.expand_dims(pdi, axis=0)
        ], axis=0).astype(np.float32)
        
        plume_id = ds_slice['plume_id'].values
        Y = (plume_id > 0).astype(np.float32)
        Y = np.expand_dims(Y, axis=0)
        
        return X, Y, (Y.sum() > 200)  # requires at least 200 dust pixels for balanced sampling

    def __getitem__(self, idx):
        if self.ds is None:
            # Lazy open for multi-processing compatibility
            self.ds = xr.open_dataset(self.nc_file_path, engine='netcdf4', chunks={'time': 1})
            
        if self.mode == 'train':
            time_idx_rel = idx // self.samples_per_time_step
            time_idx = self.time_indices[time_idx_rel]
            
            max_row = self.height - self.patch_size
            max_col = self.width - self.patch_size
            
            # Balanced sampling: 50% dust-containing patches, 50% random
            want_dust = (random.random() < 0.5)
            row_start = random.randint(0, max_row) if max_row > 0 else 0
            col_start = random.randint(0, max_col) if max_col > 0 else 0
            
            if want_dust:
                # OPTIMIZATION: Load just the plume mask for this timestep into RAM once
                # This prevents the DataLoader from making 100 separate disk reads per patch!
                full_plume = self.ds['plume_id'].isel(time=time_idx).values
                dust_coords = np.argwhere(full_plume > 0)
                
                if len(dust_coords) > 0:
                    for _attempt in range(100):
                        # Pick a random dust pixel
                        r, c = random.choice(dust_coords)
                        # Pick a random patch that contains this pixel
                        rs = random.randint(max(0, r - self.patch_size + 1), min(max_row, r))
                        cs = random.randint(max(0, c - self.patch_size + 1), min(max_col, c))
                        
                        # Validate that it has enough dust
                        patch_plume = full_plume[rs:rs+self.patch_size, cs:cs+self.patch_size]
                        if np.count_nonzero(patch_plume > 0) > 200:
                            row_start, col_start = rs, cs
                            break
                            
            X, Y, _ = self._extract_patch(time_idx, row_start, col_start)
        else:
            time_idx_rel = idx // self.patches_per_time
            time_idx = self.time_indices[time_idx_rel]
            
            patch_idx_spatial = idx % self.patches_per_time
            row_start, col_start = self.val_coords[patch_idx_spatial]
            X, Y, _ = self._extract_patch(time_idx, row_start, col_start)
        
        # Data Augmentation for Robust Training
        if self.mode == 'train':
            if random.random() > 0.5:
                X = np.flip(X, axis=2).copy()
                Y = np.flip(Y, axis=2).copy()
            if random.random() > 0.5:
                X = np.flip(X, axis=1).copy()
                Y = np.flip(Y, axis=1).copy()
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                X = np.rot90(X, k=k, axes=(1, 2)).copy()
                Y = np.rot90(Y, k=k, axes=(1, 2)).copy()
            
        X_tensor = torch.nan_to_num(torch.from_numpy(X))
        Y_tensor = torch.nan_to_num(torch.from_numpy(Y))
        
        return X_tensor, Y_tensor

# =============================================================================
# 3. Model Architecture (5 Channels)
# =============================================================================

def build_advanced_unet_model():
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3", 
        encoder_weights="imagenet",           
        in_channels=5,                  # 3 (dust_rgb) + 1 (sun_zenith) + 1 (pdi)
        classes=1,                      
        activation=None,
        decoder_attention_type="scse"
    )
    return model

# =============================================================================
# 4. Focal + Dice Loss (plume_id is ground truth for all pixels)
# =============================================================================

class FocalDiceLoss(nn.Module):
    """Focal + Dice loss. plume_id is the curated ground truth -- loss is
    computed over ALL pixels without cloud masking."""
    def __init__(self, alpha=0.85, gamma=2.0, pos_weight=10.0):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, y_pred_logits, y_true):
        pos_weight_tensor = torch.tensor([self.pos_weight], device=y_pred_logits.device)
        bce = F.binary_cross_entropy_with_logits(y_pred_logits, y_true, pos_weight=pos_weight_tensor, reduction='none')
        y_pred = torch.sigmoid(y_pred_logits)
        
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce
        focal_loss_mean = focal_loss.mean()
        
        # Dice Loss
        smooth = 1e-6
        intersection = (y_pred * y_true).sum(dim=(2, 3))
        union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
        dice_loss = 1.0 - (2. * intersection + smooth) / (union + smooth)
        dice_loss_mean = dice_loss.mean()
        
        return focal_loss_mean + dice_loss_mean

# =============================================================================
# 5. Evaluation Metrics & Training Loop
# =============================================================================

# No evaluate_metrics function needed -- TP/FP/FN accumulated directly in val loop

def train_model(nc_file_path, epochs=15, batch_size=8, accumulation_steps=4, lr=3e-4, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if device.type == 'cuda':
        # Enable CuDNN benchmark for maximum GPU throughput
        torch.backends.cudnn.benchmark = True
        # Enable TF32 for faster matmuls on modern GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    print(f"Starting training on device: {device} (Max Power Mode)")
    
    writer = SummaryWriter(log_dir="runs/dustscan_experiment")

    try:
        with xr.open_dataset(nc_file_path, engine='netcdf4') as ds:
            time_dim = 'time'
            total_time_steps = len(ds[time_dim])
    except Exception as e:
        print(f"Failed to open dataset: {e}")
        return None

    # Shuffle time indices to prevent seasonal bias in the splits
    all_indices = list(range(total_time_steps))
    random.seed(42)  # For reproducibility
    random.shuffle(all_indices)

    train_split_idx = int(0.8 * total_time_steps)
    train_indices = all_indices[:train_split_idx]
    val_indices = all_indices[train_split_idx:]

    train_dataset = DustSCANDataset(nc_file_path, patch_size=128, time_indices=train_indices, mode='train', samples_per_time_step=20)
    val_dataset = DustSCANDataset(nc_file_path, patch_size=128, time_indices=val_indices, mode='val')
    
    # Accelerate data loading (tuned for Laptop CPU/RAM limits)
    num_workers = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    model = build_advanced_unet_model().to(device)
    criterion = FocalDiceLoss(alpha=0.85, gamma=2.0, pos_weight=10.0)
    
    # --- NEW: Added weight decay for regularization ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # --- NEW: Learning Rate Scheduler ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)
    
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available())

    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (X, Y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            X, Y = X.to(device), Y.to(device)
            
            if epoch == 0 and batch_idx < 5:
                # Print the total number of positive dust pixels in this batch to verify sampling
                print(f"\nBatch {batch_idx+1} total dust pixels: {Y.sum().item()}")
                
            with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available()):
                outputs = model(X)
                loss = criterion(outputs, Y)
                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Step scheduler per optimizer update (correct for CosineAnnealingWarmRestarts)
                scheduler.step(epoch + batch_idx / len(train_loader))
                
            train_loss += loss.item() * accumulation_steps
            
        train_loss /= max(1, len(train_loader))
        writer.add_scalar('Loss/Train', train_loss, epoch)
        
        # Validation — accumulate TP/FP/FN across entire val set for proper dataset-level IoU/F1
        model.eval()
        val_loss = 0.0
        total_tp = 0.0
        total_fp = 0.0
        total_fn = 0.0
        
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")):
                X, Y = X.to(device), Y.to(device)
                
                with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available()):
                    outputs = model(X)
                    loss = criterion(outputs, Y)
                    
                val_loss += loss.item()
                
                # Accumulate TP/FP/FN for dataset-level metrics
                preds_bin = (torch.sigmoid(outputs) > 0.5).float()
                total_tp += (preds_bin * Y).sum().item()
                total_fp += (preds_bin * (1 - Y)).sum().item()
                total_fn += ((1 - preds_bin) * Y).sum().item()
                
                if batch_idx == 0:
                    imagenet_mean_t = torch.tensor(IMAGENET_MEAN.flatten(), device=X.device).view(1, 3, 1, 1)
                    imagenet_std_t  = torch.tensor(IMAGENET_STD.flatten(), device=X.device).view(1, 3, 1, 1)
                    display_rgb = torch.clamp(X[:, 0:3, :, :] * imagenet_std_t + imagenet_mean_t, 0, 1)
                    writer.add_images('Images/RGB', display_rgb, epoch)
                    writer.add_images('Masks/Ground_Truth', Y, epoch)
                    writer.add_images('Masks/Predictions', preds_bin, epoch)
                    
                    # Save a few predicted masks to disk for visual inspection
                    save_dir = os.path.join("visualizations", "epoch_predictions")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Save the first 3 images from this batch
                    num_to_save = min(3, display_rgb.size(0))
                    for i in range(num_to_save):
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        # rgb
                        img_np = display_rgb[i].cpu().permute(1, 2, 0).numpy()
                        axes[0].imshow(img_np)
                        axes[0].set_title("dust_rgb")
                        axes[0].axis("off")
                        # gt
                        gt_np = Y[i].cpu().squeeze().numpy()
                        axes[1].imshow(gt_np, vmin=0, vmax=1, cmap="gray")
                        axes[1].set_title("Ground Truth")
                        axes[1].axis("off")
                        # pred
                        pred_np = preds_bin[i].cpu().squeeze().numpy()
                        axes[2].imshow(pred_np, vmin=0, vmax=1, cmap="gray")
                        axes[2].set_title(f"Prediction (Epoch {epoch+1})")
                        axes[2].axis("off")
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1:02d}_sample_{i+1}.png"), dpi=100)
                        plt.close()
                
        val_loss /= max(1, len(val_loader))
        # Dataset-level IoU and F1 from accumulated counts
        val_iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
        val_f1  = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + 1e-6)
        
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Metrics/IoU', val_iou, epoch)
        writer.add_scalar('Metrics/F1_Score', val_f1, epoch)
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val IoU: {val_iou:.4f} | "
              f"Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_dustscan_model.pth")
            print(f"--> Saved New Best Model (F1: {best_f1:.4f})")
              
    writer.close()
    print("Training complete. Run 'tensorboard --logdir runs' to view training logs.")
    print("Your best model weights have been saved to 'best_dustscan_model.pth'.")
    return model

if __name__ == "__main__":
    DOWNLOADS_PATH = os.path.expanduser("~/Downloads/DustSCAN_2022.nc")
    
    if os.path.exists(DOWNLOADS_PATH):
        print(f"Found dataset at {DOWNLOADS_PATH}. Starting highly optimized pipeline...")
        # Train for 30 epochs as recommended
        trained_model = train_model(DOWNLOADS_PATH, epochs=30, batch_size=8, accumulation_steps=4, lr=3e-4)
    else:
        print(f"Dataset not found at {DOWNLOADS_PATH}. Please verify the exact file name and path.")
