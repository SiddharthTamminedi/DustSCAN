import random
import torch
import xarray as xr
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .dataset import DustSCANDataset
from .models import build_advanced_unet_model, FocalDiceLoss
from .utils import save_full_image_prediction, IMAGENET_MEAN, IMAGENET_STD

def worker_init_fn(worker_id):
    import numpy as np
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
    
    writer = SummaryWriter(log_dir="outputs/runs/dustscan_experiment")

    try:
        with xr.open_dataset(nc_file_path, engine='netcdf4') as ds:
            time_dim = 'time'
            total_time_steps = len(ds[time_dim])
            times = ds['time'].values
    except Exception as e:
        print(f"Failed to open dataset: {e}")
        return None

    import pandas as pd
    times_pd = pd.to_datetime(times)
    # Filter to only include months with high dust percentages (Mar, Apr, May, Jun, Jul, Aug)
    target_months = [3, 4, 5, 6, 7, 8]
    all_indices = [i for i, t in enumerate(times_pd) if t.month in target_months]
    
    print(f"Reduced dataset from {total_time_steps} to {len(all_indices)} timesteps for faster training.")

    # Shuffle time indices to prevent seasonal bias in the splits
    random.seed(42)  # For reproducibility
    random.shuffle(all_indices)

    train_split_idx = int(0.8 * len(all_indices))
    train_indices = all_indices[:train_split_idx]
    val_indices = all_indices[train_split_idx:]

    train_dataset = DustSCANDataset(nc_file_path, patch_size=128, time_indices=train_indices, mode='train', samples_per_time_step=5)
    val_dataset = DustSCANDataset(nc_file_path, patch_size=128, time_indices=val_indices, mode='val')

    
    num_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)

    model = build_advanced_unet_model().to(device)
    criterion = FocalDiceLoss(alpha=0.85, gamma=2.0, pos_weight=10.0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
                
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if scale == scaler.get_scale():
                    scheduler.step(epoch + batch_idx / len(train_loader))
                
            train_loss += loss.item() * accumulation_steps
            
        train_loss /= max(1, len(train_loader))
        writer.add_scalar('Loss/Train', train_loss, epoch)
        
        # Validation
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
                    
        val_loss /= max(1, len(val_loader))
        
        save_full_image_prediction(model, nc_file_path, val_indices, epoch, device, save_dir="outputs/visualizations/val")
        
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
            torch.save(model.state_dict(), "outputs/models/best_dustscan_model.pth")
            print(f"--> Saved New Best Model (F1: {best_f1:.4f})")
              
    writer.close()
    print("Training complete. Run 'tensorboard --logdir outputs/runs' to view training logs.")
    print("Your best model weights have been saved to 'outputs/models/best_dustscan_model.pth'.")
    return model
