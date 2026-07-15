"""
Training module for DustSCAN.
Provides functions to train the model on the DustSCAN dataset.
"""
import os
import random
import torch
import numpy as np
import xarray as xr
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .dataset import DustSCANDataset
from .models import build_advanced_unet_model, FocalDiceBCELoss
from .utils import IMAGENET_MEAN, IMAGENET_STD

def worker_init_fn(worker_id):
    """
    Initialize the worker processes for DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_model(nc_file_paths, epochs=15, batch_size=8, accumulation_steps=4, lr=3e-4, device=None, resume_path=None):
    """
    Train the DustSCAN model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    print(f"Starting training on device: {device} (Max Power Mode)")
    
    writer = SummaryWriter(log_dir="outputs/runs/dustscan_experiment")

    try:
        print(f"Loading datasets: {nc_file_paths}")
        ds = xr.open_mfdataset(nc_file_paths, engine='netcdf4', combine='nested', concat_dim='time', parallel=False)
        times = ds['time'].values
    except Exception as e:
        print(f"Failed to open datasets: {e}")
        return None

    times_pd = pd.to_datetime(times)
    target_months = [3, 4, 5, 6, 7, 8]
    valid_month_indices = [i for i, t in enumerate(times_pd) if t.month in target_months]
    
    print(f"Reduced dataset from {len(times)} to {len(valid_month_indices)} valid month timesteps.")
    
    print("Computing plume pixel sums to find active events (this might take a moment)...")
    ds_filtered = ds.isel(time=valid_month_indices)
    plume_sums = (ds_filtered['plume_id'] > 0).sum(dim=['lat', 'lon']).compute().values
    
    ds.close()
    
    active_indices_in_filtered = np.where(plume_sums > 100)[0]
    
    active_indices = [valid_month_indices[i] for i in active_indices_in_filtered]
    
    print(f"Found {len(active_indices)} active timesteps (plume sum > 100).")

    rng = np.random.RandomState(42)
    active_indices_shuffled = active_indices.copy()
    rng.shuffle(active_indices_shuffled)

    train_split_idx = int(0.9 * len(active_indices_shuffled))
    train_indices = active_indices_shuffled[:train_split_idx]
    val_indices = active_indices_shuffled[train_split_idx:]

    print(f"Split: {len(train_indices)} train / {len(val_indices)} val")

    train_dataset = DustSCANDataset(nc_file_paths, time_indices=train_indices, mode='train')
    val_dataset = DustSCANDataset(nc_file_paths, time_indices=val_indices, mode='val')
    
    train_num_workers = 8
    val_num_workers = 4
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=train_num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=val_num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2, worker_init_fn=worker_init_fn)

    model = build_advanced_unet_model().to(device)
    if resume_path and os.path.exists(resume_path):
        model.load_state_dict(torch.load(resume_path, map_location=device))
        print(f"Resumed training from {resume_path}")
        
    criterion = FocalDiceBCELoss(alpha=0.90, gamma=2.0, pos_weight=15.0)
    
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
                    
                    Y_rgb = Y.repeat(1, 3, 1, 1)
                    preds_rgb = preds_bin.repeat(1, 3, 1, 1)
                    
                    panel = torch.cat([display_rgb, Y_rgb, preds_rgb], dim=3)
                    writer.add_images('Validation/RGB_GT_Pred', panel, epoch)
                    
        val_loss /= max(1, len(val_loader))
        
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
