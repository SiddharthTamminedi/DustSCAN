"""
Utility module for DustSCAN project.
Contains common variables and visualization functions.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import matplotlib.pyplot as plt

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

GLOBAL_STATS = {
    'sun_zenith_mean': 89.064937,
    'sun_zenith_std': 43.909445,
    'pdi_mean': 0.5071591,
    'pdi_std': 0.1111518
}

def save_full_image_prediction(model, nc_file_path, val_indices, epoch, device, save_dir="outputs/visualizations/epoch_predictions"):
    """
    Loads a full image from the validation set, pads it for the UNet, predicts, and saves the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    if not val_indices:
        return
    time_idx = val_indices[0]
    
    with xr.open_dataset(nc_file_path, engine='netcdf4') as ds:
        ds_slice = ds.isel(time=time_idx)
        dust_rgb = ds_slice['dust_rgb'].values
        if dust_rgb.ndim == 3 and dust_rgb.shape[-1] == 3:
            dust_rgb = dust_rgb.transpose(2, 0, 1)
        sun_zenith = ds_slice['sun_zenith'].values
        pdi = ds_slice['pdi'].values
        plume_id = ds_slice['plume_id'].values
        
    dust_rgb_norm = (dust_rgb.astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD
    sun_zenith_norm = (sun_zenith - GLOBAL_STATS['sun_zenith_mean']) / (GLOBAL_STATS['sun_zenith_std'] + 1e-8)
    pdi_norm = (pdi - GLOBAL_STATS['pdi_mean']) / (GLOBAL_STATS['pdi_std'] + 1e-8)
    
    X = np.concatenate([
        dust_rgb_norm,
        np.expand_dims(sun_zenith_norm, axis=0),
        np.expand_dims(pdi_norm, axis=0)
    ], axis=0).astype(np.float32)
    
    Y = (plume_id > 0).astype(np.float32)
    X_tensor = torch.nan_to_num(torch.from_numpy(X)).unsqueeze(0).to(device)
    
    _, _, h, w = X_tensor.shape
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    
    if pad_h > 0 or pad_w > 0:
        X_tensor = F.pad(X_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
    model.eval()
    with torch.no_grad():
        with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available()):
            outputs = model(X_tensor)
            preds_bin = (torch.sigmoid(outputs) > 0.5).float()
            
    if pad_h > 0 or pad_w > 0:
        preds_bin = preds_bin[:, :, :h, :w]
        
    pred_np = preds_bin[0, 0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    display_rgb = np.clip(dust_rgb.transpose(1, 2, 0), 0, 1)
    
    axes[0].imshow(display_rgb)
    axes[0].set_title("dust_rgb (Full Image)")
    axes[0].axis("off")
    
    axes[1].imshow(Y, vmin=0, vmax=1, cmap="gray")
    axes[1].set_title("Ground Truth (Full Image)")
    axes[1].axis("off")
    
    axes[2].imshow(pred_np, vmin=0, vmax=1, cmap="gray")
    axes[2].set_title(f"Prediction (Epoch {epoch+1})")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1:02d}_full_image.png"), dpi=100)
    plt.close()
