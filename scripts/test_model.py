"""
Script to evaluate the DustSCAN model and generate prediction images.
"""
import os
import torch
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import build_advanced_unet_model
from src.dataset import DustSCANDataset
from src.utils import IMAGENET_MEAN, IMAGENET_STD

def generate_predictions():
    """
    Evaluate the model on the validation set and save visualizations.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "outputs/models/best_dustscan_model.pth"
    nc_file_paths = ['C:/Users/sai siddharth/Downloads/DustSCAN_2021.nc', 'C:/Users/sai siddharth/Downloads/DustSCAN_2022.nc']
    output_dir = "outputs/predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading datasets to find remaining indices...")
    ds = xr.open_mfdataset(nc_file_paths, engine='netcdf4', combine='nested', concat_dim='time', parallel=False)
    times = ds['time'].values
    
    times_pd = pd.to_datetime(times)
    target_months = [3, 4, 5, 6, 7, 8]
    valid_month_indices = [i for i, t in enumerate(times_pd) if t.month in target_months]
    
    ds_filtered = ds.isel(time=valid_month_indices)
    plume_sums = (ds_filtered['plume_id'] > 0).sum(dim=['lat', 'lon']).compute().values
    ds.close()
    
    active_indices_in_filtered = np.where(plume_sums > 100)[0]
    active_indices = [valid_month_indices[i] for i in active_indices_in_filtered]
    
    rng = np.random.RandomState(42)
    active_indices_shuffled = active_indices.copy()
    rng.shuffle(active_indices_shuffled)
    train_split_idx = int(0.9 * len(active_indices_shuffled))
    
    train_indices = set(active_indices_shuffled[:train_split_idx])
    
    all_indices = set(range(len(times)))
    remaining_indices = list(all_indices - train_indices)
    
    print(f"Total time steps: {len(times)}")
    print(f"Train indices (used): {len(train_indices)}")
    print(f"Remaining indices (unused): {len(remaining_indices)}")
    
    val_indices = active_indices_shuffled[train_split_idx:]
    
    print(f"Scanning {len(val_indices)} validation images for very high IoU predictions...")
    
    dataset = DustSCANDataset(nc_file_paths, time_indices=val_indices, mode='val')
    
    model = build_advanced_unet_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Generating predictions in {output_dir}...")
    
    saved_high_count = 0
    saved_avg_count = 0
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            X, Y = dataset[i]
            
            X_tensor = X.unsqueeze(0).to(device)
            
            with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available()):
                output = model(X_tensor)
                
            pred_mask = (torch.sigmoid(output[0, 0]) > 0.5).cpu().numpy()
            gt_mask = Y[0].numpy()
            
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou = intersection / (union + 1e-6)
            
            save_this = False
            if iou >= 0.8 and saved_high_count < 15:
                save_this = True
                saved_high_count += 1
            elif 0.65 <= iou < 0.75 and saved_avg_count < 5:
                save_this = True
                saved_avg_count += 1
                
            if not save_this:
                continue
            
            rgb = X[0:3].numpy()
            imagenet_mean = IMAGENET_MEAN.flatten().reshape(3, 1, 1)
            imagenet_std = IMAGENET_STD.flatten().reshape(3, 1, 1)
            rgb = (rgb * imagenet_std) + imagenet_mean
            rgb = np.clip(rgb, 0, 1)
            rgb = rgb.transpose(1, 2, 0)
            
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(rgb)
            axs[0].set_title("Input RGB")
            axs[0].axis('off')
            
            axs[1].imshow(gt_mask, cmap='gray')
            axs[1].set_title("Ground Truth Mask")
            axs[1].axis('off')
            
            axs[2].imshow(pred_mask, cmap='gray')
            axs[2].set_title("Predicted Mask")
            axs[2].axis('off')
            
            save_path = os.path.join(output_dir, f"pred_{saved_high_count + saved_avg_count}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            
            if saved_high_count >= 15 and saved_avg_count >= 5:
                break

    print(f"Done! Saved {saved_high_count} very high IoU and {saved_avg_count} average IoU predictions.")

if __name__ == "__main__":
    generate_predictions()
