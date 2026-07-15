import os
import torch
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import build_advanced_unet_model
from src.dataset import DustSCANDataset
from src.utils import IMAGENET_MEAN, IMAGENET_STD

def evaluate_ablation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "outputs/models/best_dustscan_model.pth"
    nc_file_paths = ['C:/Users/sai siddharth/Downloads/DustSCAN_2021.nc', 'C:/Users/sai siddharth/Downloads/DustSCAN_2022.nc']
    output_dir = "outputs/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading datasets...")
    ds = xr.open_mfdataset(nc_file_paths, engine='netcdf4', combine='nested', concat_dim='time', parallel=False)
    times = ds['time'].values
    times_pd = pd.to_datetime(times)
    
    target_months = [1, 2, 9, 10, 11, 12]
    valid_month_indices = [i for i, t in enumerate(times_pd) if t.month in target_months]
    
    ds_filtered = ds.isel(time=valid_month_indices)
    plume_sums = (ds_filtered['plume_id'] > 0).sum(dim=['lat', 'lon']).compute().values
    ds.close()
    
    active_indices_in_filtered = np.where(plume_sums > 100)[0]
    
    np.random.seed(42)
    if len(active_indices_in_filtered) > 20:
        selected_filtered_indices = np.random.choice(active_indices_in_filtered, size=20, replace=False)
    else:
        selected_filtered_indices = active_indices_in_filtered
        
    val_indices = [valid_month_indices[i] for i in selected_filtered_indices]
    dataset = DustSCANDataset(nc_file_paths, time_indices=val_indices, mode='val')
    
    model = build_advanced_unet_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    scenarios = {
        'Baseline': lambda x: x,
        'No RGB': lambda x: ablate(x, [0, 1, 2]),
        'No Sun Zenith Angle': lambda x: ablate(x, [3]),
        'No PDI': lambda x: ablate(x, [4])
    }
    
    results = {k: [] for k in scenarios.keys()}
    
    def ablate(x, channels):
        x_new = x.clone()
        for c in channels:
            x_new[:, c, :, :] = 0.0
        return x_new

    print("Evaluating feature importance by ablation...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            X, Y = dataset[i]
            X_tensor = X.unsqueeze(0).to(device)
            gt_mask = Y[0].numpy()
            
            for name, transform in scenarios.items():
                X_mod = transform(X_tensor)
                with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available()):
                    output = model(X_mod)
                pred_mask = (torch.sigmoid(output[0, 0]) > 0.5).cpu().numpy()
                
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                if union == 0:
                    iou = 1.0
                else:
                    iou = intersection / union
                results[name].append(iou)
                
    mean_ious = {k: np.mean(v) for k, v in results.items()}
    
    baseline_iou = mean_ious['Baseline']
    drops = {k: baseline_iou - v for k, v in mean_ious.items() if k != 'Baseline'}
    
    print("\nResults:")
    print(f"Baseline IoU: {baseline_iou:.4f}")
    for k, v in drops.items():
        print(f"Drop without {k.replace('No ', '')}: {v:.4f}")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    features = list(drops.keys())
    importance = list(drops.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importance)
    features = [features[i].replace('No ', '') for i in sorted_idx]
    importance = [importance[i] for i in sorted_idx]
    
    bars = ax.barh(features, importance, color=['skyblue', 'lightgreen', 'salmon'])
    ax.set_xlabel('Drop in Mean IoU (Higher = More Important)')
    ax.set_title('Feature Importance via Ablation Study')
    
    for i, v in enumerate(importance):
        ax.text(max(0, v) + 0.001, i, f'{v:.4f}', va='center')
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved feature importance plot to {save_path}")

if __name__ == "__main__":
    evaluate_ablation()
