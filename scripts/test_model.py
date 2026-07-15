"""
Script to evaluate the DustSCAN model and generate prediction images.
"""
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


def generate_predictions():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "outputs/models/best_dustscan_model.pth"
    nc_file_paths = [
        'C:/Users/sai siddharth/Downloads/DustSCAN_2021.nc',
        'C:/Users/sai siddharth/Downloads/DustSCAN_2022.nc'
    ]
    output_dir = "outputs/predictions"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading datasets to find unseen off-season indices...")
    ds = xr.open_mfdataset(
        nc_file_paths,
        engine='netcdf4',
        combine='nested',
        concat_dim='time',
        parallel=False
    )
    times = ds['time'].values

    times_pd = pd.to_datetime(times)

    # For seeing generalization outside of March to August
    target_months = [1, 2, 9, 10, 11, 12]
    valid_month_indices = [
        i for i, t in enumerate(times_pd)
        if t.month in target_months
    ]

    ds_filtered = ds.isel(time=valid_month_indices)
    plume_sums = (ds_filtered['plume_id'] > 0).sum(dim=['lat', 'lon']).compute().values
    ds.close()

    active_indices_in_filtered = np.where(plume_sums > 100)[0]

    np.random.seed(42)

    # Randomly select 20 indices from the active ones
    if len(active_indices_in_filtered) > 20:
        selected_filtered_indices = np.random.choice(
            active_indices_in_filtered,
            size=20,
            replace=False
        )
    else:
        selected_filtered_indices = active_indices_in_filtered

    val_indices = [valid_month_indices[i] for i in selected_filtered_indices]

    print(f"Total time steps in dataset: {len(times)}")
    print(f"Randomly selected {len(val_indices)} off-season dust storms for evaluation.")

    dataset = DustSCANDataset(
        nc_file_paths,
        time_indices=val_indices,
        mode='val'
    )

    model = build_advanced_unet_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Generating predictions in {output_dir}...")

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            X, Y = dataset[i]

            X_tensor = X.unsqueeze(0).to(device)

            with torch.autocast(
                'cuda' if torch.cuda.is_available() else 'cpu',
                enabled=torch.cuda.is_available()
            ):
                output = model(X_tensor)

            pred_mask = (torch.sigmoid(output[0, 0]) > 0.5).cpu().numpy()
            gt_mask = Y[0].numpy()

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

            save_path = os.path.join(
                output_dir,
                f"prediction {i+1}.png"
            )
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

    print(f"Done! Evaluated {len(val_indices)} random samples.")
    print(f"Saved all {len(val_indices)} representative samples to {output_dir}")


if __name__ == "__main__":
    generate_predictions()