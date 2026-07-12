"""
Generate Visualizations Script.
Extracts specific timesteps from DustSCAN dataset and plots them.
"""
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import shutil

DOWNLOADS_PATH = os.path.expanduser("~/Downloads/DustSCAN_2022.nc")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "visualizations", "dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for old_file in ["sample_visualization.png", "debug_patch.png", "big_dust_storm.png"]:
    if os.path.exists(old_file):
        shutil.move(old_file, os.path.join(OUTPUT_DIR, old_file))

print(f"Opening dataset from {DOWNLOADS_PATH}...")
with xr.open_dataset(DOWNLOADS_PATH, engine='netcdf4') as ds:
    print("Loading plume_id data to find the biggest dust storms...")
    plume_data = ds["plume_id"].values
    
    binary_plume = (plume_data > 0)
    
    dust_pixels_per_time = binary_plume.sum(axis=(1, 2))
    
    dusty_timesteps = np.where(dust_pixels_per_time > 1000)[0]
    
    np.random.seed(42)
    selected_indices = np.random.choice(dusty_timesteps, size=20, replace=False)
    
    print(f"Generating 20 random visualizations in '{OUTPUT_DIR}/'...")
    
    for i, t_idx in enumerate(selected_indices):
        dust_count = dust_pixels_per_time[t_idx]
        print(f"Generating image {i+1}/20 (Timestep {t_idx} with {dust_count} dust pixels)...")
        
        ds_slice = ds.isel(time=t_idx)
        plume_id = ds_slice["plume_id"].values
        dust_rgb = ds_slice["dust_rgb"].values
        
        plt.figure(figsize=(16, 8))
        
        plt.subplot(1, 2, 1)
        if dust_rgb.ndim == 3 and dust_rgb.shape[0] == 3:
            disp_rgb = dust_rgb.transpose(1, 2, 0)
        else:
            disp_rgb = dust_rgb
            
        disp_rgb = np.clip(disp_rgb, 0, 1)
        plt.imshow(disp_rgb)
        plt.title(f"SEVIRI dust_rgb (Time Index: {t_idx})")
        
        plt.subplot(1, 2, 2)
        plt.imshow(plume_id > 0, cmap='gray')
        plt.title(f"Binary Ground Truth Mask ({dust_count} px)")
        
        plt.tight_layout()
        filename = f"dust_storm_time_{t_idx}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
        plt.close()
        
print(f"Done! Check the '{OUTPUT_DIR}' folder.")
