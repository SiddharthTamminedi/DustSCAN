"""
Script to run the training pipeline for DustSCAN.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model

if __name__ == "__main__":
    DS_2021 = os.path.expanduser("~/Downloads/DustSCAN_2021.nc")
    DS_2022 = os.path.expanduser("~/Downloads/DustSCAN_2022.nc")
    
    nc_files = []
    for p in [DS_2021, DS_2022]:
        if os.path.exists(p):
            nc_files.append(p)
        else:
            print(f"Warning: Dataset not found at {p}. Please verify the exact file name and path.")

    if len(nc_files) > 0:
        print(f"Found datasets: {nc_files}. Starting highly optimized pipeline...")
        trained_model = train_model(nc_files, epochs=30, batch_size=32, accumulation_steps=1, lr=3e-4)
    else:
        print("No datasets found. Exiting.")
