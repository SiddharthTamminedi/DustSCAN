import os
import sys

# Ensure the root directory is in the PYTHONPATH so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model

if __name__ == "__main__":
    DOWNLOADS_PATH = os.path.expanduser("~/Downloads/DustSCAN_2022.nc")
    
    if os.path.exists(DOWNLOADS_PATH):
        print(f"Found dataset at {DOWNLOADS_PATH}. Starting highly optimized pipeline...")
        # Train for 30 epochs with batch size 32 
        trained_model = train_model(DOWNLOADS_PATH, epochs=30, batch_size=32, accumulation_steps=1, lr=3e-4)
    else:
        print(f"Dataset not found at {DOWNLOADS_PATH}. Please verify the exact file name and path.")
