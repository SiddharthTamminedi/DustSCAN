import os
import sys
import torch
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import build_advanced_unet_model
from src.utils import IMAGENET_MEAN, IMAGENET_STD, GLOBAL_STATS

# Constants
NLAT, NLON = 148, 357
MODEL_PATH = "outputs/models/best_dustscan_model.pth"

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="DustSCAN Prediction Interface",
    page_icon="🌪️",
    layout="wide"
)

# --- Helper Functions ---
@st.cache_resource
def load_model():
    """Load the trained PyTorch model (cached to avoid reloading)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_advanced_unet_model().to(device)
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure you've trained the model first.")
        st.stop()
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

def process_image(img, sun_zenith):
    """
    Process the uploaded PNG into a 5-channel tensor suitable for the model.
    """
    # 1. Resize to exact grid (Width=357, Height=148)
    img_resized = img.resize((NLON, NLAT), Image.Resampling.BILINEAR)
    
    # 2. Extract RGB as float [0, 1]
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    
    # Drop alpha channel if present
    if img_np.shape[-1] == 4:
        img_np = img_np[:, :, :3]
        
    dust_rgb_red = img_np[:, :, 0]
    dust_rgb_green = img_np[:, :, 1]
    dust_rgb_blue = img_np[:, :, 2]
    
    # 3. Calculate PDI: Distance from Magenta [1, 0, 1]
    # P_dist = sqrt((R-1)^2 + (G-0)^2 + (B-1)^2)
    p_dist = np.sqrt((dust_rgb_red - 1.0)**2 + (dust_rgb_green - 0.0)**2 + (dust_rgb_blue - 1.0)**2)
    max_dist = np.sqrt(3.0) # Maximum possible distance in RGB space
    
    pdi = 1.0 - (p_dist / max_dist)
    pdi = np.clip(pdi, 0, 1)
    
    # 4. Normalize
    # Convert RGB to channel-first (3, H, W)
    dust_rgb_chw = img_np.transpose(2, 0, 1)
    
    imagenet_mean = IMAGENET_MEAN.flatten().reshape(3, 1, 1)
    imagenet_std = IMAGENET_STD.flatten().reshape(3, 1, 1)
    
    rgb_normalized = (dust_rgb_chw - imagenet_mean) / imagenet_std
    
    # Create sun zenith array
    sz_array = np.full((NLAT, NLON), sun_zenith, dtype=np.float32)
    sz_normalized = (sz_array - GLOBAL_STATS['sun_zenith_mean']) / (GLOBAL_STATS['sun_zenith_std'] + 1e-8)
    
    pdi_normalized = (pdi - GLOBAL_STATS['pdi_mean']) / (GLOBAL_STATS['pdi_std'] + 1e-8)
    
    # 5. Stack into 5 channels (5, 148, 357)
    X = np.concatenate([
        rgb_normalized,
        np.expand_dims(sz_normalized, axis=0),
        np.expand_dims(pdi_normalized, axis=0),
    ], axis=0).astype(np.float32)
    
    return X, img_resized

# --- Main App ---
st.title("🌪️ DustSCAN Prediction Interface")
st.markdown("""
Upload a **Dust RGB** satellite image in PNG or JPG format to run inference. 
The model will automatically recalculate missing features (like the Pink Dust Index) and resize your image to the correct 0.25° grid spatial dimensions (148x357).
""")

# Load model early
model, device = load_model()

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    sun_zenith_val = st.slider(
        "Approximate Sun Zenith Angle (°)", 
        min_value=0.0, 
        max_value=180.0, 
        value=float(GLOBAL_STATS['sun_zenith_mean']), 
        step=1.0,
        help="0° is solar noon (sun directly overhead). >90° is night time."
    )
    
    threshold_val = st.slider(
        "Binarization Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5, 
        step=0.05
    )

# Main content
uploaded_file = st.file_uploader("Upload Dust RGB Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the image
    img = Image.open(uploaded_file).convert("RGB")
    
    st.subheader("Prediction Results")
    
    with st.spinner("Processing image and running inference..."):
        # 1. Preprocess
        X_numpy, img_resized = process_image(img, sun_zenith_val)
        
        # 2. Run Inference
        X_tensor = torch.from_numpy(X_numpy).unsqueeze(0).to(device)
        X_tensor = torch.nan_to_num(X_tensor)
        
        with torch.no_grad():
            with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available()):
                output = model(X_tensor)
                
        pred_prob = torch.sigmoid(output[0, 0]).cpu().numpy()
        pred_mask = (pred_prob > threshold_val).astype(np.uint8)
    
    # 3. Visualization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Original RGB (Resized)**")
        st.image(img_resized, use_container_width=True)
        
    with col2:
        st.markdown("**Prediction Probability**")
        fig, ax = plt.subplots(figsize=(4, 6))
        im = ax.imshow(pred_prob, cmap='hot', vmin=0, vmax=1)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        
    with col3:
        st.markdown(f"**Dust Mask (>{threshold_val})**")
        st.image(pred_mask * 255, use_container_width=True)
        st.markdown(f"*Total detected dust pixels: {pred_mask.sum()}*")

    st.success("Inference complete!")
