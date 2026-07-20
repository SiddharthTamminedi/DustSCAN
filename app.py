import os
import sys
import io
import torch
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Try importing rasterio for GeoTIFF support
try:
    import rasterio
    import rasterio.io
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Add project root to sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import build_advanced_unet_model
from src.utils import IMAGENET_MEAN, IMAGENET_STD, GLOBAL_STATS
from src.inference import preprocess_arbitrary_image, patch_wise_inference

# Constants
NLAT, NLON = 148, 357
MODEL_PATH = "outputs/models/best_dustscan_model.pth"

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="DustSCAN Prediction Interface",
    page_icon="",
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

def normalize_to_zero_one(band_data):
    """Normalize input band data to [0, 1] range based on dtype and limits."""
    band_data = np.nan_to_num(band_data)
    val_min = np.min(band_data)
    val_max = np.max(band_data)
    
    if val_max == val_min:
        return np.zeros_like(band_data, dtype=np.float32)
        
    dtype_name = band_data.dtype.name
    if 'int' in dtype_name:
        if val_max > 255:
            return (band_data.astype(np.float32) - val_min) / (65535.0 - val_min + 1e-8)
        else:
            return (band_data.astype(np.float32) - val_min) / (255.0 - val_min + 1e-8)
    else:
        if val_max > 1.0 or val_min < 0.0:
            return (band_data - val_min) / (val_max - val_min + 1e-8)
        return band_data

def read_geotiff(uploaded_file):
    """
    Reads a GeoTIFF file from Streamlit's uploaded file using rasterio.
    Returns:
        img: PIL Image in RGB format
        spatial_info: dict containing crs, transform, width, height
    """
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    
    with rasterio.io.MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            num_bands = src.count
            
            if num_bands >= 3:
                r = src.read(1, masked=True)
                g = src.read(2, masked=True)
                b = src.read(3, masked=True)
            elif num_bands == 1:
                r = src.read(1, masked=True)
                g = r
                b = r
            else:
                r = src.read(1, masked=True)
                g = src.read(2, masked=True)
                b = np.zeros_like(r)
                
            r = np.ma.filled(r, 0)
            g = np.ma.filled(g, 0)
            b = np.ma.filled(b, 0)
            
            r_norm = normalize_to_zero_one(r)
            g_norm = normalize_to_zero_one(g)
            b_norm = normalize_to_zero_one(b)
            
            rgb_stack = np.stack([r_norm, g_norm, b_norm], axis=-1)
            rgb_stack = np.clip(rgb_stack * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(rgb_stack)
            
            spatial_info = {
                'crs': src.crs,
                'transform': src.transform,
                'width': src.width,
                'height': src.height
            }
            
            return img, spatial_info

# --- Main App ---
st.title(" DustSCAN Prediction Interface")
st.markdown("""
Upload a **Dust RGB** satellite image to run inference. 
Supports **PNG, JPG, JPEG**, and **GeoTIFF** formats.
The model will automatically recalculate missing features (like the Pink Dust Index) and resize your image to the correct 0.25° grid spatial dimensions (148x357).
""")

if not HAS_RASTERIO:
    st.warning("⚠️ `rasterio` is not installed or failed to load. GeoTIFF spatial metadata support is disabled.")

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
    
    st.write("---")
    inference_mode = st.radio(
        "Inference Mode",
        options=["Resize to Standard Grid (148x357)", "Patch-wise Blending (Original Resolution)"],
        index=1,
        help="Resize scales the image to training shape. Patch-wise processes the image at its original resolution in overlapping windows."
    )
    
    if inference_mode == "Patch-wise Blending (Original Resolution)":
        overlap_val = st.slider(
            "Patch Overlap Rate",
            min_value=0.0,
            max_value=0.9,
            value=0.25,
            step=0.05,
            help="Higher overlap yields smoother transitions but increases computation time."
        )
    else:
        overlap_val = 0.25

# Main content
file_types = ["png", "jpg", "jpeg"]
if HAS_RASTERIO:
    file_types.extend(["tif", "tiff"])
    
uploaded_file = st.file_uploader("Upload Dust RGB Image", type=file_types)

if uploaded_file is not None:
    # Check if the uploaded file is a TIFF/GeoTIFF
    is_tiff = uploaded_file.name.lower().endswith(('.tif', '.tiff'))
    
    spatial_info = None
    if is_tiff and HAS_RASTERIO:
        try:
            img, spatial_info = read_geotiff(uploaded_file)
        except Exception as e:
            st.warning(f"Failed to read spatial metadata from GeoTIFF: {e}. Falling back to standard image processing.")
            uploaded_file.seek(0)
            img = Image.open(uploaded_file).convert("RGB")
    else:
        img = Image.open(uploaded_file).convert("RGB")
        
    st.subheader("Prediction Results")
    
    if spatial_info:
        st.info(f" **GeoTIFF Spatial Metadata Detected:**\n"
                f"- **Resolution:** {spatial_info['width']} x {spatial_info['height']}\n"
                f"- **Coordinate Reference System (CRS):** `{spatial_info['crs']}`")
    
    with st.spinner("Processing image and running inference..."):
        if inference_mode == "Patch-wise Blending (Original Resolution)":
            # 1. Preprocess at original size
            X_numpy = preprocess_arbitrary_image(img, sun_zenith_val)
            
            # 2. Run Inference using patch-wise mechanism
            pred_prob = patch_wise_inference(model, X_numpy, device, overlap=overlap_val)
            pred_mask = (pred_prob > threshold_val).astype(np.uint8)
            img_display = img
        else:
            # 1. Preprocess by resizing to standard grid
            X_numpy, img_resized = process_image(img, sun_zenith_val)
            
            # 2. Run Inference
            X_tensor = torch.from_numpy(X_numpy).unsqueeze(0).to(device)
            X_tensor = torch.nan_to_num(X_tensor)
            
            with torch.no_grad():
                with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available()):
                    output = model(X_tensor)
                    
            pred_prob = torch.sigmoid(output[0, 0]).cpu().numpy()
            pred_mask = (pred_prob > threshold_val).astype(np.uint8)
            img_display = img_resized
            
    # 3. Visualization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if inference_mode == "Patch-wise Blending (Original Resolution)":
            st.markdown("**Original RGB (Original Size)**")
        else:
            st.markdown("**Original RGB (Resized)**")
        st.image(img_display, use_container_width=True)
        
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
