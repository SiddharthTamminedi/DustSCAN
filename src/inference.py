"""
Inference module for DustSCAN.
Provides functions for patch-wise inference on arbitrary-sized satellite images
using overlapping patches and 2D Gaussian blending to prevent edge artifacts.
"""
import numpy as np
import torch
from PIL import Image
from .utils import IMAGENET_MEAN, IMAGENET_STD, GLOBAL_STATS

def get_gaussian_window(h, w, sigma=0.5):
    """
    Generates a 2D Gaussian window of shape (h, w).
    This window is used for smooth blending of overlapping patches.
    """
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    
    g_x = np.exp(-0.5 * (x / sigma) ** 2)
    g_y = np.exp(-0.5 * (y / sigma) ** 2)
    
    window = np.outer(g_y, g_x)
    # Ensure minimum value to avoid division by zero
    window = np.maximum(window, 1e-4)
    return window

def preprocess_arbitrary_image(img, sun_zenith):
    """
    Preprocess an arbitrary size PIL image (RGB) and a Sun Zenith angle
    into a 5-channel tensor of shape (5, H, W) at its original size.
    
    Args:
        img: PIL.Image in RGB format
        sun_zenith: float, solar zenith angle in degrees
        
    Returns:
        X: np.ndarray of shape (5, H, W)
    """
    # 1. Convert PIL image to float32 numpy in [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0
    if img_np.shape[-1] == 4:
        img_np = img_np[:, :, :3]
        
    H, W, _ = img_np.shape
    
    # 2. Split RGB channels
    dust_rgb_red = img_np[:, :, 0]
    dust_rgb_green = img_np[:, :, 1]
    dust_rgb_blue = img_np[:, :, 2]
    
    # 3. Calculate PDI: Distance from Magenta [1, 0, 1]
    p_dist = np.sqrt((dust_rgb_red - 1.0)**2 + (dust_rgb_green - 0.0)**2 + (dust_rgb_blue - 1.0)**2)
    max_dist = np.sqrt(3.0)
    pdi = np.clip(1.0 - (p_dist / max_dist), 0.0, 1.0)
    
    # 4. Transpose RGB to channel-first (3, H, W)
    dust_rgb_chw = img_np.transpose(2, 0, 1)
    
    # Normalize RGB using ImageNet statistics
    # IMAGENET_MEAN and IMAGENET_STD shapes are (3, 1, 1), broadcasting works perfectly
    rgb_normalized = (dust_rgb_chw - IMAGENET_MEAN) / IMAGENET_STD
    
    # 5. Normalize Sun Zenith
    sz_array = np.full((H, W), sun_zenith, dtype=np.float32)
    sz_normalized = (sz_array - GLOBAL_STATS['sun_zenith_mean']) / (GLOBAL_STATS['sun_zenith_std'] + 1e-8)
    
    # 6. Normalize PDI
    pdi_normalized = (pdi - GLOBAL_STATS['pdi_mean']) / (GLOBAL_STATS['pdi_std'] + 1e-8)
    
    # 7. Stack into 5 channels (5, H, W)
    X = np.concatenate([
        rgb_normalized,
        np.expand_dims(sz_normalized, axis=0),
        np.expand_dims(pdi_normalized, axis=0),
    ], axis=0).astype(np.float32)
    
    return X

def patch_wise_inference(model, X_large, device, overlap=0.25):
    """
    Perform inference on an arbitrary-sized input tensor of shape (5, H, W) 
    by splitting it into overlapping patches of training size (148, 357),
    running model predictions, and blending them using a 2D Gaussian window.
    
    Args:
        model: PyTorch model (DustSCANUNet)
        X_large: np.ndarray of shape (5, H, W)
        device: torch.device
        overlap: float, ratio of patch overlap (default: 0.25, representing 25% overlap)
        
    Returns:
        blended_pred: np.ndarray of shape (H, W) - final prediction probability map
    """
    # Define expected model input patch size
    patch_h, patch_w = 148, 357
    
    _, H_orig, W_orig = X_large.shape
    
    # 1. Handle edge case where image is smaller than patch size in any dimension
    pad_h = max(0, patch_h - H_orig)
    pad_w = max(0, patch_w - W_orig)
    
    if pad_h > 0 or pad_w > 0:
        X_padded = np.pad(X_large, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    else:
        X_padded = X_large.copy()
        
    _, H_pad, W_pad = X_padded.shape
    
    # 2. Determine strides based on overlap
    # overlap = 0.25 -> stride = 75% of patch dimensions
    stride_y = int(np.round(patch_h * (1.0 - overlap)))
    stride_x = int(np.round(patch_w * (1.0 - overlap)))
    
    # Ensure stride is at least 1 pixel
    stride_y = max(1, stride_y)
    stride_x = max(1, stride_x)
    
    # 3. Create grid of starting coordinates
    y_starts = list(range(0, H_pad - patch_h + 1, stride_y))
    if y_starts[-1] + patch_h < H_pad:
        y_starts.append(H_pad - patch_h)
        
    x_starts = list(range(0, W_pad - patch_w + 1, stride_x))
    if x_starts[-1] + patch_w < W_pad:
        x_starts.append(W_pad - patch_w)
        
    # 4. Initialize accumulators
    accum_pred = np.zeros((H_pad, W_pad), dtype=np.float32)
    accum_weight = np.zeros((H_pad, W_pad), dtype=np.float32)
    
    # Generate 2D Gaussian window for patch boundary tapering
    window = get_gaussian_window(patch_h, patch_w, sigma=0.5)
    
    model.eval()
    with torch.no_grad():
        for y_start in y_starts:
            for x_start in x_starts:
                # Extract the 5-channel patch
                patch = X_padded[:, y_start:y_start+patch_h, x_start:x_start+patch_w]
                
                # Convert to tensor and prepare batch: shape (1, 5, 148, 357)
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
                patch_tensor = torch.nan_to_num(patch_tensor)
                
                # Forward pass (supports Automatic Mixed Precision if CUDA is available)
                with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available()):
                    logits = model(patch_tensor)
                    
                # Compute probability
                prob = torch.sigmoid(logits[0, 0]).cpu().numpy()
                
                # Accumulate prediction and Gaussian weight
                accum_pred[y_start:y_start+patch_h, x_start:x_start+patch_w] += prob * window
                accum_weight[y_start:y_start+patch_h, x_start:x_start+patch_w] += window
                
    # 5. Blend predictions by dividing by accumulated weights
    blended_pred = np.zeros_like(accum_pred)
    nonzero_mask = accum_weight > 0
    blended_pred[nonzero_mask] = accum_pred[nonzero_mask] / accum_weight[nonzero_mask]
    
    # 6. Crop back to original dimensions if padded
    if pad_h > 0 or pad_w > 0:
        blended_pred = blended_pred[:H_orig, :W_orig]
        
    return blended_pred
