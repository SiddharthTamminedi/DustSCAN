import random
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
from .utils import IMAGENET_MEAN, IMAGENET_STD, GLOBAL_STATS

class DustSCANDataset(Dataset):
    """
    Custom PyTorch Dataset for the DustSCAN 2022 NetCDF file.
    Extracts dust_rgb (3 channels), sun_zenith (1 channel), and pdi (1 channel).
    Uses all timesteps (day and night) and relies solely on plume_id as ground truth.
    """
    def __init__(self, nc_file_path, patch_size=128, time_indices=None, mode='train',
                 samples_per_time_step=10):
        self.nc_file_path = nc_file_path
        self.patch_size = patch_size
        self.mode = mode
        self.ds = None  # Lazy loading in __getitem__ to support multi-worker safely
        
        try:
            with xr.open_dataset(self.nc_file_path, engine='netcdf4') as ds_temp:
                self.time_dim = 'time'
                self.lat_dim = 'lat'
                self.lon_dim = 'lon'
                
                self.total_time_steps = len(ds_temp[self.time_dim])
                self.height = len(ds_temp[self.lat_dim])
                self.width = len(ds_temp[self.lon_dim])
                
                if time_indices is None:
                    time_indices = list(range(self.total_time_steps))
                
        except Exception as e:
            raise FileNotFoundError(f"Error opening NetCDF file {self.nc_file_path}: {e}")

        # Use all timesteps (day + night)
        self.time_indices = time_indices
        
        if self.mode == 'train':
            self.samples_per_time_step = samples_per_time_step
            self.total_patches = len(self.time_indices) * self.samples_per_time_step
            
            print("Precomputing dust patch coordinates. This takes a moment but saves massive RAM during multiprocessing...")
            self.dust_coords_by_time = {}
            with xr.open_dataset(self.nc_file_path, engine='netcdf4') as temp_ds:
                from tqdm import tqdm
                for t_idx in tqdm(self.time_indices, desc="Scanning dust masks"):
                    full_plume = temp_ds['plume_id'].isel(time=t_idx).values
                    self.dust_coords_by_time[t_idx] = np.argwhere(full_plume > 0)
        else:
            # Precompute unique patch coordinates for validation
            self.val_coords = []
            seen = set()
            for r in range(0, self.height, self.patch_size):
                for c in range(0, self.width, self.patch_size):
                    r_start = min(r, max(0, self.height - self.patch_size))
                    c_start = min(c, max(0, self.width - self.patch_size))
                    if (r_start, c_start) not in seen:
                        seen.add((r_start, c_start))
                        self.val_coords.append((r_start, c_start))
            self.patches_per_time = len(self.val_coords)
            self.total_patches = len(self.time_indices) * self.patches_per_time
            
    def __len__(self):
        return self.total_patches
        
    @staticmethod
    def _normalize_zscore(data, mean, std):
        return (data - mean) / (std + 1e-8)
    
    @staticmethod
    def _normalize_imagenet(rgb):
        """Normalize RGB channels with ImageNet mean/std for pretrained encoder."""
        return (rgb - IMAGENET_MEAN) / IMAGENET_STD

    def _extract_patch(self, time_idx, row_start, col_start):
        """Extract a single patch from the dataset and return numpy arrays."""
        row_end = row_start + self.patch_size
        col_end = col_start + self.patch_size
        
        ds_slice = self.ds.isel({
            self.time_dim: time_idx,
            self.lat_dim: slice(row_start, row_end),
            self.lon_dim: slice(col_start, col_end)
        })
        
        dust_rgb = ds_slice['dust_rgb'].values  # already [0, 1] float32
        if dust_rgb.ndim == 3 and dust_rgb.shape[-1] == 3:  # (H, W, C)
            dust_rgb = dust_rgb.transpose(2, 0, 1)
        
        sun_zenith = ds_slice['sun_zenith'].values
        pdi = ds_slice['pdi'].values
        
        # ImageNet normalization for pretrained encoder (dust_rgb is already 0-1)
        dust_rgb = self._normalize_imagenet(dust_rgb.astype(np.float32))
        # Z-score only for auxiliary channels
        sun_zenith = self._normalize_zscore(sun_zenith, GLOBAL_STATS['sun_zenith_mean'], GLOBAL_STATS['sun_zenith_std'])
        pdi = self._normalize_zscore(pdi, GLOBAL_STATS['pdi_mean'], GLOBAL_STATS['pdi_std'])
        
        X = np.concatenate([
            dust_rgb,
            np.expand_dims(sun_zenith, axis=0),
            np.expand_dims(pdi, axis=0)
        ], axis=0).astype(np.float32)
        
        plume_id = ds_slice['plume_id'].values
        Y = (plume_id > 0).astype(np.float32)
        Y = np.expand_dims(Y, axis=0)
        
        return X, Y, (Y.sum() > 200)  # requires at least 200 dust pixels for balanced sampling

    def __getitem__(self, idx):
        if self.ds is None:
            # Lazy open for multi-processing compatibility
            self.ds = xr.open_dataset(self.nc_file_path, engine='netcdf4', chunks={'time': 1})
            
        if self.mode == 'train':
            time_idx_rel = idx // self.samples_per_time_step
            time_idx = self.time_indices[time_idx_rel]
            
            max_row = self.height - self.patch_size
            max_col = self.width - self.patch_size
            
            # Balanced sampling: 50% dust-containing patches, 50% random
            want_dust = (random.random() < 0.5)
            row_start = random.randint(0, max_row) if max_row > 0 else 0
            col_start = random.randint(0, max_col) if max_col > 0 else 0
            
            if want_dust:
                dust_coords = self.dust_coords_by_time[time_idx]
                
                if len(dust_coords) > 0:
                    for _attempt in range(100):
                        # Pick a random dust pixel
                        r, c = random.choice(dust_coords)
                        # Pick a random patch that contains this pixel
                        rs = random.randint(max(0, r - self.patch_size + 1), min(max_row, r))
                        cs = random.randint(max(0, c - self.patch_size + 1), min(max_col, c))
                        
                        # Validate that it has enough dust without hitting the disk!
                        in_patch = (dust_coords[:, 0] >= rs) & (dust_coords[:, 0] < rs + self.patch_size) & \
                                   (dust_coords[:, 1] >= cs) & (dust_coords[:, 1] < cs + self.patch_size)
                        if np.count_nonzero(in_patch) > 200:
                            row_start, col_start = rs, cs
                            break
                            
            X, Y, _ = self._extract_patch(time_idx, row_start, col_start)
        else:
            time_idx_rel = idx // self.patches_per_time
            time_idx = self.time_indices[time_idx_rel]
            
            patch_idx_spatial = idx % self.patches_per_time
            row_start, col_start = self.val_coords[patch_idx_spatial]
            X, Y, _ = self._extract_patch(time_idx, row_start, col_start)
        
        # Data Augmentation for Robust Training
        if self.mode == 'train':
            if random.random() > 0.5:
                X = np.flip(X, axis=2).copy()
                Y = np.flip(Y, axis=2).copy()
            if random.random() > 0.5:
                X = np.flip(X, axis=1).copy()
                Y = np.flip(Y, axis=1).copy()
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                X = np.rot90(X, k=k, axes=(1, 2)).copy()
                Y = np.rot90(Y, k=k, axes=(1, 2)).copy()
            
        X_tensor = torch.nan_to_num(torch.from_numpy(X))
        Y_tensor = torch.nan_to_num(torch.from_numpy(Y))
        
        return X_tensor, Y_tensor
