"""
Dataset module for DustSCAN project.
Handles loading and preprocessing of NetCDF4 files for training.
"""
import random
import torch
import numpy as np
import netCDF4 as nc
from torch.utils.data import Dataset
from .utils import IMAGENET_MEAN, IMAGENET_STD, GLOBAL_STATS

class DustSCANDataset(Dataset):
    """
    Custom PyTorch Dataset for DustSCAN NetCDF files.
    """
    def __init__(self, nc_file_paths, time_indices, mode='train'):
        """
        Initialize the dataset.
        """
        self.nc_file_paths = nc_file_paths
        self.mode = mode
        self.time_indices = time_indices
        self.total_images = len(self.time_indices)
        
        self.file_lengths = []
        for path in self.nc_file_paths:
            with nc.Dataset(path, 'r') as f:
                self.file_lengths.append(f.dimensions['time'].size)
                
        self.file_offsets = np.cumsum([0] + self.file_lengths[:-1])
        
        self.handles = {}
            
    def __len__(self):
        """
        Get the total number of images in the dataset.
        """
        return self.total_images
        
    @staticmethod
    def _normalize_zscore(data, mean, std):
        """
        Apply z-score normalization to the data.
        """
        return (data - mean) / (std + 1e-8)
    
    @staticmethod
    def _normalize_imagenet(rgb):
        """
        Normalize RGB data using ImageNet statistics.
        """
        return (rgb - IMAGENET_MEAN) / IMAGENET_STD

    def _extract_full_image(self, global_time_idx):
        """
        Extract and preprocess a full image for a given global time index.
        """
        file_idx = 0
        for i, offset in enumerate(self.file_offsets):
            if global_time_idx >= offset:
                file_idx = i
                
        local_time_idx = global_time_idx - self.file_offsets[file_idx]
        file_path = self.nc_file_paths[file_idx]
        
        if file_path not in self.handles:
            self.handles[file_path] = nc.Dataset(file_path, 'r')
            
        ds = self.handles[file_path]
        
        dust_rgb = np.array(ds.variables['dust_rgb'][local_time_idx])
        if hasattr(dust_rgb, 'filled'):
            dust_rgb = dust_rgb.filled(np.nan)
            
        if dust_rgb.ndim == 3 and dust_rgb.shape[-1] == 3:
            dust_rgb = dust_rgb.transpose(2, 0, 1)
            
        sun_zenith = np.array(ds.variables['sun_zenith'][local_time_idx])
        if hasattr(sun_zenith, 'filled'):
            sun_zenith = sun_zenith.filled(np.nan)
            
        pdi = np.array(ds.variables['pdi'][local_time_idx])
        if hasattr(pdi, 'filled'):
            pdi = pdi.filled(np.nan)
        
        dust_rgb = self._normalize_imagenet(dust_rgb.astype(np.float32))
        sun_zenith = self._normalize_zscore(sun_zenith, GLOBAL_STATS['sun_zenith_mean'], GLOBAL_STATS['sun_zenith_std'])
        pdi = self._normalize_zscore(pdi, GLOBAL_STATS['pdi_mean'], GLOBAL_STATS['pdi_std'])
        
        X = np.concatenate([
            dust_rgb,
            np.expand_dims(sun_zenith, axis=0),
            np.expand_dims(pdi, axis=0)
        ], axis=0).astype(np.float32)
        
        plume_id = np.array(ds.variables['plume_id'][local_time_idx])
        if hasattr(plume_id, 'filled'):
            plume_id = plume_id.filled(0)
            
        Y = (plume_id > 0).astype(np.float32)
        Y = np.expand_dims(Y, axis=0)
        
        return X, Y

    def __getitem__(self, idx):
        """
        Get a single sample and its ground truth mask from the dataset.
        """
        global_time_idx = self.time_indices[idx]
        X, Y = self._extract_full_image(global_time_idx)
        
        if self.mode == 'train':
            if random.random() > 0.5:
                X = np.flip(X, axis=2).copy()
                Y = np.flip(Y, axis=2).copy()
            if random.random() > 0.5:
                X = np.flip(X, axis=1).copy()
                Y = np.flip(Y, axis=1).copy()
            
        X_tensor = torch.nan_to_num(torch.from_numpy(X))
        Y_tensor = torch.nan_to_num(torch.from_numpy(Y))
        
        return X_tensor, Y_tensor
