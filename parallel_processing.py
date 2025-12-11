#!/usr/bin/env python3
"""
Parallel processing utilities for batch image processing.
"""

import os
from typing import List, Tuple, Optional, Callable
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np

from preprocessing import load_image, load_mask
from pso_segmentation import pso_threshold
from metrics import compute_all_metrics
from utils import get_mask_path


def process_single_image(
    img_path: str,
    mask_dir: str,
    process_tumor: bool = True
) -> Optional[Tuple[str, Tuple[float, float, float, float], Optional[float], np.ndarray, np.ndarray, np.ndarray]]:
    """
    Process a single image. Designed for parallel execution.
    
    Parameters
    ----------
    img_path : str
        Path to image file
    mask_dir : str
        Directory containing masks
    process_tumor : bool
        Whether to process tumor slices with PSO
        
    Returns
    -------
    tuple or None
        (base_name, metrics, threshold, img, mask, pred) or None if error
    """
    try:
        mask_path = get_mask_path(img_path, mask_dir)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        if not os.path.exists(mask_path):
            return None
        
        img = load_image(img_path)
        mask = load_mask(mask_path, img.shape)
        
        if mask.sum() == 0:
            # Healthy slice
            pred = np.zeros_like(mask, dtype=np.uint8)
            threshold = None
        else:
            # Tumor slice
            if process_tumor:
                threshold = pso_threshold(img, mask)
            else:
                threshold = 0.5  # Default threshold
            pred = (img > threshold).astype(np.uint8) if threshold else np.zeros_like(mask)
        
        metrics = compute_all_metrics(mask, pred)
        return (base_name, metrics, threshold, img, mask, pred)
        
    except Exception as e:
        return None


def process_images_parallel(
    image_paths: List[str],
    mask_dir: str,
    n_workers: Optional[int] = None,
    process_tumor: bool = True
) -> List[Tuple[str, Tuple[float, float, float, float], Optional[float], np.ndarray, np.ndarray, np.ndarray]]:
    """
    Process images in parallel using multiprocessing.
    
    Parameters
    ----------
    image_paths : list
        List of image file paths
    mask_dir : str
        Directory containing masks
    n_workers : int, optional
        Number of worker processes. Defaults to CPU count - 1
    process_tumor : bool
        Whether to process tumor slices with PSO
        
    Returns
    -------
    list
        List of results from process_single_image
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    # Create partial function for pool.map
    process_func = partial(process_single_image, mask_dir=mask_dir, process_tumor=process_tumor)
    
    # Process in parallel
    with Pool(n_workers) as pool:
        results = pool.map(process_func, image_paths)
    
    # Filter out None results (errors)
    return [r for r in results if r is not None]
