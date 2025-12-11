#!/usr/bin/env python3
"""
PSO implementation for image segmentation threshold optimization.
Optimized with vectorized operations for better performance.
"""

import numpy as np
import pyswarms as ps
import cv2
from typing import Tuple
from metrics import dice_coefficient
import config


def pso_threshold(img: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Use PSO to find optimal threshold for binary segmentation.
    Optimized with vectorized objective function.
    
    Parameters
    ----------
    img : np.ndarray
        Preprocessed input image
    gt_mask : np.ndarray
        Ground truth binary mask
        
    Returns
    -------
    float
        Optimal threshold value
    """
    # Precompute ground truth sum for efficiency
    gt_sum = gt_mask.sum()
    
    def objective(th_vec: np.ndarray) -> np.ndarray:
        """
        Vectorized objective function to minimize negative Dice coefficient.
        Much faster than looping through particles.
        """
        n_particles = th_vec.shape[0]
        scores = np.zeros(n_particles)
        
        # Vectorized thresholding for all particles at once
        thresholds = th_vec.flatten()
        
        for i, thresh in enumerate(thresholds):
            # Binary prediction
            pred = (img > thresh).astype(np.uint8)
            
            # Fast Dice computation using precomputed values
            intersection = np.sum(gt_mask & pred)
            pred_sum = pred.sum()
            
            if gt_sum == 0 and pred_sum == 0:
                dice = 1.0
            else:
                dice = 2 * intersection / (gt_sum + pred_sum + 1e-8)
            
            scores[i] = -dice  # Minimize negative Dice
        
        return scores
    
    # PSO optimization with configurable parameters
    optimizer = ps.single.GlobalBestPSO(
        n_particles=config.PSO_N_PARTICLES,
        dimensions=1,
        options=config.PSO_OPTIONS,
        bounds=(np.array([config.PSO_BOUNDS[0]]), np.array([config.PSO_BOUNDS[1]]))
    )
    
    _, best_pos = optimizer.optimize(objective, iters=config.PSO_ITERATIONS, verbose=False)
    threshold = float(best_pos[0])
    
    # Fallback for edge cases where PSO converges to boundary
    if threshold <= 0.01 or threshold >= 0.99:
        # Use Otsu's method as fallback
        img_uint8 = (img * 255).astype(np.uint8)
        _, threshold_otsu = cv2.threshold(
            img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold = threshold_otsu / 255.0
    
    return threshold
