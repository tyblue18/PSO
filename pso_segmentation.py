#!/usr/bin/env python3
"""
PSO implementation for image segmentation threshold optimization.
"""

import numpy as np
import pyswarms as ps
from metrics import dice_coefficient


def pso_threshold(img, gt_mask):
    """
    Use PSO to find optimal threshold for binary segmentation.
    
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
    def objective(th_vec):
        """Minimize negative Dice coefficient."""
        scores = []
        for t in th_vec:
            pred = (img > t[0]).astype(np.uint8)
            dice = dice_coefficient(gt_mask, pred)
            scores.append(-dice)
        return np.array(scores)
    
    # PSO optimization
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
    optimizer = ps.single.GlobalBestPSO(
        n_particles=30,
        dimensions=1,
        options=options,
        bounds=(np.array([0.0]), np.array([1.0]))
    )
    
    _, best_pos = optimizer.optimize(objective, iters=40, verbose=False)
    threshold = float(best_pos[0])
    
    # Fallback for edge cases where PSO converges to boundary
    if threshold <= 0.01 or threshold >= 0.99:
        # Use Otsu's method as fallback
        import cv2
        img_uint8 = (img * 255).astype(np.uint8)
        _, threshold_otsu = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = threshold_otsu / 255.0
    
    return threshold
