#!/usr/bin/env python3
"""
Basic PSO implementation for image segmentation threshold optimization.
"""

import numpy as np
import pyswarms as ps


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
            dice = dice_coef(gt_mask, pred)
            scores.append(-dice)
        return np.array(scores)
    
    # Simple Dice coefficient calculation
    def dice_coef(g, p):
        inter = np.sum(g & p)
        return 2 * inter / (g.sum() + p.sum() + 1e-8)
    
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
    
    return threshold
