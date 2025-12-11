#!/usr/bin/env python3
"""
Evaluation metrics for segmentation tasks.
Optimized for performance with vectorized operations.
"""

import numpy as np
from typing import Tuple

EPS = 1e-8


def dice_coefficient(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate Dice coefficient (F1 score for binary segmentation).
    Optimized version.
    
    Parameters
    ----------
    gt : np.ndarray
        Ground truth binary mask
    pred : np.ndarray
        Predicted binary mask
        
    Returns
    -------
    float
        Dice coefficient in [0, 1]
    """
    gt_sum = gt.sum()
    pred_sum = pred.sum()
    
    if gt_sum == 0 and pred_sum == 0:
        return 1.0
    
    # Use bitwise AND for faster intersection computation
    intersection = np.sum(gt & pred)
    return 2 * intersection / (gt_sum + pred_sum + EPS)


def iou_coefficient(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) coefficient.
    Optimized version.
    
    Parameters
    ----------
    gt : np.ndarray
        Ground truth binary mask
    pred : np.ndarray
        Predicted binary mask
        
    Returns
    -------
    float
        IoU coefficient in [0, 1]
    """
    gt_sum = gt.sum()
    pred_sum = pred.sum()
    
    if gt_sum == 0 and pred_sum == 0:
        return 1.0
    
    intersection = np.sum(gt & pred)
    union = np.sum(gt | pred)
    return intersection / (union + EPS)


def precision_coefficient(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate precision (positive predictive value).
    Optimized version.
    
    Parameters
    ----------
    gt : np.ndarray
        Ground truth binary mask
    pred : np.ndarray
        Predicted binary mask
        
    Returns
    -------
    float
        Precision in [0, 1]
    """
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    
    if pred_sum == 0:
        return 1.0 if gt_sum == 0 else 0.0
    
    intersection = np.sum(gt & pred)
    return intersection / (pred_sum + EPS)


def recall_coefficient(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate recall (sensitivity, true positive rate).
    Optimized version.
    
    Parameters
    ----------
    gt : np.ndarray
        Ground truth binary mask
    pred : np.ndarray
        Predicted binary mask
        
    Returns
    -------
    float
        Recall in [0, 1]
    """
    gt_sum = gt.sum()
    pred_sum = pred.sum()
    
    if gt_sum == 0:
        return 1.0 if pred_sum == 0 else 0.0
    
    intersection = np.sum(gt & pred)
    return intersection / (gt_sum + EPS)


def compute_all_metrics(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute all evaluation metrics at once.
    Optimized to compute intersection only once.
    
    Parameters
    ----------
    gt : np.ndarray
        Ground truth binary mask
    pred : np.ndarray
        Predicted binary mask
        
    Returns
    -------
    tuple
        (dice, iou, precision, recall)
    """
    # Compute intersection once and reuse
    intersection = np.sum(gt & pred)
    gt_sum = gt.sum()
    pred_sum = pred.sum()
    union = np.sum(gt | pred)
    
    # Handle edge cases
    if gt_sum == 0 and pred_sum == 0:
        return (1.0, 1.0, 1.0, 1.0)
    
    # Compute all metrics efficiently
    dice = 2 * intersection / (gt_sum + pred_sum + EPS) if (gt_sum + pred_sum) > 0 else 0.0
    iou = intersection / (union + EPS) if union > 0 else 0.0
    precision = intersection / (pred_sum + EPS) if pred_sum > 0 else (1.0 if gt_sum == 0 else 0.0)
    recall = intersection / (gt_sum + EPS) if gt_sum > 0 else (1.0 if pred_sum == 0 else 0.0)
    
    return (dice, iou, precision, recall)
