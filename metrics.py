#!/usr/bin/env python3
"""
Evaluation metrics for segmentation tasks.
"""

import numpy as np

EPS = 1e-8


def dice_coefficient(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate Dice coefficient (F1 score for binary segmentation).
    
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
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    
    intersection = np.sum(gt & pred)
    return 2 * intersection / (gt.sum() + pred.sum() + EPS)


def iou_coefficient(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) coefficient.
    
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
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    
    intersection = np.sum(gt & pred)
    union = np.sum(gt | pred)
    return intersection / (union + EPS)


def precision_coefficient(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate precision (positive predictive value).
    
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
    if pred.sum() == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    
    intersection = np.sum(gt & pred)
    return intersection / (pred.sum() + EPS)


def recall_coefficient(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate recall (sensitivity, true positive rate).
    
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
    if gt.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    
    intersection = np.sum(gt & pred)
    return intersection / (gt.sum() + EPS)


def compute_all_metrics(gt: np.ndarray, pred: np.ndarray) -> tuple:
    """
    Compute all evaluation metrics at once.
    
    Returns
    -------
    tuple
        (dice, iou, precision, recall)
    """
    return (
        dice_coefficient(gt, pred),
        iou_coefficient(gt, pred),
        precision_coefficient(gt, pred),
        recall_coefficient(gt, pred)
    )
