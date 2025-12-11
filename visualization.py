#!/usr/bin/env python3
"""
Visualization utilities for segmentation results.
"""

import matplotlib
matplotlib.use("Agg")  # Headless-safe plotting
import matplotlib.pyplot as plt
import numpy as np
import os


def save_triplet_comparison(img, gt_mask, pred_mask, output_path):
    """
    Save side-by-side comparison of input, ground truth, and prediction.
    
    Parameters
    ----------
    img : np.ndarray
        Input image
    gt_mask : np.ndarray
        Ground truth mask
    pred_mask : np.ndarray
        Predicted mask
    output_path : str
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    
    for ax in axes:
        ax.axis("off")
    
    # Input image
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input")
    
    # Ground truth overlay
    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(gt_mask, cmap="Reds", alpha=0.4)
    axes[1].set_title("Ground Truth")
    
    # Prediction overlay
    axes[2].imshow(img, cmap="gray")
    axes[2].imshow(pred_mask, cmap="Blues", alpha=0.4)
    axes[2].set_title("PSO Prediction")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


def plot_dice_histogram(dice_scores, output_path):
    """
    Plot histogram of Dice coefficient distribution.
    
    Parameters
    ----------
    dice_scores : list or np.ndarray
        List of Dice scores
    output_path : str
        Path to save the figure
    """
    plt.figure(figsize=(6, 5))
    plt.hist(dice_scores, bins=12, edgecolor="black")
    plt.xlabel("Dice Coefficient")
    plt.ylabel("Number of Slices")
    plt.title("Dice Coefficient Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_metrics_boxplot(metrics_array, output_path):
    """
    Plot boxplot of all metrics.
    
    Parameters
    ----------
    metrics_array : np.ndarray
        Array of shape (N, 4) with columns [Dice, IoU, Precision, Recall]
    output_path : str
        Path to save the figure
    """
    plt.figure(figsize=(7, 5))
    plt.boxplot(
        metrics_array,
        tick_labels=["Dice", "IoU", "Precision", "Recall"],
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#add8e6", linewidth=1.2),
        medianprops=dict(color="darkorange", linewidth=2)
    )
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Metric Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_threshold_vs_dice(thresholds, dice_scores, output_path):
    """
    Plot scatter plot of threshold vs Dice coefficient.
    
    Parameters
    ----------
    thresholds : list or np.ndarray
        List of threshold values
    dice_scores : list or np.ndarray
        List of corresponding Dice scores
    output_path : str
        Path to save the figure
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(thresholds, dice_scores, s=28, alpha=0.8)
    plt.xlabel("Optimal Threshold τ*")
    plt.ylabel("Dice Coefficient")
    plt.title("Threshold vs Dice Coefficient")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
