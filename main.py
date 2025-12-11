#!/usr/bin/env python3
"""
Main pipeline for PSO-based brain tumor segmentation.
"""

import os
import glob
import random
import numpy as np
from tqdm import tqdm
from preprocessing import load_image, load_mask
from pso_segmentation import pso_threshold
from metrics import compute_all_metrics
from visualization import (
    save_triplet_comparison,
    plot_dice_histogram,
    plot_metrics_boxplot,
    plot_threshold_vs_dice
)


# Configuration
IMAGE_DIR = "/Users/tanishq/Desktop/PSO/pso_ready_cases/images"
MASK_DIR = "/Users/tanishq/Desktop/PSO/pso_ready_cases/masks"
PLOT_DIR = "./visualizations"
TRIPLET_DIR = "./qualitative_examples"

# Create output directories
for d in [PLOT_DIR, TRIPLET_DIR]:
    os.makedirs(d, exist_ok=True)


def process_image(image_path, mask_path):
    """Process a single image and return metrics and threshold."""
    img = load_image(image_path)
    mask = load_mask(mask_path, img.shape)
    
    if mask.sum() == 0:
        # Healthy slice - no tumor
        pred = np.zeros_like(mask, dtype=np.uint8)
        threshold = None
    else:
        # Tumor slice - use PSO to find optimal threshold
        threshold = pso_threshold(img, mask)
        pred = (img > threshold).astype(np.uint8)
    
    metrics = compute_all_metrics(mask, pred)
    return metrics, threshold, img, mask, pred


def main(k_triplets=6, triplet_prob=0.03):
    """
    Main processing pipeline.
    
    Parameters
    ----------
    k_triplets : int
        Maximum number of qualitative examples to save
    triplet_prob : float
        Probability per tumor slice to be saved
    """
    random.seed(0)
    
    # Find all images
    image_extensions = ["jpg", "jpeg", "png", "JPG", "PNG"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, f"*.{ext}")))
    
    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images")
    
    # Process images
    tumour_metrics = []
    healthy_metrics = []
    tumour_dice = []
    tumour_thresholds = []
    saved_triplets = 0
    
    for img_path in tqdm(image_paths, desc="Processing"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(MASK_DIR, f"{base_name}_mask.jpg")
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {base_name}")
            continue
        
        try:
            metrics, threshold, img, mask, pred = process_image(img_path, mask_path)
            
            if mask.sum() > 0:  # Tumor slice
                tumour_metrics.append(metrics)
                tumour_dice.append(metrics[0])
                tumour_thresholds.append(threshold)
                
                # Save qualitative examples
                if saved_triplets < k_triplets and random.random() < triplet_prob:
                    triplet_path = os.path.join(TRIPLET_DIR, f"{base_name}_triplet.png")
                    save_triplet_comparison(img, mask, pred, triplet_path)
                    saved_triplets += 1
            else:  # Healthy slice
                healthy_metrics.append(metrics)
                
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            continue
    
    # Print summary
    if tumour_metrics:
        metrics_array = np.array(tumour_metrics)
        print("\n=== Tumor Cases ===")
        print(f"Dice:      {metrics_array[:, 0].mean():.4f}")
        print(f"IoU:       {metrics_array[:, 1].mean():.4f}")
        print(f"Precision: {metrics_array[:, 2].mean():.4f}")
        print(f"Recall:    {metrics_array[:, 3].mean():.4f}")
        
        # Generate visualizations
        plot_dice_histogram(tumour_dice, os.path.join(PLOT_DIR, "dice_hist_tumour.png"))
        plot_metrics_boxplot(metrics_array, os.path.join(PLOT_DIR, "metric_spread_tumour.png"))
        plot_threshold_vs_dice(tumour_thresholds, tumour_dice, 
                              os.path.join(PLOT_DIR, "tau_vs_dice.png"))
        print(f"\nVisualizations saved to {PLOT_DIR}")
    
    if healthy_metrics:
        metrics_array = np.array(healthy_metrics)
        print("\n=== Healthy Cases ===")
        print(f"Dice:      {metrics_array[:, 0].mean():.4f}")
        print(f"IoU:       {metrics_array[:, 1].mean():.4f}")
        print(f"Precision: {metrics_array[:, 2].mean():.4f}")
        print(f"Recall:    {metrics_array[:, 3].mean():.4f}")


if __name__ == "__main__":
    main()
