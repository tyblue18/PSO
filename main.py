#!/usr/bin/env python3
"""
Main pipeline for PSO-based brain tumor segmentation.
"""

import os
import glob
import numpy as np
from preprocessing import load_image, load_mask
from pso_segmentation import pso_threshold
from metrics import compute_all_metrics


# Configuration
IMAGE_DIR = "/Users/tanishq/Desktop/PSO/pso_ready_cases/images"
MASK_DIR = "/Users/tanishq/Desktop/PSO/pso_ready_cases/masks"


def process_image(image_path, mask_path):
    """Process a single image and return metrics."""
    img = load_image(image_path)
    mask = load_mask(mask_path, img.shape)
    
    if mask.sum() == 0:
        # Healthy slice - no tumor
        pred = np.zeros_like(mask, dtype=np.uint8)
    else:
        # Tumor slice - use PSO to find optimal threshold
        threshold = pso_threshold(img, mask)
        pred = (img > threshold).astype(np.uint8)
    
    metrics = compute_all_metrics(mask, pred)
    return metrics


def main():
    """Main processing pipeline."""
    # Find all images
    image_extensions = ["jpg", "jpeg", "png", "JPG", "PNG"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, f"*.{ext}")))
    
    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images")
    
    # Process images
    all_metrics = []
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(MASK_DIR, f"{base_name}_mask.jpg")
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {base_name}")
            continue
        
        try:
            metrics = process_image(img_path, mask_path)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            continue
    
    # Print summary
    if all_metrics:
        metrics_array = np.array(all_metrics)
        print("\n=== Results ===")
        print(f"Dice:      {metrics_array[:, 0].mean():.4f}")
        print(f"IoU:       {metrics_array[:, 1].mean():.4f}")
        print(f"Precision: {metrics_array[:, 2].mean():.4f}")
        print(f"Recall:    {metrics_array[:, 3].mean():.4f}")


if __name__ == "__main__":
    main()
