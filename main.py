#!/usr/bin/env python3
"""
Main pipeline for PSO-based brain tumor segmentation.
Improved version with parallel processing, logging, and result export.
"""

import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional

import config
from preprocessing import load_image, load_mask
from pso_segmentation import pso_threshold
from metrics import compute_all_metrics
from utils import find_image_files, get_mask_path
from visualization import (
    save_triplet_comparison,
    plot_dice_histogram,
    plot_metrics_boxplot,
    plot_threshold_vs_dice
)
from logger_config import setup_logger
from parallel_processing import process_images_parallel
from results_exporter import export_results_to_json, export_results_to_csv, generate_summary_statistics


def process_image_sequential(
    image_path: str,
    mask_path: str
) -> Tuple[Tuple[float, float, float, float], Optional[float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single image sequentially.
    
    Parameters
    ----------
    image_path : str
        Path to image file
    mask_path : str
        Path to mask file
        
    Returns
    -------
    tuple
        (metrics, threshold, img, mask, pred)
    """
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


def main(
    k_triplets: int = config.K_TRIPLETS,
    triplet_prob: float = config.TRIPLET_PROB,
    use_parallel: bool = config.USE_PARALLEL,
    n_workers: Optional[int] = config.N_WORKERS,
    export_results: bool = True
):
    """
    Main processing pipeline with improved features.
    
    Parameters
    ----------
    k_triplets : int
        Maximum number of qualitative examples to save
    triplet_prob : float
        Probability per tumor slice to be saved
    use_parallel : bool
        Whether to use parallel processing
    n_workers : int, optional
        Number of worker processes (None = auto-detect)
    export_results : bool
        Whether to export results to JSON/CSV
    """
    # Setup logging
    logger = setup_logger(level=getattr(__import__('logging'), config.LOG_LEVEL))
    logger.info("Starting PSO-based brain tumor segmentation pipeline")
    logger.info(f"Image directory: {config.IMAGE_DIR}")
    logger.info(f"Mask directory: {config.MASK_DIR}")
    
    random.seed(config.RANDOM_SEED)
    
    # Find all images
    image_paths = find_image_files(config.IMAGE_DIR)
    logger.info(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        logger.error("No images found! Check IMAGE_DIR in config.py")
        return
    
    # Process images
    tumour_metrics = []
    healthy_metrics = []
    tumour_dice = []
    tumour_thresholds = []
    tumour_image_names = []
    healthy_image_names = []
    saved_triplets = 0
    
    if use_parallel and len(image_paths) > 1:
        logger.info(f"Using parallel processing with {n_workers or 'auto'} workers")
        # Process in parallel (without saving triplets - do that sequentially)
        results = process_images_parallel(
            image_paths,
            config.MASK_DIR,
            n_workers=n_workers,
            process_tumor=True
        )
        
        logger.info(f"Successfully processed {len(results)} images")
        
        # Organize results
        for base_name, metrics, threshold, img, mask, pred in results:
            if mask.sum() > 0:  # Tumor slice
                tumour_metrics.append(metrics)
                tumour_dice.append(metrics[0])
                tumour_thresholds.append(threshold)
                tumour_image_names.append(base_name)
            else:  # Healthy slice
                healthy_metrics.append(metrics)
                healthy_image_names.append(base_name)
        
        # Save qualitative examples sequentially (to avoid race conditions)
        logger.info("Saving qualitative examples...")
        random.shuffle(tumour_image_names)
        for base_name in tumour_image_names[:k_triplets]:
            if random.random() < triplet_prob:
                try:
                    img_path = next(p for p in image_paths if base_name in p)
                    mask_path = get_mask_path(img_path, config.MASK_DIR)
                    _, _, img, mask, pred = process_image_sequential(img_path, mask_path)
                    triplet_path = os.path.join(config.TRIPLET_DIR, f"{base_name}_triplet.png")
                    save_triplet_comparison(img, mask, pred, triplet_path)
                    saved_triplets += 1
                    if saved_triplets >= k_triplets:
                        break
                except Exception as e:
                    logger.warning(f"Failed to save triplet for {base_name}: {e}")
    else:
        # Sequential processing
        logger.info("Using sequential processing")
        for img_path in tqdm(image_paths, desc="Processing"):
            mask_path = get_mask_path(img_path, config.MASK_DIR)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            if not os.path.exists(mask_path):
                logger.warning(f"Mask not found for {base_name}")
                continue
            
            try:
                metrics, threshold, img, mask, pred = process_image_sequential(img_path, mask_path)
                
                if mask.sum() > 0:  # Tumor slice
                    tumour_metrics.append(metrics)
                    tumour_dice.append(metrics[0])
                    tumour_thresholds.append(threshold)
                    tumour_image_names.append(base_name)
                    
                    # Save qualitative examples
                    if saved_triplets < k_triplets and random.random() < triplet_prob:
                        triplet_path = os.path.join(config.TRIPLET_DIR, f"{base_name}_triplet.png")
                        save_triplet_comparison(img, mask, pred, triplet_path)
                        saved_triplets += 1
                else:  # Healthy slice
                    healthy_metrics.append(metrics)
                    healthy_image_names.append(base_name)
                    
            except Exception as e:
                logger.error(f"Error processing {base_name}: {e}")
                continue
    
    # Print and log summary
    logger.info("=" * 50)
    if tumour_metrics:
        metrics_array = np.array(tumour_metrics)
        stats = {
            'dice': (metrics_array[:, 0].mean(), metrics_array[:, 0].std()),
            'iou': (metrics_array[:, 1].mean(), metrics_array[:, 1].std()),
            'precision': (metrics_array[:, 2].mean(), metrics_array[:, 2].std()),
            'recall': (metrics_array[:, 3].mean(), metrics_array[:, 3].std())
        }
        
        print("\n=== Tumor Cases ===")
        print(f"Count:     {len(tumour_metrics)}")
        print(f"Dice:      {stats['dice'][0]:.4f} ± {stats['dice'][1]:.4f}")
        print(f"IoU:       {stats['iou'][0]:.4f} ± {stats['iou'][1]:.4f}")
        print(f"Precision: {stats['precision'][0]:.4f} ± {stats['precision'][1]:.4f}")
        print(f"Recall:    {stats['recall'][0]:.4f} ± {stats['recall'][1]:.4f}")
        
        logger.info(f"Tumor cases: {len(tumour_metrics)}")
        logger.info(f"Mean Dice: {stats['dice'][0]:.4f} ± {stats['dice'][1]:.4f}")
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        plot_dice_histogram(tumour_dice, os.path.join(config.PLOT_DIR, "dice_hist_tumour.png"))
        plot_metrics_boxplot(metrics_array, os.path.join(config.PLOT_DIR, "metric_spread_tumour.png"))
        plot_threshold_vs_dice(tumour_thresholds, tumour_dice, 
                              os.path.join(config.PLOT_DIR, "tau_vs_dice.png"))
        print(f"\nVisualizations saved to {config.PLOT_DIR}")
    
    if healthy_metrics:
        metrics_array = np.array(healthy_metrics)
        stats = {
            'dice': (metrics_array[:, 0].mean(), metrics_array[:, 0].std()),
            'iou': (metrics_array[:, 1].mean(), metrics_array[:, 1].std()),
            'precision': (metrics_array[:, 2].mean(), metrics_array[:, 2].std()),
            'recall': (metrics_array[:, 3].mean(), metrics_array[:, 3].std())
        }
        
        print("\n=== Healthy Cases ===")
        print(f"Count:     {len(healthy_metrics)}")
        print(f"Dice:      {stats['dice'][0]:.4f} ± {stats['dice'][1]:.4f}")
        print(f"IoU:       {stats['iou'][0]:.4f} ± {stats['iou'][1]:.4f}")
        print(f"Precision: {stats['precision'][0]:.4f} ± {stats['precision'][1]:.4f}")
        print(f"Recall:    {stats['recall'][0]:.4f} ± {stats['recall'][1]:.4f}")
        
        logger.info(f"Healthy cases: {len(healthy_metrics)}")
    
    # Export results
    if export_results:
        logger.info("Exporting results...")
        summary_stats = generate_summary_statistics(
            tumour_metrics, healthy_metrics, tumour_thresholds
        )
        
        # Export summary to JSON
        json_path = os.path.join(config.RESULTS_DIR, "summary_statistics.json")
        export_results_to_json(summary_stats, json_path)
        logger.info(f"Summary statistics saved to {json_path}")
        
        # Export per-image metrics to CSV
        if tumour_metrics:
            csv_data = [
                (name, metrics, threshold)
                for name, metrics, threshold in zip(tumour_image_names, tumour_metrics, tumour_thresholds)
            ]
            csv_path = os.path.join(config.RESULTS_DIR, "tumour_metrics.csv")
            export_results_to_csv(csv_data, csv_path)
            logger.info(f"Tumour metrics saved to {csv_path}")
        
        if healthy_metrics:
            csv_data = [
                (name, metrics, None)
                for name, metrics in zip(healthy_image_names, healthy_metrics)
            ]
            csv_path = os.path.join(config.RESULTS_DIR, "healthy_metrics.csv")
            export_results_to_csv(csv_data, csv_path)
            logger.info(f"Healthy metrics saved to {csv_path}")
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSO-based brain tumor segmentation")
    parser.add_argument(
        "--k-triplets",
        type=int,
        default=config.K_TRIPLETS,
        help="Maximum number of qualitative examples to save"
    )
    parser.add_argument(
        "--triplet-prob",
        type=float,
        default=config.TRIPLET_PROB,
        help="Probability per tumor slice to be saved"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: auto-detect)"
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Disable result export"
    )
    
    args = parser.parse_args()
    
    main(
        k_triplets=args.k_triplets,
        triplet_prob=args.triplet_prob,
        use_parallel=not args.no_parallel,
        n_workers=args.workers,
        export_results=not args.no_export
    )
