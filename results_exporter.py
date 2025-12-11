#!/usr/bin/env python3
"""
Export results to various formats (JSON, CSV, etc.)
"""

import json
import csv
import os
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime


def export_results_to_json(
    results: Dict,
    output_path: str
) -> None:
    """
    Export results dictionary to JSON file.
    
    Parameters
    ----------
    results : dict
        Results dictionary with metrics and statistics
    output_path : str
        Path to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            json_results[key] = float(value)
        else:
            json_results[key] = value
    
    json_results['timestamp'] = datetime.now().isoformat()
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)


def export_results_to_csv(
    metrics_list: List[Tuple[str, Tuple[float, float, float, float], float]],
    output_path: str
) -> None:
    """
    Export per-image metrics to CSV file.
    
    Parameters
    ----------
    metrics_list : list
        List of (image_name, (dice, iou, precision, recall), threshold) tuples
    output_path : str
        Path to save CSV file
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Dice', 'IoU', 'Precision', 'Recall', 'Threshold'])
        
        for image_name, metrics, threshold in metrics_list:
            dice, iou, precision, recall = metrics
            threshold_str = f"{threshold:.6f}" if threshold is not None else "N/A"
            writer.writerow([
                image_name,
                f"{dice:.6f}",
                f"{iou:.6f}",
                f"{precision:.6f}",
                f"{recall:.6f}",
                threshold_str
            ])


def generate_summary_statistics(
    tumour_metrics: List[Tuple[float, float, float, float]],
    healthy_metrics: List[Tuple[float, float, float, float]],
    tumour_thresholds: List[float]
) -> Dict:
    """
    Generate comprehensive statistics from metrics.
    
    Parameters
    ----------
    tumour_metrics : list
        List of (dice, iou, precision, recall) tuples for tumor cases
    healthy_metrics : list
        List of (dice, iou, precision, recall) tuples for healthy cases
    tumour_thresholds : list
        List of optimal thresholds for tumor cases
        
    Returns
    -------
    dict
        Dictionary with summary statistics
    """
    stats = {}
    
    if tumour_metrics:
        tumour_arr = np.array(tumour_metrics)
        stats['tumour'] = {
            'count': len(tumour_metrics),
            'dice': {
                'mean': float(np.mean(tumour_arr[:, 0])),
                'std': float(np.std(tumour_arr[:, 0])),
                'min': float(np.min(tumour_arr[:, 0])),
                'max': float(np.max(tumour_arr[:, 0])),
                'median': float(np.median(tumour_arr[:, 0]))
            },
            'iou': {
                'mean': float(np.mean(tumour_arr[:, 1])),
                'std': float(np.std(tumour_arr[:, 1])),
                'min': float(np.min(tumour_arr[:, 1])),
                'max': float(np.max(tumour_arr[:, 1])),
                'median': float(np.median(tumour_arr[:, 1]))
            },
            'precision': {
                'mean': float(np.mean(tumour_arr[:, 2])),
                'std': float(np.std(tumour_arr[:, 2])),
                'min': float(np.min(tumour_arr[:, 2])),
                'max': float(np.max(tumour_arr[:, 2])),
                'median': float(np.median(tumour_arr[:, 2]))
            },
            'recall': {
                'mean': float(np.mean(tumour_arr[:, 3])),
                'std': float(np.std(tumour_arr[:, 3])),
                'min': float(np.min(tumour_arr[:, 3])),
                'max': float(np.max(tumour_arr[:, 3])),
                'median': float(np.median(tumour_arr[:, 3]))
            },
            'threshold': {
                'mean': float(np.mean(tumour_thresholds)),
                'std': float(np.std(tumour_thresholds)),
                'min': float(np.min(tumour_thresholds)),
                'max': float(np.max(tumour_thresholds)),
                'median': float(np.median(tumour_thresholds))
            }
        }
    
    if healthy_metrics:
        healthy_arr = np.array(healthy_metrics)
        stats['healthy'] = {
            'count': len(healthy_metrics),
            'dice': {
                'mean': float(np.mean(healthy_arr[:, 0])),
                'std': float(np.std(healthy_arr[:, 0])),
                'min': float(np.min(healthy_arr[:, 0])),
                'max': float(np.max(healthy_arr[:, 0])),
                'median': float(np.median(healthy_arr[:, 0]))
            },
            'iou': {
                'mean': float(np.mean(healthy_arr[:, 1])),
                'std': float(np.std(healthy_arr[:, 1])),
                'min': float(np.min(healthy_arr[:, 1])),
                'max': float(np.max(healthy_arr[:, 1])),
                'median': float(np.median(healthy_arr[:, 1]))
            },
            'precision': {
                'mean': float(np.mean(healthy_arr[:, 2])),
                'std': float(np.std(healthy_arr[:, 2])),
                'min': float(np.min(healthy_arr[:, 2])),
                'max': float(np.max(healthy_arr[:, 2])),
                'median': float(np.median(healthy_arr[:, 2]))
            },
            'recall': {
                'mean': float(np.mean(healthy_arr[:, 3])),
                'std': float(np.std(healthy_arr[:, 3])),
                'min': float(np.min(healthy_arr[:, 3])),
                'max': float(np.max(healthy_arr[:, 3])),
                'median': float(np.median(healthy_arr[:, 3]))
            }
        }
    
    return stats
