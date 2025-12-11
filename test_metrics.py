#!/usr/bin/env python3
"""
Unit tests for metrics module.
"""

import numpy as np
from metrics import (
    dice_coefficient,
    iou_coefficient,
    precision_coefficient,
    recall_coefficient,
    compute_all_metrics
)


def test_perfect_match():
    """Test metrics with perfect prediction."""
    gt = np.ones((10, 10), dtype=np.uint8)
    pred = np.ones((10, 10), dtype=np.uint8)
    
    assert dice_coefficient(gt, pred) == 1.0
    assert iou_coefficient(gt, pred) == 1.0
    assert precision_coefficient(gt, pred) == 1.0
    assert recall_coefficient(gt, pred) == 1.0


def test_no_overlap():
    """Test metrics with no overlap."""
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[0:5, 0:5] = 1
    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[5:10, 5:10] = 1
    
    assert dice_coefficient(gt, pred) == 0.0
    assert iou_coefficient(gt, pred) == 0.0
    assert precision_coefficient(gt, pred) == 0.0
    assert recall_coefficient(gt, pred) == 0.0


def test_both_empty():
    """Test metrics when both masks are empty."""
    gt = np.zeros((10, 10), dtype=np.uint8)
    pred = np.zeros((10, 10), dtype=np.uint8)
    
    assert dice_coefficient(gt, pred) == 1.0
    assert iou_coefficient(gt, pred) == 1.0
    assert precision_coefficient(gt, pred) == 1.0
    assert recall_coefficient(gt, pred) == 1.0


def test_partial_overlap():
    """Test metrics with partial overlap."""
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[0:7, 0:7] = 1
    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[3:10, 3:10] = 1
    
    # Should have some overlap
    dice = dice_coefficient(gt, pred)
    iou = iou_coefficient(gt, pred)
    
    assert 0.0 < dice < 1.0
    assert 0.0 < iou < 1.0
    assert iou <= dice  # IoU is always <= Dice


def test_compute_all_metrics():
    """Test compute_all_metrics function."""
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[0:5, 0:5] = 1
    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[0:5, 0:5] = 1
    
    dice, iou, precision, recall = compute_all_metrics(gt, pred)
    
    assert dice == 1.0
    assert iou == 1.0
    assert precision == 1.0
    assert recall == 1.0


if __name__ == "__main__":
    print("Running metric tests...")
    test_perfect_match()
    print("✓ Perfect match test passed")
    
    test_no_overlap()
    print("✓ No overlap test passed")
    
    test_both_empty()
    print("✓ Both empty test passed")
    
    test_partial_overlap()
    print("✓ Partial overlap test passed")
    
    test_compute_all_metrics()
    print("✓ Compute all metrics test passed")
    
    print("\nAll tests passed!")
