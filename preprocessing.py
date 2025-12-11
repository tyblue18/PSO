#!/usr/bin/env python3
"""
Image preprocessing utilities for medical image segmentation.
"""

import cv2
import numpy as np


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Preprocess image with Gaussian blur and histogram equalization.
    
    Parameters
    ----------
    img : np.ndarray
        Input grayscale image
        
    Returns
    -------
    np.ndarray
        Preprocessed image normalized to [0, 1]
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Histogram equalization for contrast enhancement
    equalized = cv2.equalizeHist(blurred)
    
    # Normalize to [0, 1]
    normalized = equalized.astype(np.float32) / 255.0
    
    return normalized


def load_image(path: str) -> np.ndarray:
    """
    Load and preprocess image from file path.
    
    Parameters
    ----------
    path : str
        Path to image file
        
    Returns
    -------
    np.ndarray
        Preprocessed image
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    return preprocess_image(img)


def load_mask(path: str, target_shape: tuple) -> np.ndarray:
    """
    Load and resize mask to match target image shape.
    
    Parameters
    ----------
    path : str
        Path to mask file
    target_shape : tuple
        Target (height, width) shape
        
    Returns
    -------
    np.ndarray
        Binary mask (0 or 1)
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"Cannot read mask: {path}")
    
    # Resize to match target shape
    resized = cv2.resize(
        mask, 
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Convert to binary
    binary = (resized > 0).astype(np.uint8)
    
    return binary
