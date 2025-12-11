#!/usr/bin/env python3
"""
Utility functions for file handling and path management.
"""

import os
import glob


def find_image_files(directory, extensions=None):
    """
    Find all image files in a directory.
    
    Parameters
    ----------
    directory : str
        Directory to search
    extensions : list, optional
        List of file extensions to search for
        
    Returns
    -------
    list
        Sorted list of image file paths
    """
    if extensions is None:
        extensions = ["jpg", "jpeg", "png", "JPG", "PNG"]
    
    image_paths = []
    for ext in extensions:
        pattern = os.path.join(directory, f"*.{ext}")
        image_paths.extend(glob.glob(pattern))
    
    return sorted(image_paths)


def get_mask_path(image_path, mask_dir, mask_suffix="_mask"):
    """
    Generate mask path from image path.
    
    Parameters
    ----------
    image_path : str
        Path to image file
    mask_dir : str
        Directory containing masks
    mask_suffix : str
        Suffix to add before extension
        
    Returns
    -------
    str
        Path to corresponding mask file
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    # Try common mask extensions
    for ext in ["jpg", "png", "JPG", "PNG"]:
        mask_path = os.path.join(mask_dir, f"{base_name}{mask_suffix}.{ext}")
        if os.path.exists(mask_path):
            return mask_path
    
    # Return default path if not found
    return os.path.join(mask_dir, f"{base_name}{mask_suffix}.jpg")
