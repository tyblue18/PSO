#!/usr/bin/env python3
"""
Configuration file for PSO segmentation project.
"""

import os
from typing import Tuple

# Directory paths - update these to match your setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.getenv("PSO_IMAGE_DIR", "/Users/tanishq/Desktop/PSO/pso_ready_cases/images")
MASK_DIR = os.getenv("PSO_MASK_DIR", "/Users/tanishq/Desktop/PSO/pso_ready_cases/masks")
PLOT_DIR = os.path.join(BASE_DIR, "visualizations")
TRIPLET_DIR = os.path.join(BASE_DIR, "qualitative_examples")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Create output directories
for d in [PLOT_DIR, TRIPLET_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# PSO parameters
PSO_N_PARTICLES = 30
PSO_ITERATIONS = 40
PSO_OPTIONS = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
PSO_BOUNDS: Tuple[float, float] = (0.0, 1.0)

# Visualization parameters
K_TRIPLETS = 6
TRIPLET_PROB = 0.03
PLOT_DPI = 300

# Processing parameters
USE_PARALLEL = True
N_WORKERS = None  # None = auto-detect (CPU count - 1)
BATCH_SIZE = 100  # For progress reporting

# Random seed for reproducibility
RANDOM_SEED = 0

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
