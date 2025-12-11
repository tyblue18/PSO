#!/usr/bin/env python3
"""
Configuration file for PSO segmentation project.
"""

import os

# Directory paths - update these to match your setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = "/Users/tanishq/Desktop/PSO/pso_ready_cases/images"
MASK_DIR = "/Users/tanishq/Desktop/PSO/pso_ready_cases/masks"
PLOT_DIR = os.path.join(BASE_DIR, "visualizations")
TRIPLET_DIR = os.path.join(BASE_DIR, "qualitative_examples")

# PSO parameters
PSO_N_PARTICLES = 30
PSO_ITERATIONS = 40
PSO_OPTIONS = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
PSO_BOUNDS = (0.0, 1.0)

# Visualization parameters
K_TRIPLETS = 6
TRIPLET_PROB = 0.03
PLOT_DPI = 300

# Random seed for reproducibility
RANDOM_SEED = 0
