# PSO-Based Brain Tumor Segmentation

A medical image segmentation project using Particle Swarm Optimization (PSO) for automated brain tumor detection on MRI scans.

## Overview

This project implements a PSO-based approach to segment brain tumors from medical images, using the BraTS dataset as a benchmark. The system uses Particle Swarm Optimization to find optimal thresholds for binary segmentation, achieving robust performance through adaptive optimization.

## Features

- **PSO-based threshold optimization**: Uses global-best PSO to find optimal segmentation thresholds
- **Comprehensive evaluation**: Implements Dice, IoU, Precision, and Recall metrics
- **Visualization tools**: Generates histograms, boxplots, and qualitative examples
- **Modular architecture**: Clean separation of preprocessing, optimization, metrics, and visualization

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Required packages:
- Python 3.7+
- NumPy >= 1.21.0
- OpenCV >= 4.5.0
- PySwarms >= 1.3.0
- Matplotlib >= 3.4.0
- tqdm >= 4.62.0

## Usage

1. Update the image and mask directory paths in `main.py`:
   ```python
   IMAGE_DIR = "path/to/images"
   MASK_DIR = "path/to/masks"
   ```

2. Run the segmentation pipeline:
   ```bash
   python main.py
   ```

3. Results will be saved in:
   - `visualizations/`: Metric plots and histograms
   - `qualitative_examples/`: Side-by-side comparisons

## Project Structure

```
.
├── main.py                 # Main processing pipeline
├── pso_segmentation.py     # PSO threshold optimization
├── preprocessing.py         # Image preprocessing utilities
├── metrics.py              # Evaluation metrics
├── visualization.py        # Plotting and visualization
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Methodology

The segmentation pipeline:
1. Preprocesses images with Gaussian blur and histogram equalization
2. Uses PSO to optimize threshold for each tumor-containing slice
3. Evaluates performance using multiple metrics
4. Generates visualizations and qualitative examples

## Methodology

The segmentation pipeline:
1. Preprocesses images with Gaussian blur and histogram equalization
2. Uses PSO to optimize threshold for each tumor-containing slice
3. Falls back to Otsu's method if PSO converges to boundary values
4. Evaluates performance using multiple metrics
5. Generates visualizations and qualitative examples

## Results

The system processes medical images and generates:
- Quantitative metrics (Dice, IoU, Precision, Recall)
- Distribution plots and statistical analysis
- Qualitative visualizations comparing input, ground truth, and predictions

## License

MIT License
