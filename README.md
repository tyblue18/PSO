# PSO-Based Brain Tumor Segmentation

A medical image segmentation project using Particle Swarm Optimization (PSO) for automated brain tumor detection on MRI scans.

## Overview

This project implements a PSO-based approach to segment brain tumors from medical images, using the BraTS dataset as a benchmark. The system uses Particle Swarm Optimization to find optimal thresholds for binary segmentation, achieving robust performance through adaptive optimization.

## Features

- **PSO-based threshold optimization**: Uses global-best PSO to find optimal segmentation thresholds
- **Parallel processing**: Multi-core support for faster batch processing
- **Comprehensive evaluation**: Implements Dice, IoU, Precision, and Recall metrics with detailed statistics
- **Visualization tools**: Generates histograms, boxplots, and qualitative examples
- **Result export**: Exports results to JSON and CSV formats
- **Logging system**: Comprehensive logging for debugging and monitoring
- **Modular architecture**: Clean separation of preprocessing, optimization, metrics, and visualization
- **Command-line interface**: Flexible CLI with configurable parameters

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

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tyblue18/PSO.git
   cd PSO
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Update configuration in `config.py`:
   ```python
   IMAGE_DIR = "path/to/images"
   MASK_DIR = "path/to/masks"
   ```

## Usage

### Basic Usage

Run the segmentation pipeline with default settings:

```bash
python main.py
```

### Command-Line Options

```bash
python main.py [OPTIONS]

Options:
  --k-triplets INT      Maximum number of qualitative examples (default: 6)
  --triplet-prob FLOAT  Probability per tumor slice to save (default: 0.03)
  --no-parallel         Disable parallel processing
  --workers INT         Number of worker processes (default: auto-detect)
  --no-export           Disable result export
```

### Examples

Process with custom triplet count:
```bash
python main.py --k-triplets 10 --triplet-prob 0.05
```

Disable parallel processing (useful for debugging):
```bash
python main.py --no-parallel
```

Specify number of workers:
```bash
python main.py --workers 4
```

### Output

Results are saved in:
- `visualizations/`: Metric plots and histograms
- `qualitative_examples/`: Side-by-side comparisons
- `results/`: JSON summary statistics and CSV per-image metrics
- `logs/`: Processing logs with timestamps

## Project Structure

```
.
├── main.py                 # Main processing pipeline
├── pso_segmentation.py     # PSO threshold optimization (optimized)
├── preprocessing.py        # Image preprocessing utilities
├── metrics.py              # Evaluation metrics (optimized)
├── visualization.py        # Plotting and visualization
├── parallel_processing.py # Parallel batch processing
├── results_exporter.py     # Export results to JSON/CSV
├── logger_config.py        # Logging configuration
├── config.py              # Centralized configuration
├── utils.py               # Utility functions
├── test_metrics.py        # Unit tests for metrics
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Methodology

The segmentation pipeline:

1. **Preprocessing**: Applies Gaussian blur and histogram equalization to enhance image quality
2. **PSO Optimization**: Uses Particle Swarm Optimization to find optimal threshold for each tumor-containing slice
3. **Fallback Mechanism**: Falls back to Otsu's method if PSO converges to boundary values
4. **Evaluation**: Computes comprehensive metrics (Dice, IoU, Precision, Recall)
5. **Visualization**: Generates distribution plots and qualitative examples
6. **Export**: Saves detailed statistics and per-image results

## Performance Improvements

This version includes several optimizations:

- **Vectorized operations**: Faster metric computation using NumPy vectorization
- **Parallel processing**: Multi-core support for batch image processing
- **Optimized PSO**: Improved objective function with precomputed values
- **Efficient I/O**: Better file handling and path management
- **Memory optimization**: Reduced memory footprint for large datasets

## Results

The system processes medical images and generates:

- **Quantitative metrics**: Dice, IoU, Precision, Recall with mean, std, min, max, median
- **Distribution plots**: Histograms and boxplots for metric analysis
- **Qualitative visualizations**: Side-by-side comparisons of input, ground truth, and predictions
- **Exported data**: JSON summary statistics and CSV per-image metrics

## Testing

Run unit tests:

```bash
python test_metrics.py
```

## Configuration

All configuration is centralized in `config.py`. Key settings:

- Directory paths (IMAGE_DIR, MASK_DIR)
- PSO parameters (particles, iterations, options)
- Visualization settings
- Processing options (parallel, workers)

## Logging

The project includes comprehensive logging:

- File logs saved in `logs/` directory with timestamps
- Console output for real-time monitoring
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pso_brain_tumor_segmentation,
  title={PSO-Based Brain Tumor Segmentation},
  author={Your Name},
  year={2025},
  url={https://github.com/tyblue18/PSO}
}
```
