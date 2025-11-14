# OriFlake

A tool for analyzing the orientation distribution of triangular MoS₂ flake edges in thin film micrographs.

[中文版 README](README_CN.md)

## Overview

OriFlake is designed to process micrograph images of thin films and statistically analyze the orientation distribution of flake edges. It supports both command-line and GUI interfaces, making it suitable for batch processing and interactive analysis.

## Features

- **Dual Interface**: Command-line tool for batch processing and modern GUI for interactive analysis
- **Robust Image Processing**: Supports various image formats (8-bit, 16-bit, 32-bit, floating-point)
- **Automatic Color Peak Detection**: Identifies dual color peaks from RGB histograms for thresholding
- **Edge Detection & Line Fitting**: Advanced edge detection with Hough transform for line segment extraction
- **Orientation Statistics**: Comprehensive statistical analysis of edge orientations
- **Visualization**: Generates overlay images, edge maps, histograms, and statistical plots

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

The project uses a YAML configuration file (`oriflake/config.yaml`) to control processing parameters. Key configuration sections include:

### File Paths
- `input`: Input image path (file or directory)
- `output`: Output directory path

### Image Preprocessing
- `crop_ratio`: Center region crop ratio (0.3 means crop 30% from edges, keeping center 70%)
- `threshold_offset`: Threshold offset for fine-tuning binarization (positive = higher threshold, negative = lower)

### Segmentation Parameters
- `morph_open_ks`: Morphological opening kernel size for noise removal
- `morph_close_ks`: Morphological closing kernel size (0 = disabled)

### Edge Detection Parameters
- `boundary_kernel_size`: Morphological kernel size for boundary extraction
- `edge_cleanup_kernel_size`: Kernel size for edge cleanup
- `min_contour_area`: Minimum contour area (pixels) for filtering small edge regions

### Line Fitting Parameters
- `min_line_length`: Minimum line segment length (pixels)
- `max_line_gap`: Maximum gap between line segments (pixels)
- `hough_threshold`: Hough transform threshold
- `min_segment_distance`: Minimum distance between segments to avoid duplicate counting

### Other Options
- `verbose`: Enable detailed logging
- `preview`: Show preview window during processing

## Usage

### Command-Line Interface

Basic usage with default configuration:

```bash
python -m oriflake.main
```

With custom configuration file:

```bash
python -m oriflake.main --config path/to/config.yaml
```

Override configuration parameters:

```bash
python -m oriflake.main --input path/to/images --output path/to/output --verbose
```

Available command-line arguments:
- `--config`: Path to configuration file (overrides default config.yaml)
- `--input`: Input image path or directory (overrides config)
- `--output`: Output directory (overrides config)
- `--preview`: Show preview window (overrides config)
- `--verbose`: Print detailed logs (overrides config)

### GUI Interface

Launch the graphical user interface:

```bash
python oriflake_gui.py
```

The GUI provides:
- Interactive image loading and processing
- Real-time parameter adjustment
- Step-by-step visualization of processing pipeline
- Incremental processing from specific steps
- Export of results and visualizations

## Implementation Details

### Processing Pipeline

The image processing pipeline consists of the following steps:

1. **Image Loading & Preprocessing**
   - Load image in original bit depth
   - Convert to 8-bit format (0-255 range per channel)
   - Support for various formats: 8-bit, 16-bit, 24-bit RGB, 32-bit, floating-point

2. **Color Peak Detection**
   - Analyze RGB histogram of full image
   - Identify dual color peaks (color1, color2) for threshold calculation
   - Use full image to avoid missing edge peaks after cropping

3. **Center Region Cropping**
   - Crop center region based on `crop_ratio` parameter
   - Reduces processing area and focuses on central region

4. **Blue Channel Extraction & Thresholding**
   - Extract blue channel from cropped RGB image
   - Calculate threshold from dual peak average
   - Apply threshold offset for fine-tuning
   - Generate binary mask

5. **Edge Detection & Enhancement**
   - Extract boundaries from binary mask
   - Apply morphological operations for cleanup
   - Filter small contours and border regions
   - Enhance edges using robust algorithms

6. **Line Segment Fitting**
   - Apply Hough transform to detect line segments
   - Filter segments by minimum length and maximum gap
   - Remove duplicate segments based on distance threshold
   - Extract orientation angles from line segments

7. **Statistics & Visualization**
   - Calculate orientation statistics
   - Generate overlay images with detected edges
   - Create histograms and statistical plots
   - Export results to CSV format

### Output Files

For each processed image, the tool generates:
- `{filename}_overlay.png`: Original image with detected edges overlaid
- `{filename}_edges.png`: Edge map visualization
- `{filename}_gray.png`: Blue channel grayscale image
- `{filename}_binary.png`: Binary threshold mask
- `{filename}_histogram.png`: RGB histogram with color peaks and threshold

Aggregate outputs:
- `oriflake_results.csv`: CSV file containing all detected edge segments with angles
- `orientations_hist.png`: Histogram of all orientation angles across all images

## Project Structure

```
OriFlake/
├── oriflake/              # Main package
│   ├── __init__.py
│   ├── main.py            # Command-line interface and core processing
│   ├── config.yaml        # Default configuration file
│   ├── geometry.py        # Geometric operations
│   ├── orientation_stats.py  # Orientation statistics
│   ├── seg.py             # Segmentation functions
│   └── utils.py           # Utility functions
├── oriflake_gui.py        # GUI application
├── requirements.txt       # Python dependencies
├── README.md             # This file (English)
└── README_CN.md          # Chinese documentation
```

## Dependencies

See `requirements.txt` for the complete list of dependencies. Main packages include:
- PyQt6: GUI framework
- OpenCV (cv2): Image processing
- NumPy: Numerical operations
- Pandas: Data handling
- Matplotlib: Visualization
- SciPy: Scientific computing
- PyYAML: Configuration file parsing
- scikit-image: Additional image processing utilities
- Seaborn: Statistical visualization

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here]

