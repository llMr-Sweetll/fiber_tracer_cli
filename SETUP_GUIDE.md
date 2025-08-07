# Fiber Tracer Setup and Usage Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Testing the Application](#testing-the-application)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)
7. [Project Structure](#project-structure)
8. [Technical Details](#technical-details)

---

## Project Overview

The **Fiber Tracer** is a sophisticated Python application for analyzing fiber-reinforced polymer composites (GFRP/CFRP) from X-ray CT images. It provides:

- **3D Volume Reconstruction** from TIFF image stacks
- **Advanced Fiber Segmentation** using adaptive thresholding
- **Comprehensive Fiber Analysis** including:
  - Length, diameter, volume measurements
  - Orientation and tortuosity calculations
  - Fiber volume fraction determination
  - Connectivity analysis
- **Rich Visualizations** including heatmaps, histograms, and 3D renderings

### Key Improvements in Version 2.0

The refactored version includes:
- **Modular Architecture**: Separated into distinct modules for preprocessing, segmentation, analysis, and visualization
- **Configuration Management**: YAML/JSON configuration file support
- **Enhanced Error Handling**: Robust error handling and logging
- **Memory Optimization**: Chunked processing for large datasets
- **Parallel Processing**: Multi-core support for faster processing
- **Extended Analysis**: Added connectivity analysis and fiber classification
- **Better Visualizations**: Interactive Plotly visualizations and comprehensive reports

---

## Installation

### Prerequisites

- Python 3.6 or higher
- 8GB+ RAM recommended for large datasets
- Windows, macOS, or Linux

### Step 1: Install Dependencies

```bash
# Navigate to the project directory
cd fiber_tracer_cli

# Install required packages
pip install -r requirements.txt
```

**Note**: If you encounter issues with specific packages:

```bash
# For visualization packages
pip install matplotlib seaborn pandas

# For image processing
pip install numpy scipy scikit-image opencv-python

# For interactive plots (optional but recommended)
pip install plotly

# For YAML support
pip install pyyaml

# For Excel export (optional)
pip install openpyxl
```

### Step 2: Verify Installation

Run the test script to verify everything is working:

```bash
python test_fiber_tracer.py --test basic
```

This will create synthetic test data and run a basic pipeline test.

---

## Quick Start

### Option 1: Command Line Usage

```bash
# Basic usage with required parameters
python fiber_tracer_v2.py --data_dir /path/to/tiff/images --output_dir /path/to/output --voxel_size 1.1

# With additional parameters
python fiber_tracer_v2.py \
    --data_dir /path/to/tiff/images \
    --output_dir /path/to/output \
    --voxel_size 1.1 \
    --min_diameter 10 \
    --max_diameter 50 \
    --num_images 500 \
    --chunk_size 100
```

### Option 2: Configuration File Usage

1. Copy and modify the example configuration:

```bash
cp config_example.yaml my_config.yaml
# Edit my_config.yaml with your parameters
```

2. Run with configuration file:

```bash
python fiber_tracer_v2.py --config my_config.yaml
```

---

## Testing the Application

### Run All Tests

```bash
python test_fiber_tracer.py --test all
```

This will:
1. Create synthetic fiber data
2. Test the basic pipeline
3. Test configuration file loading
4. Generate sample outputs

### Test Output

After running tests, check the generated directories:
- `test_run/`: Basic pipeline test results
- `test_run_config/`: Configuration file test results

Each directory contains:
- `test_data/`: Synthetic TIFF images
- `test_output/`: Analysis results and visualizations

---

## Usage Examples

### Example 1: Analyzing a Small Dataset

```bash
python fiber_tracer_v2.py \
    --data_dir ./sample_data \
    --output_dir ./results \
    --voxel_size 1.1 \
    --num_images 100 \
    --chunk_size 50
```

### Example 2: High-Resolution Analysis

```bash
python fiber_tracer_v2.py \
    --data_dir ./high_res_data \
    --output_dir ./high_res_results \
    --voxel_size 0.5 \
    --min_diameter 5 \
    --max_diameter 30 \
    --scale_factor 1 \
    --num_workers 8
```

### Example 3: Memory-Constrained System

```bash
python fiber_tracer_v2.py \
    --data_dir ./large_dataset \
    --output_dir ./results \
    --voxel_size 1.1 \
    --chunk_size 25 \
    --scale_factor 2 \
    --num_workers 2
```

### Example 4: Using Python Script

```python
from fiber_tracer import FiberTracer, Config

# Create configuration
config = Config(
    data_dir="path/to/tiff/images",
    output_dir="path/to/output",
    log_level="INFO"
)

# Adjust parameters
config.analysis.voxel_size = 1.1
config.analysis.min_diameter = 10.0
config.analysis.max_diameter = 50.0

# Run analysis
tracer = FiberTracer(config)
success = tracer.run()

if success:
    print(f"Analysis complete! Found {len(tracer.fibers)} fibers")
    print(f"Volume fraction: {tracer.statistics['volume_fraction']:.2f}%")
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Errors

**Problem**: "MemoryError" or system becomes unresponsive

**Solutions**:
- Reduce `chunk_size` (e.g., from 100 to 50 or 25)
- Increase `scale_factor` to downsample images (e.g., 2 or 3)
- Process fewer images using `num_images`
- Close other applications to free memory

#### 2. No Fibers Detected

**Problem**: Analysis completes but no fibers are found

**Solutions**:
- Check image quality and contrast
- Adjust `min_diameter` and `max_diameter` parameters
- Modify segmentation parameters in config:
  ```yaml
  segmentation:
    block_size: 31  # Try different values (must be odd)
    min_object_size: 100  # Reduce if fibers are small
  ```

#### 3. Import Errors

**Problem**: "ModuleNotFoundError" for specific packages

**Solutions**:
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# Or install missing package specifically
pip install [package_name]
```

#### 4. Slow Processing

**Problem**: Analysis takes too long

**Solutions**:
- Increase `num_workers` for parallel processing
- Use `scale_factor` > 1 to reduce image resolution
- Process a subset using `num_images`

#### 5. Visualization Errors

**Problem**: Plots not generating or errors with Mayavi/Plotly

**Solutions**:
- Set `use_mayavi: false` in config (Mayavi is optional)
- Install Plotly: `pip install plotly`
- Check matplotlib backend: `python -c "import matplotlib; print(matplotlib.get_backend())"`

---

## Project Structure

```
fiber_tracer_cli/
│
├── fiber_tracer/              # Main package directory
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration management
│   ├── preprocessing.py      # Image loading and preprocessing
│   ├── segmentation.py       # Fiber segmentation algorithms
│   ├── analysis.py           # Fiber property analysis
│   ├── visualization.py      # Visualization generation
│   └── core.py              # Main pipeline orchestration
│
├── fiber_tracer_cli.py       # Original CLI script (v1)
├── fiber_tracer_v2.py        # Refactored CLI script (v2)
├── test_fiber_tracer.py      # Test suite
├── config_example.yaml       # Example configuration
├── requirements.txt          # Python dependencies
├── README.md                # Original documentation
├── SETUP_GUIDE.md           # This file
└── LICENSE                  # MIT License
```

---

## Technical Details

### Processing Pipeline

1. **Image Loading**: 
   - Loads TIFF images in sorted order
   - Validates image dimensions
   - Applies preprocessing (noise reduction, contrast enhancement)

2. **Volume Construction**:
   - Processes images in chunks for memory efficiency
   - Creates memory-mapped 3D volume
   - Handles large datasets (>15GB)

3. **Segmentation**:
   - Adaptive thresholding for local intensity variations
   - Morphological operations to clean segmentation
   - Optional advanced methods (watershed, multi-scale)

4. **Fiber Labeling**:
   - Connected component analysis
   - 26-connectivity for 3D structures
   - Size filtering to remove noise

5. **Analysis**:
   - PCA-based orientation calculation
   - Tortuosity measurement
   - Physical property extraction
   - Connectivity analysis

6. **Visualization**:
   - Statistical plots (histograms, correlations)
   - Spatial heatmaps
   - 3D renderings
   - Interactive HTML reports

### Key Parameters

#### Processing Parameters
- `chunk_size`: Images per processing batch (memory vs speed tradeoff)
- `scale_factor`: Downsampling factor (1 = original resolution)
- `num_workers`: Parallel processing threads

#### Segmentation Parameters
- `block_size`: Local threshold window size (odd number)
- `min_object_size`: Minimum fiber size in pixels
- `adaptive_threshold`: Use local vs global thresholding

#### Analysis Parameters
- `voxel_size`: Physical size of each voxel (μm)
- `min_diameter`, `max_diameter`: Fiber diameter range (μm)
- `calculate_tortuosity`: Enable tortuosity calculation
- `calculate_orientation`: Enable orientation analysis

### Output Files

The application generates:

1. **Data Files**:
   - `fiber_properties.csv`: Detailed properties of each fiber
   - `statistics.json`: Statistical summary
   - `volume_fraction.txt`: Fiber volume fraction
   - `config.json`: Used configuration

2. **Reports**:
   - `summary_report.txt`: Text summary of results
   - `fiber_classification.csv`: Fibers grouped by orientation

3. **Visualizations**:
   - `fiber_histograms.png`: Property distributions
   - `fiber_length_heatmap.png`: Spatial distribution
   - `orientation_distribution.png`: Angular distributions
   - `property_correlations.png`: Correlation analysis
   - `summary_report.png`: Visual summary
   - `fiber_3d_interactive.html`: Interactive 3D view (if Plotly available)

### Performance Considerations

- **Memory Usage**: Approximately 4 bytes per voxel
  - Example: 2000 images of 2048×2048 pixels = ~32 GB
  - Chunked processing reduces peak memory usage

- **Processing Time**: Depends on:
  - Image resolution and count
  - Number of CPU cores
  - Segmentation complexity
  - Number of detected fibers

- **Optimization Tips**:
  - Use SSD for faster I/O
  - Increase `num_workers` on multi-core systems
  - Use `scale_factor` for initial testing
  - Process representative subset first

---

## Next Steps

1. **Run the test suite** to verify installation
2. **Process sample data** with default parameters
3. **Adjust parameters** based on your specific material
4. **Review outputs** and refine configuration
5. **Scale up** to full dataset

For questions or issues, refer to the original README.md or check the log files in your output directory.

---

*Last Updated: Aug 2025*
*Version: 2.0.0*
