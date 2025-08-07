# 🔬 Fiber Tracer v2.0
## Advanced 3D Analysis Tool for Fiber-Reinforced Polymer Composites

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.0.0-orange)](https://github.com/yourusername/fiber_tracer)

A comprehensive Python application for analyzing fiber-reinforced polymer composites (GFRP/CFRP) from X-ray computed tomography (CT) images. This tool provides automated fiber detection, tracking, and quantitative analysis with advanced visualization capabilities.

---

## 📋 Table of Contents

- [Features](#-features)
- [What's New in v2.0](#-whats-new-in-v20)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Output Files](#-output-files)
- [API Documentation](#-api-documentation)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

---

## ✨ Features

### Core Capabilities
- **🖼️ 3D Volume Reconstruction**: Build 3D volumes from TIFF image stacks
- **🔍 Advanced Segmentation**: Multiple segmentation algorithms (adaptive, Otsu, watershed)
- **📊 Comprehensive Analysis**: Extract 15+ fiber properties per fiber
- **📈 Rich Visualizations**: Interactive 3D plots, heatmaps, and statistical charts
- **⚡ High Performance**: Parallel processing and memory-optimized algorithms
- **🔧 Flexible Configuration**: YAML/JSON configuration files

### Measured Properties
| Property | Unit | Description |
|----------|------|-------------|
| Length | μm | Major axis length of fiber |
| Diameter | μm | Equivalent circular diameter |
| Volume | μm³ | Total fiber volume |
| Surface Area | μm² | Fiber surface area |
| Orientation | degrees | Angle with respect to Z-axis |
| Tortuosity | ratio | Path length / straight distance |
| Polar Angle | degrees | Angle from Z-axis |
| Azimuthal Angle | degrees | Angle in XY-plane |
| Aspect Ratio | ratio | Length / width ratio |
| Volume Fraction | % | Fiber volume / total volume |

---

## 🆕 What's New in v2.0

### Architecture Improvements
- ✅ **Modular Design**: Separated into distinct modules for better maintainability
- ✅ **Package Structure**: Proper Python package with namespace organization
- ✅ **Configuration Management**: Dataclass-based configuration with validation
- ✅ **Enhanced Logging**: Comprehensive logging with file and console output

### New Features
- ✅ **Connectivity Analysis**: Analyze fiber-to-fiber connections
- ✅ **Fiber Classification**: Automatic grouping by orientation
- ✅ **Interactive Visualizations**: Plotly-based 3D interactive plots
- ✅ **Batch Processing**: Process multiple datasets with different configurations
- ✅ **Memory Optimization**: Chunked processing for datasets >15GB

### Performance Enhancements
- ✅ **Parallel Processing**: Multi-core support with configurable workers
- ✅ **Memory Mapping**: Handle large volumes without loading into RAM
- ✅ **Optimized Algorithms**: Faster segmentation and analysis

---

## 💻 Installation

### System Requirements
- **OS**: Windows 10+, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 2x size of dataset for processing

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/fiber_tracer.git
cd fiber_tracer/fiber_tracer_cli
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n fiber_tracer python=3.9
conda activate fiber_tracer
```

### Step 3: Install Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install manually
pip install numpy scipy scikit-image opencv-python matplotlib pandas seaborn pyyaml plotly tqdm openpyxl
```

### Step 4: Verify Installation
```bash
python test_fiber_tracer.py --test basic
```

---

## 🚀 Quick Start

### Basic Usage
```bash
python fiber_tracer_v2.py \
    --data_dir ./sample_data \
    --output_dir ./results \
    --voxel_size 1.1
```

### Using Configuration File
```bash
# Copy and edit configuration
cp config_example.yaml my_config.yaml

# Run with config
python fiber_tracer_v2.py --config my_config.yaml
```

### Python API
```python
from fiber_tracer import FiberTracer, Config

# Create configuration
config = Config(
    data_dir="path/to/tiff/images",
    output_dir="path/to/output",
    log_level="INFO"
)

# Configure parameters
config.analysis.voxel_size = 1.1
config.analysis.min_diameter = 10.0

# Run analysis
tracer = FiberTracer(config)
success = tracer.run()

# Access results
if success:
    print(f"Found {len(tracer.fibers)} fibers")
    print(f"Volume fraction: {tracer.statistics['volume_fraction']:.2f}%")
```

---

## 📖 Usage

### Command Line Interface

#### Required Arguments
- `--data_dir`: Directory containing TIFF images
- `--output_dir`: Directory for output files
- `--voxel_size`: Physical size of voxel in micrometers

#### Optional Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | None | Path to configuration file |
| `--num_images` | All | Number of images to process |
| `--min_diameter` | 10.0 | Minimum fiber diameter (μm) |
| `--max_diameter` | 50.0 | Maximum fiber diameter (μm) |
| `--chunk_size` | 100 | Images per processing chunk |
| `--scale_factor` | 1 | Downscaling factor |
| `--num_workers` | CPU count | Parallel processing threads |
| `--log_level` | INFO | Logging verbosity |

### Examples

#### Example 1: Process subset of images
```bash
python fiber_tracer_v2.py \
    --data_dir ./data \
    --output_dir ./output \
    --voxel_size 1.1 \
    --num_images 500
```

#### Example 2: High-resolution analysis
```bash
python fiber_tracer_v2.py \
    --data_dir ./high_res_data \
    --output_dir ./high_res_output \
    --voxel_size 0.5 \
    --min_diameter 5 \
    --max_diameter 30 \
    --scale_factor 1 \
    --num_workers 8
```

#### Example 3: Memory-constrained system
```bash
python fiber_tracer_v2.py \
    --data_dir ./large_dataset \
    --output_dir ./output \
    --voxel_size 1.1 \
    --chunk_size 25 \
    --scale_factor 2 \
    --num_workers 2
```

---

## ⚙️ Configuration

### Configuration File Format

The application supports YAML and JSON configuration files:

```yaml
# config.yaml
data_dir: "/path/to/tiff/images"
output_dir: "/path/to/output"
log_level: "INFO"

processing:
  chunk_size: 100
  scale_factor: 1
  num_workers: 4
  median_disk_size: 2
  gaussian_sigma: 1.0
  clahe_clip_limit: 0.03

segmentation:
  block_size: 51
  min_object_size: 500
  adaptive_threshold: true
  fill_holes: true

analysis:
  min_diameter: 10.0
  max_diameter: 50.0
  voxel_size: 1.1
  calculate_tortuosity: true
  calculate_orientation: true

visualization:
  generate_heatmap: true
  generate_histogram: true
  generate_3d_visualization: true
  use_plotly: true
  colormap: "viridis"
  figure_dpi: 150
```

### Parameter Groups

#### Processing Parameters
- `chunk_size`: Number of images per batch (affects memory usage)
- `scale_factor`: Image downscaling factor (1 = no scaling)
- `num_workers`: Parallel processing threads
- `median_disk_size`: Size of median filter kernel
- `gaussian_sigma`: Gaussian smoothing parameter
- `clahe_clip_limit`: Contrast enhancement limit

#### Segmentation Parameters
- `block_size`: Adaptive threshold window size (must be odd)
- `min_object_size`: Minimum fiber size in pixels
- `adaptive_threshold`: Use local vs global thresholding
- `fill_holes`: Fill holes in segmented objects

#### Analysis Parameters
- `voxel_size`: Physical size of voxel (μm)
- `min_diameter`: Minimum fiber diameter (μm)
- `max_diameter`: Maximum fiber diameter (μm)
- `calculate_tortuosity`: Enable tortuosity calculation
- `calculate_orientation`: Enable orientation analysis

---

## 📁 Output Files

### Data Files
| File | Format | Description |
|------|--------|-------------|
| `fiber_properties.csv` | CSV | Detailed properties of each fiber |
| `statistics.json` | JSON | Statistical summary of all fibers |
| `volume_fraction.txt` | Text | Fiber volume fraction percentage |
| `config.json` | JSON | Configuration used for processing |
| `summary_report.txt` | Text | Human-readable summary |

### Visualizations
| File | Type | Description |
|------|------|-------------|
| `fiber_histograms.png` | Image | Distribution plots of fiber properties |
| `fiber_length_heatmap.png` | Image | Spatial distribution of fiber lengths |
| `orientation_distribution.png` | Image | Angular distribution plots |
| `property_correlations.png` | Image | Correlation matrix and scatter plots |
| `summary_report.png` | Image | Visual summary dashboard |
| `fiber_3d_interactive.html` | HTML | Interactive 3D visualization |

### Log Files
- `fiber_tracer_YYYYMMDD_HHMMSS.log`: Detailed processing log

---

## 📚 API Documentation

### Main Classes

#### `FiberTracer`
Main orchestrator class for the analysis pipeline.

```python
from fiber_tracer import FiberTracer, Config

tracer = FiberTracer(config: Config)
success = tracer.run() -> bool
```

#### `Config`
Configuration management class.

```python
from fiber_tracer import Config

# Create from arguments
config = Config(
    data_dir="path/to/data",
    output_dir="path/to/output"
)

# Load from file
config = Config.from_file("config.yaml")

# Save to file
config.save("new_config.yaml")

# Validate
is_valid = config.validate() -> bool
```

#### `FiberProperties`
Data class for fiber properties.

```python
@dataclass
class FiberProperties:
    fiber_id: int
    length: float  # μm
    diameter: float  # μm
    volume: float  # μm³
    tortuosity: float
    orientation: float  # degrees
    # ... more properties
```

### Module Functions

#### Preprocessing
```python
from fiber_tracer.preprocessing import get_sorted_tiff_files, validate_images

# Get sorted file list
files = get_sorted_tiff_files(
    data_dir="path/to/data",
    sort_order="Name (Ascending)",
    num_images=100
)

# Validate images
valid, errors = validate_images(files)
```

#### Analysis
```python
from fiber_tracer.analysis import FiberAnalyzer

analyzer = FiberAnalyzer(config)
fibers = analyzer.analyze_fibers(labeled_volume)
stats = analyzer.calculate_statistics(fibers)
volume_fraction = analyzer.calculate_volume_fraction(binary_volume)
```

---

## ⚡ Performance

### Benchmarks

| Dataset Size | Images | Resolution | Processing Time | RAM Usage |
|-------------|--------|------------|-----------------|-----------|
| Small | 100 | 512×512 | ~5 min | 2 GB |
| Medium | 500 | 1024×1024 | ~30 min | 8 GB |
| Large | 1000 | 2048×2048 | ~2 hours | 16 GB |
| Extra Large | 2000 | 2048×2048 | ~4 hours | 32 GB |

*Benchmarks on Intel i7-9700K, 32GB RAM, SSD storage*

### Optimization Tips

1. **Memory Management**
   - Reduce `chunk_size` for limited RAM
   - Use `scale_factor` > 1 to downsample
   - Close other applications

2. **Speed Optimization**
   - Increase `num_workers` for more cores
   - Use SSD for data storage
   - Process subset first for testing

3. **Quality vs Speed**
   - Higher `scale_factor` = faster but less accurate
   - Smaller `chunk_size` = slower but less memory
   - Simpler segmentation = faster processing

---

## 🔧 Troubleshooting

### Common Issues

#### Installation Issues

**Problem**: `ModuleNotFoundError` for packages
```bash
# Solution: Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Problem**: Mayavi installation fails
```bash
# Solution: Use conda instead
conda install -c anaconda mayavi
```

#### Runtime Issues

**Problem**: Memory error during processing
- Reduce `chunk_size` (e.g., 50 → 25)
- Increase `scale_factor` (e.g., 1 → 2)
- Process fewer images with `num_images`

**Problem**: No fibers detected
- Check image quality and contrast
- Adjust `min_diameter` and `max_diameter`
- Modify `block_size` in segmentation

**Problem**: Slow processing
- Increase `num_workers`
- Use `scale_factor` for initial testing
- Ensure data is on SSD

### Warning Suppression

The application automatically suppresses harmless warnings like:
- Paramiko Blowfish deprecation
- Matplotlib backend warnings
- NumPy future warnings

Important warnings are still logged.

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black fiber_tracer/

# Type checking
mypy fiber_tracer/
```

### Reporting Issues
Please use the GitHub issue tracker to report bugs or request features.

---

## 📝 Citation

If you use this software in your research, please cite:

```bibtex
@software{fiber_tracer_2024,
  title = {Fiber Tracer: Advanced 3D Analysis Tool for Fiber-Reinforced Polymer Composites},
  author = {Mr Sweet},
  year = {2024},
  version = {2.0.0},
  url = {https://github.com/yourusername/fiber_tracer}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Open-source community for the excellent libraries
- Research community for algorithm development
- Contributors and users for feedback and improvements

---

## 📞 Contact

- **Author**: Mr Sweet
- **Email**: remember.me@example.com
- **GitHub**: [https://github.com/yourusername/fiber_tracer](https://github.com/yourusername/fiber_tracer)

---

*Last Updated: January 2025 | Version 2.0.0*
