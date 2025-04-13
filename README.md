# Fiber Tracing CLI Application for 3D GFRP Composites

---

## **Table of Contents**

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start Guide](#quick-start-guide)
5. [Detailed Usage Instructions](#detailed-usage-instructions)
   - [Command-Line Arguments](#command-line-arguments)
   - [Parameter Descriptions and Impact](#parameter-descriptions-and-impact)
6. [Examples](#examples)
7. [Understanding the Output](#understanding-the-output)
8. [Explanation of How the Application Works](#explanation-of-how-the-application-works)
9. [Troubleshooting](#troubleshooting)
10. [Dependencies](#dependencies)
11. [Credits](#credits)
12. [License](#license)

---

## **Overview**

The **Fiber Tracing CLI Application** is a comprehensive tool designed to process high-resolution TIFF images of **Glass Fiber Reinforced Polymer (GFRP) composites** captured via X-ray computed tomography (CT). It reconstructs a continuous 3D volume of fibers, traces their paths with high precision, and extracts quantitative data including:

- **Fiber Length** (μm)
- **Diameter** (μm)
- **Volume** (μm³)
- **Orientation** (degrees)
- **Tortuosity**
- **Polar Angle**
- **Azimuthal Angle**
- **Fiber Volume Fraction** (%)
- **Fiber Classification Based on Orientation**

Additionally, the application provides visualizations such as:

- **Heat Map for Fiber Lengths**
- **Histogram for Fiber Diameters**
- **3D Visualization of Fibers**

This tool is optimized for accuracy and efficiency, handling large datasets (over 2000 images totaling approximately 15 GB) while considering potential challenges such as image noise, CT artifacts, and memory limitations.

---

## **Features**

- **Efficient Data Processing**: Handles large datasets by processing images in chunks and using memory-mapped arrays.
- **Advanced Preprocessing**: Applies noise reduction, contrast enhancement, and normalization to improve image quality.
- **Accurate Segmentation**: Uses adaptive thresholding and morphological operations tailored for fiber detection.
- **Parallel Processing**: Utilizes multiple CPU cores to speed up computation.
- **Customizable Parameters**: Allows users to adjust various parameters to optimize performance and accuracy.
- **Comprehensive Analysis**: Extracts detailed fiber properties and computes the fiber volume fraction.
- **Fiber Tracking**: Assigns unique identifiers to fibers and tracks them through the volume.
- **Fiber Classification**: Classifies fibers into bundles based on their orientation angles.
- **Visualizations**:
  - **Heat Map for Fiber Length**: Visual representation of fiber lengths across the volume.
  - **Histogram for Fiber Diameters**: Analyzes the distribution of fiber diameters.
  - **3D Visualization**: Generates 3D renderings of fibers (requires Mayavi).
- **Robust Error Handling**: Includes logging and exception handling to manage potential issues gracefully.
- **Modular Design**: Structured code with clear functions and classes for maintainability and extensibility.

---

## **Installation**

### **Prerequisites**

- **Python**: Version **3.6** or higher.
- **Operating System**: Compatible with **Windows**, **macOS**, and **Linux**.

### **Dependencies**

Ensure you have the required Python packages installed. Install them using the following command:

```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`:**

```txt
numpy
scipy
scikit-image
tqdm
opencv-python
matplotlib
mayavi
```

**Note**: Installing **Mayavi** can be challenging due to its dependencies. It's recommended to use **Anaconda** or **Miniconda** and install Mayavi via conda:

```bash
conda install -c anaconda mayavi
```

---

## **Quick Start Guide**

1. **Clone or Download the Repository (Note: This is a private repository)**:

   ```bash
   git clone https://github.com/yourusernam/llMr-Sweetll/fiber_tracer.git
   cd fiber_tracer
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Your Data**:

   - Ensure your TIFF images are in a directory (e.g., `/path/to/Data_TIFF`).
   - Images should be cross-sectional slices of the composite material.

4. **Run the Application**:

   ```bash
   python fiber_tracer_cli.py --data_dir /path/to/Data_TIFF --output_dir /path/to/output --voxel_size 1.1
   ```

5. **Review Outputs**:

   - **fiber_properties.csv**: Contains the calculated properties for each fiber.
   - **volume_fraction.txt**: Contains the calculated fiber volume fraction.
   - **fiber_length_heatmap.png**: Heat map of fiber lengths.
   - **fiber_diameter_histogram.png**: Histogram of fiber diameters.
   - **fiber_classification.csv**: Classification of fibers based on orientation.
   - **fiber_visualization.png**: 3D visualization of fibers (if Mayavi is installed).

---

## **Detailed Usage Instructions**

### **Running the Application**

Use the command line to run the script with the desired parameters:

```bash
python fiber_tracer_cli.py [OPTIONS]
```

### **Command-Line Arguments**

| Argument             | Type     | Default               | Description                                                                                                       |
|----------------------|----------|-----------------------|-------------------------------------------------------------------------------------------------------------------|
| `--data_dir`         | `str`    | **Required**          | Path to the directory containing TIFF images.                                                                     |
| `--output_dir`       | `str`    | **Required**          | Directory to save output results.                                                                                 |
| `--voxel_size`       | `float`  | `1.1`                 | Voxel size in micrometers.                                                                                        |
| `--min_diameter`     | `float`  | `10.0`                | Minimum fiber diameter to consider (μm).                                                                          |
| `--max_diameter`     | `float`  | `50.0`                | Maximum fiber diameter to consider (μm).                                                                          |
| `--chunk_size`       | `int`    | `100`                 | Number of images to process per chunk.                                                                            |
| `--scale_factor`     | `int`    | `1`                   | Factor to downscale images.                                                                                       |
| `--num_images`       | `int`    | `All images`          | Total number of images to process.                                                                                |
| `--sort_order`       | `str`    | `'Name (Ascending)'`  | Order in which to process images.                                                                                 |
| `--num_workers`      | `int`    | `Number of CPU cores` | Number of worker processes for parallel processing.                                                               |
| `--log_level`        | `str`    | `'INFO'`              | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).                                                  |
| `-h`, `--help`       |          |                       | Show help message and exit.                                                                                       |

### **Parameter Descriptions and Impact**


#### `--data_dir`

- **Description**: Specifies the directory containing the TIFF images to be processed.
- **Required**: Yes
- **Impact**: The source of the images for processing. Ensure that the directory contains valid TIFF images. Incorrect or missing images will prevent the application from running.

#### `--output_dir`

- **Description**: Specifies the directory where the output results will be saved.
- **Required**: Yes
- **Impact**: Output files, including fiber properties, visualizations, and logs, will be saved here. Ensure you have write permissions to this directory.

#### `--voxel_size`

- **Description**: Sets the voxel size in micrometers (μm).
- **Default**: `1.1`
- **Impact**: Used to scale measurements from pixels to physical units. Accurate voxel size is crucial for correct dimensional analysis. An incorrect voxel size will result in inaccurate physical measurements of fiber properties.

#### `--min_diameter` and `--max_diameter`

- **Description**: Defines the acceptable range of fiber diameters to consider during analysis.
- **Default**: `10.0` μm (min), `50.0` μm (max)
- **Impact**:
  - **Lower `min_diameter`**: Includes smaller fibers, but may also include noise or artifacts as fibers.
  - **Higher `max_diameter`**: Includes larger fibers, but may capture overlapping fibers or defects.
- **Recommendation**: Set based on the known diameter range of fibers in your composite material. Adjusting these values can help filter out noise or focus on fibers of specific sizes.

#### `--chunk_size`

- **Description**: Number of images to process in each chunk.
- **Default**: `100`
- **Impact**:
  - **Smaller `chunk_size`**: Reduces memory usage but increases the number of processing cycles, potentially increasing total processing time.
  - **Larger `chunk_size`**: Increases memory usage but may speed up processing due to reduced overhead.
- **Recommendation**: Adjust based on your system's available memory. Monitor memory usage during processing to prevent crashes.

#### `--scale_factor`

- **Description**: Factor to downscale images.
- **Default**: `1` (no downscaling)
- **Impact**:
  - **Higher `scale_factor`** (e.g., 2 or 3): Reduces image resolution, decreasing processing time and memory usage but may reduce accuracy in detecting smaller fibers.
  - **Lower `scale_factor`**: Preserves image detail but increases computational load and memory usage.
- **Recommendation**: Use `1` for maximum accuracy. Increase cautiously if encountering memory or performance issues.

#### `--num_images`

- **Description**: Specifies the total number of images to process.
- **Default**: All images in the directory.
- **Impact**:
  - **Fewer images**: Quicker processing but less data for analysis, which might not represent the entire sample.
  - **All images**: Comprehensive analysis but requires more time and resources.
- **Recommendation**: For testing or debugging, start with a smaller number of images.

#### `--sort_order`

- **Description**: Determines the order in which images are processed.
- **Default**: `'Name (Ascending)'`
- **Options**:
  - `'Name (Ascending)'`: Processes images sorted by filename in ascending order.
  - `'Name (Descending)'`: Processes images sorted by filename in descending order.
  - `'Modified Time (Newest First)'`: Processes images starting with the most recently modified.
  - `'Modified Time (Oldest First)'`: Processes images starting with the oldest modified.
- **Impact**:
  - **Correct sequence is crucial**: Ensure images are processed in the correct order to maintain spatial continuity in the 3D volume reconstruction.
- **Recommendation**: Use `'Name (Ascending)'` if images are named sequentially based on their position in the sample.

#### `--num_workers`

- **Description**: Number of worker processes for parallel processing.
- **Default**: Number of CPU cores available on your system.
- **Impact**:
  - **Higher `num_workers`**: Potentially faster processing but increases CPU usage and may lead to higher system load.
  - **Lower `num_workers`**: Reduces CPU load but may slow down processing.
- **Recommendation**: Use the default value for optimal performance unless system resources are limited.

#### `--log_level`

- **Description**: Sets the level of logging detail.
- **Default**: `'INFO'`
- **Options**: `'DEBUG'`, `'INFO'`, `'WARNING'`, `'ERROR'`, `'CRITICAL'`
- **Impact**:
  - **`DEBUG`**: Very detailed logs; useful for troubleshooting but may produce large log outputs.
  - **`INFO`**: General information about program execution.
  - **`WARNING`**, **`ERROR`**, **`CRITICAL`**: Increasingly severe messages, fewer details.
- **Recommendation**: Use `'DEBUG'` when diagnosing issues; otherwise, `'INFO'` is suitable for general use.


---

## **Examples**


### **Example 1: Processing All Images with Default Settings**

```bash
python fiber_tracer_cli.py \
    --data_dir /path/to/Data_TIFF \
    --output_dir /path/to/output \
    --voxel_size 1.1
```

- **Explanation**: Processes all images in `/path/to/Data_TIFF` using default parameters. Outputs are saved in `/path/to/output`.

### **Example 2: Processing the First 500 Images in Descending Order**

```bash
python fiber_tracer_cli.py \
    --data_dir /path/to/Data_TIFF \
    --output_dir /path/to/output \
    --voxel_size 1.1 \
    --num_images 500 \
    --sort_order 'Name (Descending)'
```

- **Explanation**: Processes the first 500 images sorted in descending filename order.

### **Example 3: Reducing Memory Usage by Downscaling Images**

```bash
python fiber_tracer_cli.py \
    --data_dir /path/to/Data_TIFF \
    --output_dir /path/to/output \
    --voxel_size 1.1 \
    --scale_factor 2 \
    --chunk_size 50 \
    --num_workers 2
```

- **Explanation**: Downscales images by a factor of 2, processes images in chunks of 50, and limits to 2 worker processes to reduce memory and CPU usage.

### **Example 4: Detailed Logging for Troubleshooting**

```bash
python fiber_tracer_cli.py \
    --data_dir /path/to/Data_TIFF \
    --output_dir /path/to/output \
    --voxel_size 1.1 \
    --log_level DEBUG
```

- **Explanation**: Sets the logging level to `DEBUG` to capture detailed logs for troubleshooting.

### **Example 5: Generating Visualizations and Classifications**

```bash
python fiber_tracer_cli.py \
    --data_dir /path/to/Data_TIFF \
    --output_dir /path/to/output \
    --voxel_size 1.1 \
    --num_workers 4
```

- **Explanation**: Processes all images, generates visualizations including heat maps, histograms, and 3D renderings (if Mayavi is installed), and saves outputs in the specified directory.


---

## **Understanding the Output**


After running the application, several output files will be generated in the `output_dir`:

1. **`fiber_properties.csv`**:
   - Contains detailed properties of each fiber, including:
     - Fiber ID
     - Length (μm)
     - Diameter (μm)
     - Volume (μm³)
     - Tortuosity
     - Orientation (degrees)

2. **`volume_fraction.txt`**:
   - Contains the calculated fiber volume fraction as a percentage of the total composite volume.

3. **`fiber_length_heatmap.png`**:
   - A heat map visualizing the spatial distribution of fiber lengths across the sample.

4. **`fiber_diameter_histogram.png`**:
   - A histogram showing the distribution of fiber diameters in the sample.

5. **`fiber_classification.csv`**:
   - Classification of fibers based on their orientation angles.

6. **`fiber_visualization.png`** (if Mayavi is installed):
   - A 3D rendering of the fibers in the sample.


---

## **Explanation of How the Application Works**

This section provides a detailed explanation of how the Fiber Tracing CLI Application processes the images, reconstructs the 3D volume, segments fibers, analyzes their properties, and generates visualizations.

### **1. Initialization**

The application begins by parsing command-line arguments and setting up logging based on the specified `--log_level`. It then creates an instance of the `FiberTracer` class, passing the parsed arguments.

### **2. Retrieving and Sorting TIFF Files**

- **Function**: `get_tiff_files()`
- **Process**:
  - Scans the specified `--data_dir` for TIFF images (`.tif` or `.tiff` extensions).
  - Sorts the files based on the `--sort_order` parameter (e.g., by name or modification time).
  - Limits the number of images to process based on `--num_images` (if specified).
- **Outcome**: A sorted list of image file paths to be processed.

### **3. Processing Images**

- **Function**: `process_images()`
- **Process**:
  - Divides the list of images into chunks based on `--chunk_size`.
  - Uses multiprocessing to process chunks in parallel (`--num_workers` controls the number of processes).
  - For each chunk:
    - **Function**: `process_chunk(chunk_files)`
    - **Process**:
      - Loads and preprocesses each image in the chunk:
        - **Function**: `load_image(file_path)`
        - **Preprocessing Steps**:
          - **Loading**: Reads the image using `skimage.io.imread`.
          - **Downscaling**: Resizes the image if `--scale_factor` > 1.
          - **Conversion to Float**: Ensures the image data is in floating-point format.
          - **Normalization**: Rescales intensity values to the range [0, 1].
          - **Noise Reduction**:
            - Applies median filtering to reduce noise while preserving edges.
            - Uses Gaussian filtering for smoothing.
          - **Contrast Enhancement**: Applies CLAHE to improve local contrast.
      - Stacks the preprocessed images to form a 3D volume chunk.
    - Saves the volume chunk to disk to manage memory efficiently.
- **Outcome**: A series of volume chunks saved as `.npy` files.

### **4. Reconstructing the Full Volume**

- **Function**: Part of `process_images()`
- **Process**:
  - Calculates the total number of slices by summing the slices in each chunk.
  - Creates a memory-mapped array (`numpy.memmap`) to represent the full 3D volume without loading it entirely into RAM.
  - Loads each volume chunk from disk and writes it into the appropriate location in the memory-mapped array.
  - Deletes the chunk from memory and disk after processing to free up resources.
- **Outcome**: A reconstructed 3D volume stored in a memory-mapped file.

### **5. Segmenting the Volume**

- **Function**: `segment_volume()`
- **Process**:
  - Converts the full volume to a floating-point format.
  - Initializes a binary volume to store segmentation results.
  - Processes each slice individually to conserve memory:
    - Applies adaptive (local) thresholding to account for intensity variations.
    - Binarizes the slice based on the threshold.
    - Removes small objects (noise) using morphological operations.
    - Fills holes in the binary objects to improve segmentation.
  - Updates the binary volume with the segmented slice.
- **Outcome**: A binary 3D volume where the fibers are represented as foreground (True) pixels.

### **6. Labeling Connected Components**

- **Function**: `label_components()`
- **Process**:
  - Uses `scipy.ndimage.label` to assign unique labels to connected components (fibers) in the binary volume.
- **Outcome**: A labeled 3D volume where each fiber has a unique identifier.

### **7. Analyzing Fibers**

- **Function**: `analyze_fibers()`
- **Process**:
  - Iterates over each labeled region (fiber) using `skimage.measure.regionprops`.
  - For each fiber:
    - **Diameter**: Calculates the equivalent diameter and scales it by `--voxel_size`.
      - Filters out fibers whose diameters are outside the range specified by `--min_diameter` and `--max_diameter`.
    - **Length**: Calculates the major axis length and scales it by `--voxel_size`.
    - **Orientation**:
      - Performs Principal Component Analysis (PCA) on the fiber's coordinates to find the principal axis.
      - Calculates the orientation angle with respect to the Z-axis (assuming fibers are aligned along Z).
    - **Volume**: Computes the fiber's volume based on the number of voxels and `--voxel_size`.
    - **Tortuosity**:
      - Calculates the path length by summing the distances between consecutive coordinates.
      - Computes the Euclidean distance between the fiber's endpoints.
      - Defines tortuosity as the ratio of the path length to the Euclidean distance.
    - Stores the calculated properties in a dictionary.
- **Outcome**: A list of dictionaries containing properties for each fiber.

### **8. Saving Fiber Properties**

- **Function**: `save_fiber_properties()`
- **Process**:
  - Writes the fiber properties to a CSV file (`fiber_properties.csv`) in the `--output_dir`.
- **Outcome**: A CSV file containing detailed fiber properties.

### **9. Calculating Fiber Volume Fraction**

- **Function**: `calculate_volume_fraction()`
- **Process**:
  - Calculates the total volume of fibers by summing the foreground pixels in the binary volume and scaling by `--voxel_size`.
  - Computes the total volume of the composite based on the size of the binary volume and `--voxel_size`.
  - Calculates the fiber volume fraction as a percentage.
- **Outcome**: A text file (`volume_fraction.txt`) containing the fiber volume fraction.

### **10. Generating Visualizations**

#### **a. Heat Map for Fiber Lengths**

- **Function**: `generate_heatmap()`
- **Process**:
  - Creates a 3D array (`length_map`) with the same shape as the labeled volume.
  - Assigns the length of each fiber to the corresponding voxels in `length_map`.
  - Generates a maximum intensity projection (MIP) of `length_map` along the Z-axis.
  - Uses Matplotlib to create and save a heat map image (`fiber_length_heatmap.png`).
- **Outcome**: A heat map image showing the spatial distribution of fiber lengths.

#### **b. Histogram for Fiber Diameters**

- **Function**: `generate_histogram()`
- **Process**:
  - Extracts the diameters of all fibers.
  - Uses Matplotlib to create and save a histogram (`fiber_diameter_histogram.png`).
- **Outcome**: A histogram image showing the distribution of fiber diameters.

#### **c. 3D Visualization of Fibers**

- **Function**: `visualize_fibers()`
- **Process**:
  - Checks if Mayavi is available; skips visualization if not.
  - Creates a scalar field from the labeled volume.
  - Uses Mayavi to render an isosurface for each fiber label, applying a colormap.
  - Adds axes, labels, and a color bar.
  - Saves the visualization as an image (`fiber_visualization.png`).
- **Outcome**: A 3D visualization image of the fibers.

### **11. Classifying Fibers Based on Orientation**

- **Function**: `classify_fibers()`
- **Process**:
  - Defines angle bins (e.g., every 15 degrees) to classify fibers.
  - Assigns each fiber to a class based on its orientation angle.
  - Adds the class label to each fiber's properties.
  - Saves the classification results to a CSV file (`fiber_classification.csv`).
- **Outcome**: A CSV file containing fibers classified by orientation.

### **12. Error Handling and Logging**

Throughout the application, try-except blocks and logging statements are used to handle exceptions gracefully and provide informative messages. This ensures that the application continues processing even if certain elements fail and that users are informed of any issues.

### **13. Finalization**

- After completing all processing steps, the application logs a success message.
- Any temporary files are cleaned up to free disk space.

---

## **Troubleshooting**


### **1. No TIFF Files Found**

- **Problem**: The script reports "No TIFF files found in the specified directory."
- **Solution**:
  - Ensure the `--data_dir` path is correct.
  - Verify that the directory contains TIFF images with `.tif` or `.tiff` extensions.

### **2. Memory Errors**

- **Problem**: The script crashes or slows down significantly due to memory issues.
- **Solution**:
  - Reduce the `--chunk_size` to process fewer images at a time.
  - Increase the `--scale_factor` to downscale images.
  - Limit the `--num_workers` to reduce concurrent memory usage.
  - Ensure sufficient disk space is available for temporary files.

### **3. Unexpected Results or Low Accuracy**

- **Problem**: The output data seems incorrect or lacks expected details.
- **Solution**:
  - Adjust the `--min_diameter` and `--max_diameter` to better match your fiber sizes.
  - Experiment with different `--scale_factor` values; downscaling too much can reduce accuracy.
  - Use the `--log_level DEBUG` to inspect detailed processing logs for anomalies.
  - Ensure the images are of good quality and not corrupted.

### **4. Slow Processing Speed**

- **Problem**: The script takes an excessively long time to complete.
- **Solution**:
  - Increase the `--num_workers` if CPU resources allow.
  - Increase the `--chunk_size` if memory resources allow.
  - Close other applications to free up system resources.
  - Consider downscaling images with a higher `--scale_factor`.

### **5. Errors During Execution**

- **Problem**: The script encounters an error and exits.
- **Solution**:
  - Check the console output and logs for error messages.
  - Use `--log_level DEBUG` for more detailed error information.
  - Ensure all dependencies are correctly installed (refer to the [Dependencies](#dependencies) section).
  - Verify that Mayavi is installed correctly if 3D visualization is required.

### **6. Mayavi Import Error**

- **Problem**: The script cannot import Mayavi for 3D visualization.
- **Solution**:
  - Install Mayavi using Anaconda or Miniconda:
    ```bash
    conda install -c anaconda mayavi
    ```
  - If Mayavi is not essential, the script will skip 3D visualization and continue.


---

## **Dependencies**


Ensure the following Python packages are installed:

- **NumPy**
- **SciPy**
- **scikit-image**
- **tqdm**
- **OpenCV (cv2)**
- **Matplotlib**
- **Mayavi** (optional, for 3D visualization)

Install all dependencies using:

```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`:**

```txt
numpy
scipy
scikit-image
tqdm
opencv-python
matplotlib
mayavi
```

**Note**: For Mayavi installation issues, consider using Anaconda or Miniconda.


---

## **Credits**

- **Author**: Mr Sweet
- **Date**: 15/09/2024
- **Contact**: remember me

**Acknowledgments**:

- This application was developed as a comprehensive solution for fiber tracing in 3D GFRP composites, considering advanced image processing techniques and efficient computational strategies.
- Special thanks to the open-source community for providing the powerful libraries used in this project.

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Disclaimer**: This software is provided "as is" without warranty of any kind. The author is not responsible for any damages or losses arising from the use of this software.

---

# **Additional Notes**

- **Understanding the Algorithms**:
  - **Adaptive Thresholding**: Used to segment fibers by applying a local threshold that adapts to variations in image intensity.
  - **Morphological Operations**: Remove small objects (noise) and fill holes to clean up the binary segmentation.
  - **Connected Component Labeling**: Identifies individual fibers by assigning unique labels to connected regions in the binary volume.
  - **Principal Component Analysis (PCA)**: Determines the main orientation of each fiber in 3D space.
  - **Tortuosity Calculation**: Measures how much a fiber deviates from a straight line, indicating its curvature.

- **Performance Considerations**:
  - **Chunk Processing**: Dividing images into chunks helps manage memory usage and allows for parallel processing.
  - **Memory Mapping**: Using `numpy.memmap` for the full volume avoids loading the entire dataset into RAM.
  - **Parallel Processing**: Utilizing multiple CPU cores speeds up computation but should be balanced against system resources.

- **Customizing the Application**:
  - **Adjustable Parameters**: Users can fine-tune parameters like `block_size`, `min_size`, and `angle_bins` to optimize segmentation and analysis for their specific data.
  - **Extensibility**: The modular design allows for adding new analysis methods or visualizations as needed.

- **Data Integrity**:
  - **Input Data**: Ensure that the input images are of high quality and correctly represent the cross-sectional slices of the composite.
  - **Output Data**: Review the output files and visualizations to verify that the results align with expectations.



- **Instructions**: 
  - This README provides step-by-step instructions and explanations of all parameters, their default values, and the impact of changing them. It is designed to be beginner-friendly and to guide users through the entire process of using the application.
- **Adjusting Parameters**: 
  - Users are encouraged to experiment with different parameter values to optimize the analysis for their specific data. The impact of each parameter is explained to aid in making informed decisions.
- **Extensibility**: 
  - The application is designed with modularity in mind, making it easier to extend or customize for additional features or specific research needs.
- **Data Backup**: 
  - Always back up your data before processing to prevent any accidental data loss.
- **Testing**: 
  - Before processing the entire dataset, test the application with a smaller subset of images to ensure it works correctly and to adjust parameters as needed.

---

If you have any questions, suggestions, or need further assistance, please feel free to contact the author.

Happy fiber tracing!
