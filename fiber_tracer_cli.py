#!/usr/bin/env python3
"""
Fiber Tracing CLI Application for 3D GFRP Composites

This script processes a series of TIFF images representing cross-sectional slices of a Glass Fiber Reinforced Polymer (GFRP) composite specimen.
It reconstructs a continuous 3D volume of fibers, traces their paths with high precision, and extracts quantitative data including fiber length,
diameter, volume, orientation, tortuosity, and volume fraction. It also provides visualizations such as heat maps, histograms, and 3D renderings.

Author: Mr Sweet
Date: 15/09/2024

Usage:
    python fiber_tracer_cli.py --data_dir /path/to/Data_TIFF --output_dir /path/to/output --voxel_size 1.1

Requirements:
    - Python 3.6 or higher
    - NumPy
    - SciPy
    - scikit-image
    - tqdm
    - multiprocessing
    - OpenCV (cv2)
    - Matplotlib
    - Mayavi (for advanced 3D visualization)
"""

import os
import sys
import argparse 
import logging
import multiprocessing as mp
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, exposure, util
from scipy import ndimage
from tqdm import tqdm
import cv2
import csv  

# Optional: For 3D visualization
try:
    from mayavi import mlab
    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False


class FiberTracer:
    """
    Class to handle the fiber tracing process.
    """

    def __init__(self, args):
        self.args = args
        self.tiff_files = []
        self.volume_shape = None
        self.segmented_volume_files = []
        self.full_volume = None
        self.binary_volume = None
        self.labeled_volume = None
        self.fibers = []
        self.output_dir = args.output_dir
        self.voxel_size = args.voxel_size

    def get_tiff_files(self):
        """
        Retrieve and sort TIFF files according to the specified order.
        """
        data_dir = self.args.data_dir
        num_images = self.args.num_images
        sort_order = self.args.sort_order

        tiff_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                      if f.lower().endswith(('.tif', '.tiff'))]

        if not tiff_files:
            logging.error("No TIFF files found in the specified directory.")
            sys.exit(1)

        # Sort files according to the specified order
        if sort_order == 'Name (Ascending)':
            tiff_files.sort()
        elif sort_order == 'Name (Descending)':
            tiff_files.sort(reverse=True)
        elif sort_order == 'Modified Time (Newest First)':
            tiff_files.sort(key=os.path.getmtime, reverse=True)
        elif sort_order == 'Modified Time (Oldest First)':
            tiff_files.sort(key=os.path.getmtime)
        else:
            logging.warning(f"Unknown sort order '{sort_order}'. Using default.")

        if num_images is not None:
            tiff_files = tiff_files[:num_images]

        logging.info(f"Processing {len(tiff_files)} TIFF files.")
        self.tiff_files = tiff_files

    def load_image(self, file_path):
        """
        Load and preprocess a single image.
        """
        scale_factor = self.args.scale_factor
        try:
            logging.debug(f"Loading image: {file_path}")
            image = io.imread(file_path)

            # Downscale image if scale_factor > 1
            if scale_factor > 1:
                image = cv2.resize(image, (image.shape[1] // scale_factor, image.shape[0] // scale_factor),
                                   interpolation=cv2.INTER_AREA)

            # Convert to float
            image = util.img_as_float(image)

            # Normalize image to range [0, 1]
            image = exposure.rescale_intensity(image, out_range=(0.0, 1.0))

            # Apply median filtering to reduce noise while preserving edges
            image = filters.median(image, morphology.disk(2))

            # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
            image = exposure.equalize_adapthist(image, clip_limit=0.03)

            # Apply Gaussian filter for smoothing
            image = filters.gaussian(image, sigma=1)

            logging.debug(f"Image {file_path} loaded and preprocessed successfully")
            return image
        except Exception as e:
            logging.error(f"Error loading image {file_path}: {e}")
            return None

    def process_chunk(self, chunk_files):
        """
        Process a chunk of images.
        """
        images = []
        for file_path in chunk_files:
            image = self.load_image(file_path)
            if image is not None:
                images.append(image)
        if images:
            volume_chunk = np.stack(images, axis=0)
            return volume_chunk
        else:
            return None

    def process_images(self):
        """
        Process images in chunks and reconstruct the full volume.
        """
        chunk_size = self.args.chunk_size
        num_workers = self.args.num_workers

        total_files = len(self.tiff_files)
        chunk_indices = list(range(0, total_files, chunk_size))
        chunks = [self.tiff_files[i:i + chunk_size] for i in chunk_indices]

        logging.info("Processing images in chunks...")

        # Partial function for multiprocessing
        process_chunk_partial = partial(self.process_chunk)

        # Process chunks in parallel
        with mp.Pool(processes=num_workers) as pool:
            for volume_chunk in tqdm(pool.imap(process_chunk_partial, chunks), total=len(chunks), desc='Processing Chunks', unit='chunk'):
                if volume_chunk is not None:
                    # Save chunk to disk to manage memory
                    chunk_idx = len(self.segmented_volume_files)
                    chunk_file = os.path.join(self.output_dir, f'volume_chunk_{chunk_idx}.npy')
                    np.save(chunk_file, volume_chunk)
                    self.segmented_volume_files.append(chunk_file)

                    if self.volume_shape is None:
                        self.volume_shape = volume_chunk.shape[1:]  # Exclude chunk dimension

                    del volume_chunk  # Free memory
                else:
                    logging.warning("An empty volume chunk was returned and will be skipped.")

        if not self.segmented_volume_files:
            logging.error("No volume chunks were processed successfully.")
            sys.exit(1)

        # Load and concatenate chunks to form the full volume
        total_slices = sum([np.load(f).shape[0] for f in self.segmented_volume_files])
        full_volume_shape = (total_slices,) + self.volume_shape

        logging.info("Reconstructing the full volume...")

        self.full_volume = np.memmap(os.path.join(self.output_dir, 'full_volume.dat'),
                                     dtype='float32', mode='w+', shape=full_volume_shape)

        current_index = 0
        for chunk_file in self.segmented_volume_files:
            volume_chunk = np.load(chunk_file)
            num_slices = volume_chunk.shape[0]
            self.full_volume[current_index:current_index + num_slices] = volume_chunk
            current_index += num_slices
            del volume_chunk  # Free memory
            os.remove(chunk_file)  # Remove chunk file to save disk space
        self.full_volume.flush()

    def segment_volume(self):
        """
        Segment fibers in the 3D volume.
        """
        logging.info("Starting volume segmentation...")

        # Apply adaptive thresholding (local thresholding)
        block_size = 51  # Adjust based on image characteristics
        volume = util.img_as_float(self.full_volume)
        binary_volume = np.zeros_like(volume, dtype=bool)

        # Process slice by slice to save memory
        for i in tqdm(range(volume.shape[0]), desc='Segmenting Volume', unit='slice'):
            try:
                slice_img = volume[i]
                threshold = filters.threshold_local(slice_img, block_size)
                binary_slice = slice_img > threshold

                # Remove small objects (noise)
                min_size = 500  # Minimum size of fibers in pixels
                binary_slice = morphology.remove_small_objects(binary_slice, min_size=min_size)

                # Fill holes
                binary_slice = ndimage.binary_fill_holes(binary_slice)

                binary_volume[i] = binary_slice
            except Exception as e:
                logging.error(f"Error processing slice {i}: {e}")

        logging.info("Volume segmentation completed.")
        self.binary_volume = binary_volume

    def label_components(self):
        """
        Label connected components in the binary volume.
        """
        logging.info("Labeling connected components in the volume...")
        self.labeled_volume, num_features = ndimage.label(self.binary_volume)
        logging.info(f"Number of fibers detected: {num_features}")

    def analyze_fibers(self):
        """
        Analyze fiber properties in the labeled volume.
        """
        logging.info("Analyzing fibers...")
        min_diameter = self.args.min_diameter
        max_diameter = self.args.max_diameter
        voxel_size = self.voxel_size

        properties = measure.regionprops(self.labeled_volume)
        fibers = []
        for prop in tqdm(properties, desc='Analyzing Fibers', unit='fiber'):
            try:
                # Calculate fiber properties
                diameter = prop.equivalent_diameter * voxel_size
                if diameter < min_diameter or diameter > max_diameter:
                    continue  # Exclude fibers outside the diameter range

                # Fiber length along the major axis
                length = prop.major_axis_length * voxel_size

                # Orientation calculation using PCA
                coords = prop.coords * voxel_size  # Coordinates in micrometers
                if coords.shape[0] < 3:
                    # Not enough points to determine orientation
                    orientation_deg = 0.0
                else:
                    # Perform PCA on the coordinates to find the principal axis
                    mean_coords = coords.mean(axis=0)
                    centered_coords = coords - mean_coords
                    cov_matrix = np.cov(centered_coords, rowvar=False)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
                    # Calculate orientation angles in degrees
                    orientation_deg = np.degrees(np.arccos(np.abs(principal_axis[2])))

                # Calculate volume
                volume = prop.area * (voxel_size ** 3)

                # Calculate tortuosity
                if coords.shape[0] < 2:
                    tortuosity = 0.0
                else:
                    euclidean_distance = np.linalg.norm(coords[-1] - coords[0])
                    path_length = np.sum(np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1)))
                    tortuosity = path_length / euclidean_distance if euclidean_distance != 0 else 0.0

                fibers.append({
                    'Fiber ID': prop.label,
                    'Length (μm)': length,
                    'Diameter (μm)': diameter,
                    'Volume (μm³)': volume,
                    'Tortuosity': tortuosity,
                    'Orientation (degrees)': orientation_deg
                    # Add more properties as needed
                })
            except Exception as e:
                logging.error(f"Error analyzing fiber {prop.label}: {e}")

        logging.info(f"Fiber analysis completed. {len(fibers)} fibers found.")
        self.fibers = fibers

    def save_fiber_properties(self):
        """
        Save fiber properties to a CSV file.
        """
        output_csv = os.path.join(self.output_dir, 'fiber_properties.csv')
        fieldnames = ['Fiber ID', 'Length (μm)', 'Diameter (μm)', 'Volume (μm³)', 'Tortuosity', 'Orientation (degrees)']
        try:
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for fiber in self.fibers:
                    writer.writerow(fiber)
            logging.info(f"Fiber properties saved to {output_csv}")
        except Exception as e:
            logging.error(f"Error saving fiber properties: {e}")

    def calculate_volume_fraction(self): 
        """
        Calculate and save the fiber volume fraction.
        """
        try:
            total_fiber_volume = np.sum(self.binary_volume) * (self.voxel_size ** 3)
            total_volume = self.binary_volume.size * (self.voxel_size ** 3)
            volume_fraction = (total_fiber_volume / total_volume) * 100
            logging.info(f"Fiber Volume Fraction: {volume_fraction:.2f}%")

            # Save volume fraction to a text file
            with open(os.path.join(self.output_dir, 'volume_fraction.txt'), 'w') as vf_file:
                vf_file.write(f"Fiber Volume Fraction: {volume_fraction:.2f}%\n")
        except Exception as e:
            logging.error(f"Error calculating volume fraction: {e}")

    def generate_heatmap(self):
        """
        Generate a heat map for fiber lengths.
        """
        logging.info("Generating heat map for fiber lengths...")
        try:
            length_map = np.zeros_like(self.labeled_volume, dtype=float)
            for fiber in self.fibers:
                mask = self.labeled_volume == fiber['Fiber ID']
                length_map[mask] = fiber['Length (μm)']

            # Display a maximum intensity projection as the heat map
            mip = np.max(length_map, axis=0)
            plt.figure(figsize=(8, 6))
            plt.imshow(mip, cmap='hot')
            plt.colorbar(label='Fiber Length (μm)')
            plt.title('Heat Map of Fiber Lengths (Maximum Intensity Projection)')
            plt.savefig(os.path.join(self.output_dir, 'fiber_length_heatmap.png'))
            plt.close()
            logging.info("Heat map saved.")
        except Exception as e:
            logging.error(f"Error generating heat map: {e}")

    def generate_histogram(self):
        """
        Generate a histogram for fiber diameters.
        """
        logging.info("Generating histogram for fiber diameters...")
        try:
            diameters = [fiber['Diameter (μm)'] for fiber in self.fibers]
            plt.figure(figsize=(8, 6))
            plt.hist(diameters, bins=50, color='blue', edgecolor='black')
            plt.xlabel('Fiber Diameter (μm)')
            plt.ylabel('Frequency')
            plt.title('Histogram of Fiber Diameters')
            plt.savefig(os.path.join(self.output_dir, 'fiber_diameter_histogram.png'))
            plt.close()
            logging.info("Histogram saved.")
        except Exception as e:
            logging.error(f"Error generating histogram: {e}")

    def classify_fibers(self):
        """
        Classify fibers into bundles based on orientation.
        """
        logging.info("Classifying fibers based on orientation...")
        try:
            # Define angle bins (e.g., every 15 degrees)
            angle_bins = np.arange(0, 181, 15)
            fiber_angles = [fiber['Orientation (degrees)'] for fiber in self.fibers]
            fiber_classes = np.digitize(fiber_angles, angle_bins)

            # Update fibers with class labels
            for idx, fiber in enumerate(self.fibers):
                fiber['Class'] = int(fiber_classes[idx])

            # Save classification results
            output_csv = os.path.join(self.output_dir, 'fiber_classification.csv')
            fieldnames = ['Fiber ID', 'Class', 'Orientation (degrees)', 'Length (μm)', 'Diameter (μm)', 'Volume (μm³)', 'Tortuosity']
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for fiber in self.fibers:
                    writer.writerow({
                        'Fiber ID': fiber['Fiber ID'],
                        'Class': fiber['Class'],
                        'Orientation (degrees)': fiber['Orientation (degrees)'],
                        'Length (μm)': fiber['Length (μm)'],
                        'Diameter (μm)': fiber['Diameter (μm)'],
                        'Volume (μm³)': fiber['Volume (μm³)'],
                        'Tortuosity': fiber['Tortuosity']
                    })
            logging.info("Fiber classification completed and saved.")
        except Exception as e:
            logging.error(f"Error classifying fibers: {e}")

    def visualize_fibers(self):
        """
        Visualize fibers using 3D rendering (if Mayavi is available).
        """
        if not MAYAVI_AVAILABLE:
            logging.warning("Mayavi is not available. Skipping 3D visualization.")
            return

        logging.info("Generating 3D visualization of fibers...")
        try:
            # Generate a random color map for fibers
            num_labels = self.labeled_volume.max()
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1, num_labels + 1))

            # Create Mayavi figure
            mlab.figure(bgcolor=(1, 1, 1))

            # Render fibers
            src = mlab.pipeline.scalar_field(self.labeled_volume.astype(np.int32))
            mlab.pipeline.iso_surface(src, contours=num_labels, colormap='jet', opacity=0.5)

            # Add axes and labels
            mlab.axes(xlabel='X (pixels)', ylabel='Y (pixels)', zlabel='Z (slices)')
            mlab.colorbar(title='Fiber ID', orientation='vertical')

            # Save visualization
            mlab.view(azimuth=45, elevation=75, distance='auto')
            mlab.savefig(os.path.join(self.output_dir, 'fiber_visualization.png'))
            mlab.close()
            logging.info("3D visualization saved.")
        except Exception as e:
            logging.error(f"Error generating 3D visualization: {e}")

    def run(self):
        """
        Run the complete fiber tracing process.
        """
        try:
            self.get_tiff_files()
            self.process_images()
            self.segment_volume()
            self.label_components()
            self.analyze_fibers()
            self.save_fiber_properties()
            self.calculate_volume_fraction()
            self.generate_heatmap()
            self.generate_histogram()
            self.classify_fibers()
            self.visualize_fibers()
            logging.info("Fiber tracing completed successfully.")
        except Exception as e:
            logging.error(f"An error occurred during processing: {e}")
            sys.exit(1)


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        args: Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Fiber Tracing CLI Application for 3D GFRP Composites')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the directory containing TIFF images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output results')
    parser.add_argument('--voxel_size', type=float, default=1.1,
                        help='Voxel size in micrometers (default: 1.1)')
    parser.add_argument('--min_diameter', type=float, default=10.0,
                        help='Minimum fiber diameter to consider (μm)')
    parser.add_argument('--max_diameter', type=float, default=50.0,
                        help='Maximum fiber diameter to consider (μm)')
    parser.add_argument('--chunk_size', type=int, default=100,
                        help='Number of images to process per chunk')
    parser.add_argument('--scale_factor', type=int, default=1,
                        help='Factor to downscale images')
    parser.add_argument('--num_images', type=int, default=None,
                        help='Total number of images to process')
    parser.add_argument('--sort_order', type=str, default='Name (Ascending)',
                        choices=['Name (Ascending)', 'Name (Descending)', 'Modified Time (Newest First)', 'Modified Time (Oldest First)'],
                        help='Order in which to process images')
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count(),
                        help='Number of worker processes for parallel processing')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    args = parser.parse_args()
    return args


def setup_logging(log_level):
    """
    Set up the logging configuration.

    Args:
        log_level (str): Logging level as a string.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {log_level}")
        numeric_level = logging.INFO

    logging.basicConfig(level=numeric_level,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[
                            logging.StreamHandler(sys.stdout)
                        ])


def main():
    """
    Main function to run the fiber tracing application.
    """
    args = parse_arguments()
    setup_logging(args.log_level)

    logging.info("Starting Fiber Tracing CLI Application")

    os.makedirs(args.output_dir, exist_ok=True)

    tracer = FiberTracer(args)
    tracer.run()


if __name__ == '__main__':
    main()
