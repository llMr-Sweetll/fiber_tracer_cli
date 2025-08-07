"""
Image preprocessing module for fiber tracer.
"""

import os
import logging
from typing import List, Optional, Tuple, Union
import numpy as np
from skimage import io, filters, morphology, exposure, util
import cv2
from tqdm import tqdm
import concurrent.futures

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handle image loading and preprocessing operations."""
    
    def __init__(self, config):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: ProcessingConfig object with preprocessing parameters
        """
        self.config = config
        self.median_disk_size = config.median_disk_size
        self.gaussian_sigma = config.gaussian_sigma
        self.clahe_clip_limit = config.clahe_clip_limit
        self.scale_factor = config.scale_factor
    
    def load_image(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess a single image.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Preprocessed image array or None if loading fails
        """
        try:
            logger.debug(f"Loading image: {file_path}")
            
            # Load image
            image = io.imread(file_path)
            
            # Handle multi-channel images
            if len(image.shape) > 2:
                # Convert to grayscale if RGB
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
                else:
                    # Take first channel
                    image = image[:, :, 0]
            
            # Downscale if needed
            if self.scale_factor > 1:
                new_height = image.shape[0] // self.scale_factor
                new_width = image.shape[1] // self.scale_factor
                image = cv2.resize(image, (new_width, new_height), 
                                  interpolation=cv2.INTER_AREA)
            
            # Convert to float
            image = util.img_as_float(image)
            
            # Apply preprocessing pipeline
            image = self._apply_preprocessing(image)
            
            logger.debug(f"Image {os.path.basename(file_path)} preprocessed successfully")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            return None
    
    def _apply_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to image.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image
        """
        # Normalize to [0, 1]
        image = exposure.rescale_intensity(image, out_range=(0.0, 1.0))
        
        # Noise reduction with median filter
        if self.median_disk_size > 0:
            image = filters.median(image, morphology.disk(self.median_disk_size))
        
        # Contrast enhancement with CLAHE
        if self.clahe_clip_limit > 0:
            image = exposure.equalize_adapthist(image, clip_limit=self.clahe_clip_limit)
        
        # Gaussian smoothing
        if self.gaussian_sigma > 0:
            image = filters.gaussian(image, sigma=self.gaussian_sigma)
        
        return image
    
    def process_batch(self, file_paths: List[str]) -> Optional[np.ndarray]:
        """
        Process a batch of images.
        
        Args:
            file_paths: List of image file paths
            
        Returns:
            Stacked volume of preprocessed images or None if all fail
        """
        images = []
        for file_path in file_paths:
            image = self.load_image(file_path)
            if image is not None:
                images.append(image)
        
        if images:
            return np.stack(images, axis=0)
        return None
    
    def process_batch_parallel(self, file_paths: List[str], 
                             num_workers: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Process a batch of images in parallel.
        
        Args:
            file_paths: List of image file paths
            num_workers: Number of parallel workers
            
        Returns:
            Stacked volume of preprocessed images or None if all fail
        """
        if num_workers is None:
            num_workers = self.config.num_workers
        
        images = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.load_image, fp): i 
                      for i, fp in enumerate(file_paths)}
            
            # Collect results in order
            results = [None] * len(file_paths)
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                result = future.result()
                if result is not None:
                    results[idx] = result
            
            # Filter out None values while preserving order
            images = [img for img in results if img is not None]
        
        if images:
            return np.stack(images, axis=0)
        return None


class VolumeBuilder:
    """Build 3D volumes from image sequences."""
    
    def __init__(self, output_dir: str):
        """
        Initialize volume builder.
        
        Args:
            output_dir: Directory for temporary files
        """
        self.output_dir = output_dir
        self.temp_files = []
    
    def build_volume_chunked(self, file_paths: List[str], 
                            preprocessor: ImagePreprocessor,
                            chunk_size: int = 100,
                            num_workers: int = 4) -> Tuple[np.memmap, Tuple[int, int, int]]:
        """
        Build volume by processing images in chunks.
        
        Args:
            file_paths: List of image file paths
            preprocessor: ImagePreprocessor instance
            chunk_size: Number of images per chunk
            num_workers: Number of parallel workers
            
        Returns:
            Memory-mapped volume and shape tuple
        """
        total_files = len(file_paths)
        chunk_indices = list(range(0, total_files, chunk_size))
        chunks = [file_paths[i:i + chunk_size] for i in chunk_indices]
        
        logger.info(f"Processing {total_files} images in {len(chunks)} chunks")
        
        # Process first chunk to get dimensions
        first_chunk = preprocessor.process_batch_parallel(chunks[0], num_workers)
        if first_chunk is None:
            raise ValueError("Failed to process first chunk of images")
        
        # Get volume shape
        slice_shape = first_chunk.shape[1:]
        total_slices = sum(len(chunk) for chunk in chunks)
        volume_shape = (total_slices,) + slice_shape
        
        # Create memory-mapped array
        volume_path = os.path.join(self.output_dir, 'volume.dat')
        volume = np.memmap(volume_path, dtype='float32', mode='w+', shape=volume_shape)
        self.temp_files.append(volume_path)
        
        # Process chunks and fill volume
        current_idx = 0
        for i, chunk_files in enumerate(tqdm(chunks, desc='Processing chunks')):
            if i == 0:
                # Use already processed first chunk
                chunk_data = first_chunk
            else:
                chunk_data = preprocessor.process_batch_parallel(chunk_files, num_workers)
            
            if chunk_data is not None:
                num_slices = chunk_data.shape[0]
                volume[current_idx:current_idx + num_slices] = chunk_data
                current_idx += num_slices
                del chunk_data  # Free memory
        
        volume.flush()
        return volume, volume_shape
    
    def cleanup(self):
        """Remove temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_file}: {e}")
        self.temp_files.clear()


def get_sorted_tiff_files(data_dir: str, 
                         sort_order: str = 'Name (Ascending)',
                         num_images: Optional[int] = None) -> List[str]:
    """
    Get sorted list of TIFF files from directory.
    
    Args:
        data_dir: Directory containing TIFF files
        sort_order: Sorting method
        num_images: Maximum number of images to return
        
    Returns:
        List of sorted file paths
    """
    # Find all TIFF files
    tiff_files = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.tif', '.tiff')):
            tiff_files.append(os.path.join(data_dir, filename))
    
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {data_dir}")
    
    # Sort files
    if sort_order == 'Name (Ascending)':
        tiff_files.sort()
    elif sort_order == 'Name (Descending)':
        tiff_files.sort(reverse=True)
    elif sort_order == 'Modified Time (Newest First)':
        tiff_files.sort(key=os.path.getmtime, reverse=True)
    elif sort_order == 'Modified Time (Oldest First)':
        tiff_files.sort(key=os.path.getmtime)
    else:
        logger.warning(f"Unknown sort order '{sort_order}'. Using default.")
        tiff_files.sort()
    
    # Limit number of files
    if num_images is not None and num_images > 0:
        tiff_files = tiff_files[:num_images]
    
    logger.info(f"Found {len(tiff_files)} TIFF files to process")
    return tiff_files


def validate_images(file_paths: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that images can be loaded and have consistent dimensions.
    
    Args:
        file_paths: List of image file paths
        
    Returns:
        Tuple of (success, list of error messages)
    """
    errors = []
    
    if not file_paths:
        return False, ["No image files provided"]
    
    # Check first few images for consistency
    check_count = min(5, len(file_paths))
    shapes = []
    
    for i in range(check_count):
        try:
            img = io.imread(file_paths[i])
            # Get 2D shape (ignore channels)
            shape = img.shape[:2] if len(img.shape) > 2 else img.shape
            shapes.append(shape)
        except Exception as e:
            errors.append(f"Cannot read {file_paths[i]}: {e}")
    
    if errors:
        return False, errors
    
    # Check consistency
    if len(set(shapes)) > 1:
        errors.append(f"Inconsistent image dimensions: {shapes}")
        return False, errors
    
    return True, []
