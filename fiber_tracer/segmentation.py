"""
Segmentation module for fiber detection and labeling.
"""

import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy import ndimage
from skimage import filters, morphology, measure, segmentation
from tqdm import tqdm
import warnings

logger = logging.getLogger(__name__)


class FiberSegmenter:
    """Handle fiber segmentation in 3D volumes."""
    
    def __init__(self, config):
        """
        Initialize segmenter with configuration.
        
        Args:
            config: SegmentationConfig object
        """
        self.config = config
        self.block_size = config.block_size
        self.min_object_size = config.min_object_size
        self.adaptive_threshold = config.adaptive_threshold
        self.fill_holes = config.fill_holes
    
    def segment_volume(self, volume: np.ndarray, 
                      method: str = 'adaptive') -> np.ndarray:
        """
        Segment fibers in 3D volume.
        
        Args:
            volume: 3D volume array
            method: Segmentation method ('adaptive', 'otsu', 'watershed')
            
        Returns:
            Binary segmentation mask
        """
        logger.info(f"Starting volume segmentation using {method} method")
        
        if method == 'adaptive':
            return self._segment_adaptive(volume)
        elif method == 'otsu':
            return self._segment_otsu(volume)
        elif method == 'watershed':
            return self._segment_watershed(volume)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
    
    def _segment_adaptive(self, volume: np.ndarray) -> np.ndarray:
        """
        Segment using adaptive thresholding.
        
        Args:
            volume: 3D volume array
            
        Returns:
            Binary segmentation mask
        """
        binary_volume = np.zeros(volume.shape, dtype=bool)
        
        # Process slice by slice for memory efficiency
        for i in tqdm(range(volume.shape[0]), desc='Segmenting slices'):
            try:
                slice_img = volume[i]
                
                # Adaptive thresholding
                if self.adaptive_threshold:
                    threshold = filters.threshold_local(slice_img, self.block_size)
                    binary_slice = slice_img > threshold
                else:
                    # Global thresholding using Otsu
                    threshold = filters.threshold_otsu(slice_img)
                    binary_slice = slice_img > threshold
                
                # Morphological operations
                binary_slice = self._apply_morphology(binary_slice)
                
                binary_volume[i] = binary_slice
                
            except Exception as e:
                logger.error(f"Error segmenting slice {i}: {e}")
                binary_volume[i] = False
        
        return binary_volume
    
    def _segment_otsu(self, volume: np.ndarray) -> np.ndarray:
        """
        Segment using Otsu's method with multi-level thresholding.
        
        Args:
            volume: 3D volume array
            
        Returns:
            Binary segmentation mask
        """
        logger.info("Computing global Otsu threshold")
        
        # Sample volume for threshold calculation
        sample_size = min(50, volume.shape[0])
        sample_indices = np.linspace(0, volume.shape[0]-1, sample_size, dtype=int)
        sample_volume = volume[sample_indices]
        
        # Calculate threshold
        try:
            threshold = filters.threshold_otsu(sample_volume)
        except:
            threshold = np.median(sample_volume)
            logger.warning("Otsu thresholding failed, using median")
        
        # Apply threshold
        binary_volume = volume > threshold
        
        # Apply morphology slice by slice
        for i in tqdm(range(binary_volume.shape[0]), desc='Applying morphology'):
            binary_volume[i] = self._apply_morphology(binary_volume[i])
        
        return binary_volume
    
    def _segment_watershed(self, volume: np.ndarray) -> np.ndarray:
        """
        Segment using watershed algorithm.
        
        Args:
            volume: 3D volume array
            
        Returns:
            Binary segmentation mask
        """
        logger.info("Applying watershed segmentation")
        
        binary_volume = np.zeros(volume.shape, dtype=bool)
        
        for i in tqdm(range(volume.shape[0]), desc='Watershed segmentation'):
            try:
                slice_img = volume[i]
                
                # Initial segmentation
                threshold = filters.threshold_otsu(slice_img)
                binary = slice_img > threshold
                
                # Distance transform
                distance = ndimage.distance_transform_edt(binary)
                
                # Find markers
                local_maxima = morphology.local_maxima(distance)
                markers = measure.label(local_maxima)
                
                # Apply watershed
                labels = segmentation.watershed(-distance, markers, mask=binary)
                
                # Convert to binary
                binary_slice = labels > 0
                
                # Apply morphology
                binary_slice = self._apply_morphology(binary_slice)
                
                binary_volume[i] = binary_slice
                
            except Exception as e:
                logger.error(f"Error in watershed segmentation for slice {i}: {e}")
                binary_volume[i] = False
        
        return binary_volume
    
    def _apply_morphology(self, binary_slice: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up segmentation.
        
        Args:
            binary_slice: Binary image slice
            
        Returns:
            Cleaned binary slice
        """
        # Remove small objects
        if self.min_object_size > 0:
            binary_slice = morphology.remove_small_objects(
                binary_slice, min_size=self.min_object_size
            )
        
        # Fill holes
        if self.fill_holes:
            binary_slice = ndimage.binary_fill_holes(binary_slice)
        
        # Opening to remove small connections
        binary_slice = morphology.binary_opening(binary_slice)
        
        # Closing to connect nearby regions
        binary_slice = morphology.binary_closing(binary_slice)
        
        return binary_slice
    
    def label_fibers(self, binary_volume: np.ndarray,
                     connectivity: int = 26) -> Tuple[np.ndarray, int]:
        """
        Label connected components in binary volume.
        
        Args:
            binary_volume: Binary segmentation mask
            connectivity: Connectivity for labeling (6, 18, or 26)
            
        Returns:
            Labeled volume and number of fibers
        """
        logger.info("Labeling connected components")
        
        # Define connectivity structure
        if connectivity == 6:
            structure = ndimage.generate_binary_structure(3, 1)
        elif connectivity == 18:
            structure = ndimage.generate_binary_structure(3, 2)
        elif connectivity == 26:
            structure = ndimage.generate_binary_structure(3, 3)
        else:
            raise ValueError(f"Invalid connectivity: {connectivity}")
        
        # Label components
        labeled_volume, num_features = ndimage.label(binary_volume, structure=structure)
        
        logger.info(f"Found {num_features} connected components")
        
        # Filter by size
        if self.min_object_size > 0:
            labeled_volume = self._filter_small_components(
                labeled_volume, self.min_object_size
            )
            # Recount features
            unique_labels = np.unique(labeled_volume[labeled_volume > 0])
            num_features = len(unique_labels)
            logger.info(f"After filtering: {num_features} fibers remain")
        
        return labeled_volume, num_features
    
    def _filter_small_components(self, labeled_volume: np.ndarray,
                                min_size: int) -> np.ndarray:
        """
        Remove small connected components.
        
        Args:
            labeled_volume: Labeled volume
            min_size: Minimum component size in voxels
            
        Returns:
            Filtered labeled volume
        """
        # Count voxels for each label
        unique_labels, counts = np.unique(labeled_volume, return_counts=True)
        
        # Find small components
        small_labels = unique_labels[counts < min_size]
        
        # Remove small components
        for label in small_labels:
            if label != 0:  # Don't remove background
                labeled_volume[labeled_volume == label] = 0
        
        return labeled_volume
    
    def refine_segmentation(self, binary_volume: np.ndarray,
                          original_volume: np.ndarray) -> np.ndarray:
        """
        Refine segmentation using additional criteria.
        
        Args:
            binary_volume: Initial binary segmentation
            original_volume: Original grayscale volume
            
        Returns:
            Refined binary segmentation
        """
        logger.info("Refining segmentation")
        
        # Edge-based refinement
        for i in range(binary_volume.shape[0]):
            if np.any(binary_volume[i]):
                # Compute edges
                edges = filters.sobel(original_volume[i])
                
                # Use edges to refine boundaries
                edge_mask = edges > np.percentile(edges, 75)
                
                # Combine with original segmentation
                binary_volume[i] = binary_volume[i] & ~edge_mask
        
        return binary_volume


class AdvancedSegmenter:
    """Advanced segmentation methods for improved accuracy."""
    
    def __init__(self, config):
        """
        Initialize advanced segmenter.
        
        Args:
            config: SegmentationConfig object
        """
        self.config = config
    
    def segment_multiscale(self, volume: np.ndarray,
                          scales: Tuple[float, ...] = (1, 2, 4)) -> np.ndarray:
        """
        Multi-scale segmentation for better fiber detection.
        
        Args:
            volume: 3D volume array
            scales: Tuple of scale factors
            
        Returns:
            Binary segmentation mask
        """
        logger.info(f"Performing multi-scale segmentation with scales {scales}")
        
        segmentations = []
        
        for scale in scales:
            # Apply Gaussian filter at different scales
            smoothed = filters.gaussian(volume, sigma=scale)
            
            # Segment at this scale
            binary = smoothed > filters.threshold_otsu(smoothed)
            segmentations.append(binary)
        
        # Combine segmentations
        combined = np.sum(segmentations, axis=0)
        
        # Majority voting
        threshold = len(scales) // 2 + 1
        final_segmentation = combined >= threshold
        
        return final_segmentation
    
    def segment_active_contours(self, volume: np.ndarray,
                               initial_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Use active contours for segmentation refinement.
        
        Args:
            volume: 3D volume array
            initial_mask: Initial segmentation mask
            
        Returns:
            Refined segmentation mask
        """
        logger.info("Applying active contours segmentation")
        
        if initial_mask is None:
            # Create initial mask using Otsu
            initial_mask = volume > filters.threshold_otsu(volume)
        
        refined_mask = np.zeros_like(initial_mask)
        
        # Process slice by slice
        for i in tqdm(range(volume.shape[0]), desc='Active contours'):
            try:
                # Skip empty slices
                if not np.any(initial_mask[i]):
                    continue
                
                # Apply active contours
                snake = segmentation.active_contour(
                    volume[i],
                    initial_mask[i].astype(float),
                    alpha=0.015,
                    beta=10,
                    gamma=0.001
                )
                
                refined_mask[i] = snake > 0.5
                
            except Exception as e:
                logger.warning(f"Active contours failed for slice {i}: {e}")
                refined_mask[i] = initial_mask[i]
        
        return refined_mask


def track_fibers_across_slices(labeled_volume: np.ndarray,
                              overlap_threshold: float = 0.5) -> np.ndarray:
    """
    Track fibers across slices to ensure continuity.
    
    Args:
        labeled_volume: Labeled volume with disconnected components
        overlap_threshold: Minimum overlap ratio to connect components
        
    Returns:
        Relabeled volume with connected fibers
    """
    logger.info("Tracking fibers across slices")
    
    relabeled = np.zeros_like(labeled_volume)
    current_label = 1
    label_mapping = {}
    
    for z in tqdm(range(labeled_volume.shape[0]), desc='Tracking fibers'):
        slice_labels = labeled_volume[z]
        
        if z == 0:
            # First slice: direct copy
            unique_labels = np.unique(slice_labels[slice_labels > 0])
            for old_label in unique_labels:
                label_mapping[old_label] = current_label
                relabeled[z][slice_labels == old_label] = current_label
                current_label += 1
        else:
            # Track from previous slice
            prev_slice = relabeled[z-1]
            unique_labels = np.unique(slice_labels[slice_labels > 0])
            
            for old_label in unique_labels:
                current_mask = slice_labels == old_label
                
                # Find overlapping labels in previous slice
                overlap_labels = prev_slice[current_mask]
                overlap_labels = overlap_labels[overlap_labels > 0]
                
                if len(overlap_labels) > 0:
                    # Find most common overlapping label
                    unique_overlaps, counts = np.unique(overlap_labels, return_counts=True)
                    best_match = unique_overlaps[np.argmax(counts)]
                    
                    # Check overlap ratio
                    overlap_ratio = np.max(counts) / np.sum(current_mask)
                    
                    if overlap_ratio >= overlap_threshold:
                        # Continue the same fiber
                        relabeled[z][current_mask] = best_match
                    else:
                        # New fiber
                        relabeled[z][current_mask] = current_label
                        current_label += 1
                else:
                    # New fiber
                    relabeled[z][current_mask] = current_label
                    current_label += 1
    
    logger.info(f"Tracking complete: {current_label-1} continuous fibers identified")
    return relabeled
