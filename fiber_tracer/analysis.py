"""
Fiber analysis module for extracting quantitative properties.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import ndimage, spatial
from skimage import measure
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class FiberProperties:
    """Data class for storing fiber properties."""
    fiber_id: int
    length: float  # micrometers
    diameter: float  # micrometers
    volume: float  # cubic micrometers
    tortuosity: float
    orientation: float  # degrees
    polar_angle: float  # degrees
    azimuthal_angle: float  # degrees
    centroid: Tuple[float, float, float]
    bbox: Tuple[int, int, int, int, int, int]
    surface_area: float  # square micrometers
    aspect_ratio: float
    fiber_class: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'Fiber ID': self.fiber_id,
            'Length (μm)': self.length,
            'Diameter (μm)': self.diameter,
            'Volume (μm³)': self.volume,
            'Surface Area (μm²)': self.surface_area,
            'Tortuosity': self.tortuosity,
            'Orientation (degrees)': self.orientation,
            'Polar Angle (degrees)': self.polar_angle,
            'Azimuthal Angle (degrees)': self.azimuthal_angle,
            'Aspect Ratio': self.aspect_ratio,
            'Centroid X (μm)': self.centroid[0],
            'Centroid Y (μm)': self.centroid[1],
            'Centroid Z (μm)': self.centroid[2],
            'Class': self.fiber_class
        }


class FiberAnalyzer:
    """Analyze fiber properties from labeled volumes."""
    
    def __init__(self, config):
        """
        Initialize analyzer with configuration.
        
        Args:
            config: FiberAnalysisConfig object
        """
        self.config = config
        self.voxel_size = config.voxel_size
        self.min_diameter = config.min_diameter
        self.max_diameter = config.max_diameter
        self.angle_bins = config.angle_bins
    
    def analyze_fibers(self, labeled_volume: np.ndarray,
                       binary_volume: Optional[np.ndarray] = None) -> List[FiberProperties]:
        """
        Analyze all fibers in labeled volume.
        
        Args:
            labeled_volume: Labeled 3D volume
            binary_volume: Optional binary mask
            
        Returns:
            List of FiberProperties objects
        """
        logger.info("Starting fiber analysis")
        
        # Get region properties
        regions = measure.regionprops(labeled_volume)
        
        fibers = []
        for region in tqdm(regions, desc='Analyzing fibers'):
            try:
                fiber_props = self._analyze_single_fiber(region)
                
                # Filter by diameter
                if (self.min_diameter <= fiber_props.diameter <= self.max_diameter):
                    fibers.append(fiber_props)
                    
            except Exception as e:
                logger.error(f"Error analyzing fiber {region.label}: {e}")
                continue
        
        logger.info(f"Analysis complete: {len(fibers)} fibers analyzed")
        
        # Classify fibers
        if self.config.calculate_orientation:
            fibers = self._classify_fibers(fibers)
        
        return fibers
    
    def _analyze_single_fiber(self, region) -> FiberProperties:
        """
        Analyze properties of a single fiber.
        
        Args:
            region: Region properties from skimage.measure
            
        Returns:
            FiberProperties object
        """
        # Basic properties
        fiber_id = region.label
        
        # Physical dimensions
        diameter = region.equivalent_diameter * self.voxel_size
        volume = region.area * (self.voxel_size ** 3)
        surface_area = region.surface_area * (self.voxel_size ** 2) if hasattr(region, 'surface_area') else 0
        
        # Fiber length (major axis)
        length = region.major_axis_length * self.voxel_size
        
        # Aspect ratio
        if region.minor_axis_length > 0:
            aspect_ratio = region.major_axis_length / region.minor_axis_length
        else:
            aspect_ratio = region.major_axis_length
        
        # Centroid in physical units
        centroid = tuple(c * self.voxel_size for c in region.centroid)
        
        # Bounding box
        bbox = region.bbox
        
        # Calculate orientation
        orientation, polar_angle, azimuthal_angle = self._calculate_orientation(region)
        
        # Calculate tortuosity
        tortuosity = self._calculate_tortuosity(region)
        
        return FiberProperties(
            fiber_id=fiber_id,
            length=length,
            diameter=diameter,
            volume=volume,
            tortuosity=tortuosity,
            orientation=orientation,
            polar_angle=polar_angle,
            azimuthal_angle=azimuthal_angle,
            centroid=centroid,
            bbox=bbox,
            surface_area=surface_area,
            aspect_ratio=aspect_ratio
        )
    
    def _calculate_orientation(self, region) -> Tuple[float, float, float]:
        """
        Calculate fiber orientation using PCA.
        
        Args:
            region: Region properties
            
        Returns:
            Tuple of (orientation, polar_angle, azimuthal_angle) in degrees
        """
        if not self.config.calculate_orientation:
            return 0.0, 0.0, 0.0
        
        coords = region.coords
        
        if coords.shape[0] < 3:
            return 0.0, 0.0, 0.0
        
        # Perform PCA
        mean_coords = coords.mean(axis=0)
        centered_coords = coords - mean_coords
        
        try:
            cov_matrix = np.cov(centered_coords, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Principal axis is the eigenvector with largest eigenvalue
            principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
            
            # Ensure positive z-component for consistency
            if principal_axis[2] < 0:
                principal_axis = -principal_axis
            
            # Calculate angles
            # Orientation with z-axis
            orientation = np.degrees(np.arccos(np.abs(principal_axis[2])))
            
            # Polar angle (angle from z-axis)
            polar_angle = np.degrees(np.arccos(principal_axis[2] / np.linalg.norm(principal_axis)))
            
            # Azimuthal angle (angle in xy-plane)
            azimuthal_angle = np.degrees(np.arctan2(principal_axis[1], principal_axis[0]))
            
        except:
            orientation = 0.0
            polar_angle = 0.0
            azimuthal_angle = 0.0
        
        return orientation, polar_angle, azimuthal_angle
    
    def _calculate_tortuosity(self, region) -> float:
        """
        Calculate fiber tortuosity.
        
        Args:
            region: Region properties
            
        Returns:
            Tortuosity value
        """
        if not self.config.calculate_tortuosity:
            return 1.0
        
        coords = region.coords * self.voxel_size
        
        if coords.shape[0] < 2:
            return 1.0
        
        try:
            # Calculate path length along fiber
            path_length = 0
            for i in range(1, len(coords)):
                path_length += np.linalg.norm(coords[i] - coords[i-1])
            
            # Calculate straight-line distance
            euclidean_distance = np.linalg.norm(coords[-1] - coords[0])
            
            if euclidean_distance > 0:
                tortuosity = path_length / euclidean_distance
            else:
                tortuosity = 1.0
                
        except:
            tortuosity = 1.0
        
        return tortuosity
    
    def _classify_fibers(self, fibers: List[FiberProperties]) -> List[FiberProperties]:
        """
        Classify fibers based on orientation.
        
        Args:
            fibers: List of FiberProperties
            
        Returns:
            Updated list with classification
        """
        for fiber in fibers:
            # Classify based on orientation angle
            angle = fiber.orientation
            
            if angle < 15:
                fiber.fiber_class = "Aligned (0-15°)"
            elif angle < 30:
                fiber.fiber_class = "Slightly Misaligned (15-30°)"
            elif angle < 45:
                fiber.fiber_class = "Moderately Misaligned (30-45°)"
            elif angle < 60:
                fiber.fiber_class = "Highly Misaligned (45-60°)"
            else:
                fiber.fiber_class = "Random (>60°)"
        
        return fibers
    
    def calculate_statistics(self, fibers: List[FiberProperties]) -> Dict[str, Any]:
        """
        Calculate statistical summary of fiber properties.
        
        Args:
            fibers: List of FiberProperties
            
        Returns:
            Dictionary with statistical summary
        """
        if not fibers:
            return {}
        
        # Convert to DataFrame for easy statistics
        df = pd.DataFrame([f.to_dict() for f in fibers])
        
        stats = {
            'total_fibers': len(fibers),
            'length_stats': {
                'mean': df['Length (μm)'].mean(),
                'std': df['Length (μm)'].std(),
                'min': df['Length (μm)'].min(),
                'max': df['Length (μm)'].max(),
                'median': df['Length (μm)'].median()
            },
            'diameter_stats': {
                'mean': df['Diameter (μm)'].mean(),
                'std': df['Diameter (μm)'].std(),
                'min': df['Diameter (μm)'].min(),
                'max': df['Diameter (μm)'].max(),
                'median': df['Diameter (μm)'].median()
            },
            'orientation_stats': {
                'mean': df['Orientation (degrees)'].mean(),
                'std': df['Orientation (degrees)'].std(),
                'min': df['Orientation (degrees)'].min(),
                'max': df['Orientation (degrees)'].max(),
                'median': df['Orientation (degrees)'].median()
            },
            'tortuosity_stats': {
                'mean': df['Tortuosity'].mean(),
                'std': df['Tortuosity'].std(),
                'min': df['Tortuosity'].min(),
                'max': df['Tortuosity'].max(),
                'median': df['Tortuosity'].median()
            }
        }
        
        # Add class distribution if available
        if 'Class' in df.columns:
            stats['class_distribution'] = df['Class'].value_counts().to_dict()
        
        return stats
    
    def calculate_volume_fraction(self, binary_volume: np.ndarray) -> float:
        """
        Calculate fiber volume fraction.
        
        Args:
            binary_volume: Binary segmentation mask
            
        Returns:
            Volume fraction as percentage
        """
        total_fiber_voxels = np.sum(binary_volume)
        total_voxels = binary_volume.size
        
        volume_fraction = (total_fiber_voxels / total_voxels) * 100
        
        logger.info(f"Fiber volume fraction: {volume_fraction:.2f}%")
        
        return volume_fraction
    
    def save_results(self, fibers: List[FiberProperties],
                    output_path: str,
                    format: str = 'csv'):
        """
        Save fiber analysis results.
        
        Args:
            fibers: List of FiberProperties
            output_path: Path to save file
            format: Output format ('csv', 'excel', 'json')
        """
        if not fibers:
            logger.warning("No fibers to save")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([f.to_dict() for f in fibers])
        
        # Save based on format
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'excel':
            df.to_excel(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Results saved to {output_path}")


class FiberConnectivityAnalyzer:
    """Analyze fiber connectivity and network properties."""
    
    def __init__(self, voxel_size: float = 1.0):
        """
        Initialize connectivity analyzer.
        
        Args:
            voxel_size: Voxel size in micrometers
        """
        self.voxel_size = voxel_size
    
    def analyze_connectivity(self, labeled_volume: np.ndarray,
                            distance_threshold: float = 10.0) -> Dict[str, Any]:
        """
        Analyze fiber-to-fiber connectivity.
        
        Args:
            labeled_volume: Labeled volume
            distance_threshold: Maximum distance for connectivity (micrometers)
            
        Returns:
            Dictionary with connectivity metrics
        """
        logger.info("Analyzing fiber connectivity")
        
        regions = measure.regionprops(labeled_volume)
        n_fibers = len(regions)
        
        if n_fibers == 0:
            return {}
        
        # Calculate centroids
        centroids = np.array([r.centroid for r in regions]) * self.voxel_size
        
        # Build distance matrix
        dist_matrix = spatial.distance_matrix(centroids, centroids)
        
        # Find connections
        connections = (dist_matrix < distance_threshold) & (dist_matrix > 0)
        
        # Calculate metrics
        connectivity_metrics = {
            'total_fibers': n_fibers,
            'total_connections': np.sum(connections) // 2,  # Divide by 2 for undirected
            'average_connections_per_fiber': np.mean(np.sum(connections, axis=1)),
            'max_connections': np.max(np.sum(connections, axis=1)),
            'min_connections': np.min(np.sum(connections, axis=1)),
            'connectivity_density': np.sum(connections) / (n_fibers * (n_fibers - 1)) if n_fibers > 1 else 0
        }
        
        # Find isolated fibers
        isolated = np.sum(connections, axis=1) == 0
        connectivity_metrics['isolated_fibers'] = np.sum(isolated)
        connectivity_metrics['isolation_ratio'] = np.sum(isolated) / n_fibers
        
        return connectivity_metrics
    
    def find_fiber_bundles(self, labeled_volume: np.ndarray,
                          proximity_threshold: float = 5.0) -> np.ndarray:
        """
        Identify fiber bundles based on proximity.
        
        Args:
            labeled_volume: Labeled volume
            proximity_threshold: Distance threshold for bundle membership
            
        Returns:
            Array with bundle labels
        """
        logger.info("Identifying fiber bundles")
        
        # Dilate each fiber to find overlaps
        bundle_volume = np.zeros_like(labeled_volume)
        
        unique_labels = np.unique(labeled_volume[labeled_volume > 0])
        
        for label in tqdm(unique_labels, desc='Finding bundles'):
            fiber_mask = labeled_volume == label
            
            # Dilate fiber
            struct_elem = ndimage.generate_binary_structure(3, 1)
            dilated = ndimage.binary_dilation(fiber_mask, 
                                             structure=struct_elem,
                                             iterations=int(proximity_threshold/self.voxel_size))
            
            # Find overlapping fibers
            overlapping_labels = np.unique(labeled_volume[dilated & (labeled_volume > 0)])
            
            # Assign to same bundle
            bundle_id = min(overlapping_labels)
            for overlap_label in overlapping_labels:
                bundle_volume[labeled_volume == overlap_label] = bundle_id
        
        return bundle_volume
