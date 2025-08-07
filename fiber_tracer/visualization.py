"""
Visualization module for fiber analysis results.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tqdm import tqdm

# Optional imports for advanced visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    
try:
    from mayavi import mlab
    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False

logger = logging.getLogger(__name__)


class FiberVisualizer:
    """Create visualizations for fiber analysis results."""
    
    def __init__(self, config):
        """
        Initialize visualizer with configuration.
        
        Args:
            config: VisualizationConfig object
        """
        self.config = config
        self.colormap = config.colormap
        self.figure_dpi = config.figure_dpi
        self.save_format = config.save_format
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def generate_all_visualizations(self, 
                                   fibers: List[Any],
                                   labeled_volume: Optional[np.ndarray],
                                   output_dir: str):
        """
        Generate all configured visualizations.
        
        Args:
            fibers: List of FiberProperties objects
            labeled_volume: Labeled 3D volume
            output_dir: Directory to save visualizations
        """
        logger.info("Generating visualizations")
        
        if self.config.generate_histogram:
            self.plot_fiber_histograms(fibers, output_dir)
        
        if self.config.generate_heatmap and labeled_volume is not None:
            self.plot_length_heatmap(fibers, labeled_volume, output_dir)
        
        if self.config.generate_3d_visualization:
            if self.config.use_plotly and PLOTLY_AVAILABLE:
                self.plot_3d_fibers_plotly(fibers, output_dir)
            elif self.config.use_mayavi and MAYAVI_AVAILABLE and labeled_volume is not None:
                self.plot_3d_fibers_mayavi(labeled_volume, output_dir)
            else:
                self.plot_3d_scatter(fibers, output_dir)
        
        # Additional visualizations
        self.plot_orientation_distribution(fibers, output_dir)
        self.plot_property_correlations(fibers, output_dir)
        self.create_summary_report(fibers, output_dir)
    
    def plot_fiber_histograms(self, fibers: List[Any], output_dir: str):
        """
        Create histograms for fiber properties.
        
        Args:
            fibers: List of FiberProperties objects
            output_dir: Directory to save plots
        """
        if not fibers:
            logger.warning("No fibers to visualize")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([f.to_dict() for f in fibers])
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Fiber Property Distributions', fontsize=16)
        
        # Length distribution
        axes[0, 0].hist(df['Length (μm)'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Length (μm)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Fiber Length Distribution')
        axes[0, 0].axvline(df['Length (μm)'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["Length (μm)"].mean():.1f}')
        axes[0, 0].legend()
        
        # Diameter distribution
        axes[0, 1].hist(df['Diameter (μm)'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Diameter (μm)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Fiber Diameter Distribution')
        axes[0, 1].axvline(df['Diameter (μm)'].mean(), color='red', linestyle='--',
                          label=f'Mean: {df["Diameter (μm)"].mean():.1f}')
        axes[0, 1].legend()
        
        # Orientation distribution
        axes[0, 2].hist(df['Orientation (degrees)'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 2].set_xlabel('Orientation (degrees)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Fiber Orientation Distribution')
        
        # Tortuosity distribution
        axes[1, 0].hist(df['Tortuosity'], bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Tortuosity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Fiber Tortuosity Distribution')
        
        # Aspect ratio distribution
        axes[1, 1].hist(df['Aspect Ratio'], bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Aspect Ratio')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Fiber Aspect Ratio Distribution')
        
        # Volume distribution (log scale)
        axes[1, 2].hist(np.log10(df['Volume (μm³)']), bins=30, edgecolor='black', alpha=0.7)
        axes[1, 2].set_xlabel('log10(Volume) (μm³)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Fiber Volume Distribution (log scale)')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'fiber_histograms.{self.save_format}')
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Histograms saved to {output_path}")
    
    def plot_length_heatmap(self, fibers: List[Any], 
                          labeled_volume: np.ndarray, 
                          output_dir: str):
        """
        Create heatmap of fiber lengths.
        
        Args:
            fibers: List of FiberProperties objects
            labeled_volume: Labeled 3D volume
            output_dir: Directory to save plot
        """
        logger.info("Generating fiber length heatmap")
        
        # Create length map
        length_map = np.zeros_like(labeled_volume, dtype=float)
        
        for fiber in fibers:
            mask = labeled_volume == fiber.fiber_id
            length_map[mask] = fiber.length
        
        # Maximum intensity projection
        mip_xy = np.max(length_map, axis=2)
        mip_xz = np.max(length_map, axis=1)
        mip_yz = np.max(length_map, axis=0)
        
        # Create figure with three projections
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Fiber Length Heatmap - Maximum Intensity Projections', fontsize=14)
        
        # XY projection
        im1 = axes[0].imshow(mip_xy.T, cmap=self.colormap, origin='lower')
        axes[0].set_title('XY Projection')
        axes[0].set_xlabel('X (pixels)')
        axes[0].set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=axes[0], label='Length (μm)')
        
        # XZ projection
        im2 = axes[1].imshow(mip_xz.T, cmap=self.colormap, origin='lower', aspect='auto')
        axes[1].set_title('XZ Projection')
        axes[1].set_xlabel('X (pixels)')
        axes[1].set_ylabel('Z (slices)')
        plt.colorbar(im2, ax=axes[1], label='Length (μm)')
        
        # YZ projection
        im3 = axes[2].imshow(mip_yz, cmap=self.colormap, origin='lower', aspect='auto')
        axes[2].set_title('YZ Projection')
        axes[2].set_xlabel('Y (pixels)')
        axes[2].set_ylabel('Z (slices)')
        plt.colorbar(im3, ax=axes[2], label='Length (μm)')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'fiber_length_heatmap.{self.save_format}')
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Heatmap saved to {output_path}")
    
    def plot_orientation_distribution(self, fibers: List[Any], output_dir: str):
        """
        Create polar plot of fiber orientations.
        
        Args:
            fibers: List of FiberProperties objects
            output_dir: Directory to save plot
        """
        if not fibers:
            return
        
        # Extract angles
        polar_angles = [f.polar_angle for f in fibers]
        azimuthal_angles = [f.azimuthal_angle for f in fibers]
        
        # Create polar plot
        fig = plt.figure(figsize=(12, 5))
        
        # Polar angle distribution
        ax1 = fig.add_subplot(121, projection='polar')
        theta = np.radians(polar_angles)
        r = np.ones_like(theta)
        
        # Create histogram in polar coordinates
        bins = np.linspace(0, np.pi, 37)
        hist, bin_edges = np.histogram(theta, bins=bins)
        width = bin_edges[1] - bin_edges[0]
        
        bars = ax1.bar(bin_edges[:-1], hist, width=width, bottom=0.0)
        ax1.set_title('Polar Angle Distribution')
        
        # Azimuthal angle distribution
        ax2 = fig.add_subplot(122, projection='polar')
        phi = np.radians(azimuthal_angles)
        
        bins = np.linspace(-np.pi, np.pi, 37)
        hist, bin_edges = np.histogram(phi, bins=bins)
        
        bars = ax2.bar(bin_edges[:-1], hist, width=width, bottom=0.0)
        ax2.set_title('Azimuthal Angle Distribution')
        
        plt.suptitle('Fiber Orientation Angular Distributions', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'orientation_distribution.{self.save_format}')
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Orientation distribution saved to {output_path}")
    
    def plot_property_correlations(self, fibers: List[Any], output_dir: str):
        """
        Create correlation matrix and scatter plots.
        
        Args:
            fibers: List of FiberProperties objects
            output_dir: Directory to save plot
        """
        if not fibers:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([f.to_dict() for f in fibers])
        
        # Select numeric columns
        numeric_cols = ['Length (μm)', 'Diameter (μm)', 'Volume (μm³)', 
                       'Tortuosity', 'Orientation (degrees)', 'Aspect Ratio']
        
        # Create correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Fiber Property Correlations', fontsize=14)
        
        # Correlation heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=axes[0, 0], cbar_kws={'label': 'Correlation'})
        axes[0, 0].set_title('Correlation Matrix')
        
        # Length vs Diameter
        axes[0, 1].scatter(df['Length (μm)'], df['Diameter (μm)'], 
                          alpha=0.5, s=20)
        axes[0, 1].set_xlabel('Length (μm)')
        axes[0, 1].set_ylabel('Diameter (μm)')
        axes[0, 1].set_title('Length vs Diameter')
        
        # Orientation vs Tortuosity
        axes[1, 0].scatter(df['Orientation (degrees)'], df['Tortuosity'], 
                          alpha=0.5, s=20)
        axes[1, 0].set_xlabel('Orientation (degrees)')
        axes[1, 0].set_ylabel('Tortuosity')
        axes[1, 0].set_title('Orientation vs Tortuosity')
        
        # Aspect Ratio vs Volume
        axes[1, 1].scatter(df['Aspect Ratio'], np.log10(df['Volume (μm³)']), 
                          alpha=0.5, s=20)
        axes[1, 1].set_xlabel('Aspect Ratio')
        axes[1, 1].set_ylabel('log10(Volume) (μm³)')
        axes[1, 1].set_title('Aspect Ratio vs Volume')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'property_correlations.{self.save_format}')
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Correlations saved to {output_path}")
    
    def plot_3d_scatter(self, fibers: List[Any], output_dir: str):
        """
        Create 3D scatter plot of fiber centroids.
        
        Args:
            fibers: List of FiberProperties objects
            output_dir: Directory to save plot
        """
        if not fibers:
            return
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract centroids and properties
        x = [f.centroid[0] for f in fibers]
        y = [f.centroid[1] for f in fibers]
        z = [f.centroid[2] for f in fibers]
        colors = [f.length for f in fibers]
        
        # Create scatter plot
        scatter = ax.scatter(x, y, z, c=colors, cmap=self.colormap, 
                           s=20, alpha=0.6)
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_zlabel('Z (μm)')
        ax.set_title('3D Fiber Distribution')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Fiber Length (μm)')
        
        # Save figure
        output_path = os.path.join(output_dir, f'fiber_3d_scatter.{self.save_format}')
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"3D scatter plot saved to {output_path}")
    
    def plot_3d_fibers_plotly(self, fibers: List[Any], output_dir: str):
        """
        Create interactive 3D visualization using Plotly.
        
        Args:
            fibers: List of FiberProperties objects
            output_dir: Directory to save plot
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, skipping interactive visualization")
            return
        
        if not fibers:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([f.to_dict() for f in fibers])
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=df['Centroid X (μm)'],
            y=df['Centroid Y (μm)'],
            z=df['Centroid Z (μm)'],
            mode='markers',
            marker=dict(
                size=3,
                color=df['Length (μm)'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Length (μm)"),
                opacity=0.8
            ),
            text=[f"ID: {row['Fiber ID']}<br>"
                  f"Length: {row['Length (μm)']:.1f}<br>"
                  f"Diameter: {row['Diameter (μm)']:.1f}<br>"
                  f"Orientation: {row['Orientation (degrees)']:.1f}"
                  for _, row in df.iterrows()],
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title='Interactive 3D Fiber Visualization',
            scene=dict(
                xaxis_title='X (μm)',
                yaxis_title='Y (μm)',
                zaxis_title='Z (μm)'
            ),
            width=900,
            height=700
        )
        
        # Save as HTML
        output_path = os.path.join(output_dir, 'fiber_3d_interactive.html')
        fig.write_html(output_path)
        
        logger.info(f"Interactive 3D plot saved to {output_path}")
    
    def plot_3d_fibers_mayavi(self, labeled_volume: np.ndarray, output_dir: str):
        """
        Create 3D visualization using Mayavi.
        
        Args:
            labeled_volume: Labeled 3D volume
            output_dir: Directory to save plot
        """
        if not MAYAVI_AVAILABLE:
            logger.warning("Mayavi not available, skipping 3D visualization")
            return
        
        try:
            mlab.figure(bgcolor=(1, 1, 1), size=(800, 600))
            
            # Create isosurface
            src = mlab.pipeline.scalar_field(labeled_volume.astype(np.float32))
            
            # Add contours for each fiber
            num_fibers = labeled_volume.max()
            mlab.pipeline.iso_surface(src, contours=list(range(1, min(num_fibers+1, 100))),
                                    colormap='jet', opacity=0.5)
            
            # Add axes and labels
            mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
            mlab.colorbar(title='Fiber ID', orientation='vertical')
            mlab.title('3D Fiber Visualization')
            
            # Save figure
            output_path = os.path.join(output_dir, f'fiber_3d_mayavi.{self.save_format}')
            mlab.savefig(output_path)
            mlab.close()
            
            logger.info(f"Mayavi 3D visualization saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating Mayavi visualization: {e}")
    
    def create_summary_report(self, fibers: List[Any], output_dir: str):
        """
        Create a summary report figure.
        
        Args:
            fibers: List of FiberProperties objects
            output_dir: Directory to save report
        """
        if not fibers:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([f.to_dict() for f in fibers])
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Fiber Analysis Summary Report', fontsize=16, fontweight='bold')
        
        # Statistics text
        ax_stats = fig.add_subplot(gs[0, :])
        ax_stats.axis('off')
        
        stats_text = f"""
        Total Fibers: {len(fibers)}
        
        Length: {df['Length (μm)'].mean():.1f} ± {df['Length (μm)'].std():.1f} μm
        Diameter: {df['Diameter (μm)'].mean():.1f} ± {df['Diameter (μm)'].std():.1f} μm
        Orientation: {df['Orientation (degrees)'].mean():.1f} ± {df['Orientation (degrees)'].std():.1f}°
        Tortuosity: {df['Tortuosity'].mean():.2f} ± {df['Tortuosity'].std():.2f}
        """
        
        ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
                     fontsize=12, ha='center', va='center',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        # Length distribution
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.hist(df['Length (μm)'], bins=20, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Length (μm)')
        ax1.set_ylabel('Count')
        ax1.set_title('Length Distribution')
        
        # Diameter distribution
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.hist(df['Diameter (μm)'], bins=20, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Diameter (μm)')
        ax2.set_ylabel('Count')
        ax2.set_title('Diameter Distribution')
        
        # Orientation distribution
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.hist(df['Orientation (degrees)'], bins=20, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Orientation (°)')
        ax3.set_ylabel('Count')
        ax3.set_title('Orientation Distribution')
        
        # Class distribution pie chart
        if 'Class' in df.columns:
            ax4 = fig.add_subplot(gs[2, 0])
            class_counts = df['Class'].value_counts()
            ax4.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
            ax4.set_title('Fiber Classification')
        
        # Length vs Diameter scatter
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.scatter(df['Length (μm)'], df['Diameter (μm)'], alpha=0.5, s=10)
        ax5.set_xlabel('Length (μm)')
        ax5.set_ylabel('Diameter (μm)')
        ax5.set_title('Length vs Diameter')
        
        # Box plot of key properties
        ax6 = fig.add_subplot(gs[2, 2])
        box_data = [df['Length (μm)']/df['Length (μm)'].max(),
                   df['Diameter (μm)']/df['Diameter (μm)'].max(),
                   df['Tortuosity']/df['Tortuosity'].max()]
        ax6.boxplot(box_data, labels=['Length', 'Diameter', 'Tortuosity'])
        ax6.set_ylabel('Normalized Value')
        ax6.set_title('Property Distributions')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'summary_report.{self.save_format}')
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary report saved to {output_path}")
