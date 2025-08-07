#!/usr/bin/env python3
"""
Fiber Tracing CLI Application for 3D GFRP/CFRP Composites
Version 2.0 - Refactored and Enhanced

This script provides a command-line interface to the fiber tracer package
for processing X-ray CT images of fiber-reinforced polymer composites.

Author: Mr Sweet
Contact: hegde.g.chandrashekhar@gmail.com
Date: 8/7/2025
"""

import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to path to import the package
sys.path.insert(0, str(Path(__file__).parent))

from fiber_tracer import FiberTracer, Config
from fiber_tracer.core import run_from_args
from fiber_tracer.ascii_art import animate_startup, show_completion, buzz_lightyear_mode


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Fiber Tracing CLI Application for 3D GFRP/CFRP Composites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage:
    python fiber_tracer_v2.py --data_dir /path/to/tiff --output_dir /path/to/output --voxel_size 1.1
  
  Process first 500 images:
    python fiber_tracer_v2.py --data_dir /path/to/tiff --output_dir /path/to/output --num_images 500
  
  Use configuration file:
    python fiber_tracer_v2.py --config config.yaml
  
  Advanced options:
    python fiber_tracer_v2.py --data_dir /path/to/tiff --output_dir /path/to/output \\
        --voxel_size 1.1 --min_diameter 5 --max_diameter 30 --chunk_size 50 \\
        --scale_factor 2 --num_workers 4 --log_level DEBUG
        """
    )
    
    # Required arguments (unless using config file)
    parser.add_argument('--data_dir', type=str,
                        help='Path to directory containing TIFF images')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save output results')
    
    # Configuration file option
    parser.add_argument('--config', type=str,
                        help='Path to configuration file (JSON or YAML)')
    
    # Physical parameters
    parser.add_argument('--voxel_size', type=float, default=1.1,
                        help='Voxel size in micrometers (default: 1.1)')
    parser.add_argument('--min_diameter', type=float, default=10.0,
                        help='Minimum fiber diameter to consider in μm (default: 10.0)')
    parser.add_argument('--max_diameter', type=float, default=50.0,
                        help='Maximum fiber diameter to consider in μm (default: 50.0)')
    
    # Processing parameters
    parser.add_argument('--chunk_size', type=int, default=100,
                        help='Number of images to process per chunk (default: 100)')
    parser.add_argument('--scale_factor', type=int, default=1,
                        help='Factor to downscale images (default: 1)')
    parser.add_argument('--num_images', type=int, default=None,
                        help='Total number of images to process (default: all)')
    parser.add_argument('--sort_order', type=str, default='Name (Ascending)',
                        choices=['Name (Ascending)', 'Name (Descending)', 
                                'Modified Time (Newest First)', 'Modified Time (Oldest First)'],
                        help='Order in which to process images')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: number of CPU cores)')
    
    # Logging
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    
    # Additional options
    parser.add_argument('--no_visualization', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--method', type=str, default='adaptive',
                        choices=['adaptive', 'otsu', 'watershed'],
                        help='Segmentation method (default: adaptive)')
    
    # Easter egg
    parser.add_argument('--buzz', action='store_true',
                        help=argparse.SUPPRESS)  # Hidden easter egg
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Check for Easter egg
    if hasattr(args, 'buzz') and args.buzz:
        buzz_lightyear_mode()
        sys.exit(0)
    
    # Show startup animation
    animate_startup()
    
    # Check if using configuration file
    if args.config:
        # Load configuration from file
        try:
            config = Config.from_file(args.config)
            print(f"Loaded configuration from {args.config}")
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            sys.exit(1)
        
        # Override with any command-line arguments if provided
        if args.data_dir:
            config.data_dir = args.data_dir
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.voxel_size != 1.1:  # Check if non-default
            config.analysis.voxel_size = args.voxel_size
        
        # Create and run tracer
        tracer = FiberTracer(config)
        success = tracer.run()
    else:
        # Check required arguments
        if not args.data_dir or not args.output_dir:
            print("Error: --data_dir and --output_dir are required unless using --config")
            print("Use -h or --help for usage information")
            sys.exit(1)
        
        # Run from command-line arguments
        success = run_from_args(args)
    
    # Show completion message
    show_completion(success=success)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
