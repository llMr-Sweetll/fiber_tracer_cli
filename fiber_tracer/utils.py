"""
Utility functions and warning suppressors for the Fiber Tracer application.
"""

import warnings
import logging
import sys
from functools import wraps

logger = logging.getLogger(__name__)


def suppress_warnings():
    """
    Suppress known harmless warnings that don't affect functionality.
    """
    # Suppress Paramiko Blowfish deprecation warning
    warnings.filterwarnings("ignore", message=".*Blowfish has been deprecated.*")
    
    # Suppress matplotlib backend warnings
    warnings.filterwarnings("ignore", message=".*Matplotlib is currently using.*")
    
    # Suppress numpy future warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Suppress scikit-image deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="skimage")
    
    # Suppress scipy sparse matrix warnings
    warnings.filterwarnings("ignore", message=".*sparse matrix.*")
    
    # Log that warnings are being suppressed
    logger.debug("Harmless warnings suppressed. Important warnings will still be shown.")


def setup_matplotlib_backend():
    """
    Setup matplotlib backend for non-interactive environments.
    """
    import matplotlib
    
    # Use non-interactive backend if no display is available
    if not hasattr(sys, 'ps1'):  # Not in interactive mode
        try:
            import tkinter
        except ImportError:
            matplotlib.use('Agg')  # Use non-interactive backend
            logger.debug("Using Agg backend for matplotlib (no display available)")


def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Returns:
        tuple: (success, missing_packages)
    """
    required_packages = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'skimage': 'scikit-image',
        'cv2': 'opencv-python',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'seaborn': 'seaborn',
        'yaml': 'pyyaml',
        'tqdm': 'tqdm',
        'openpyxl': 'openpyxl'
    }
    
    optional_packages = {
        'plotly': 'plotly',
        'mayavi': 'mayavi'
    }
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_required.append(package)
    
    # Check optional packages
    for module, package in optional_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(package)
    
    if missing_required:
        logger.error(f"Missing required packages: {', '.join(missing_required)}")
        logger.error(f"Install with: pip install {' '.join(missing_required)}")
        return False, missing_required
    
    if missing_optional:
        logger.info(f"Optional packages not installed: {', '.join(missing_optional)}")
        logger.info("These provide additional features but are not required.")
    
    return True, []


def format_time(seconds):
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def format_bytes(bytes_value):
    """
    Format bytes into human-readable size string.
    
    Args:
        bytes_value: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def safe_divide(numerator, denominator, default=0):
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def ensure_list(value):
    """
    Ensure value is a list.
    
    Args:
        value: Value to convert to list
        
    Returns:
        List containing value or value itself if already a list
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def deprecated(message="This function is deprecated"):
    """
    Decorator to mark functions as deprecated.
    
    Args:
        message: Deprecation message
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {message}",
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


class ProgressLogger:
    """
    Context manager for logging progress of long-running operations.
    """
    
    def __init__(self, operation_name, logger=None):
        """
        Initialize progress logger.
        
        Args:
            operation_name: Name of the operation
            logger: Logger instance (uses module logger if None)
        """
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
    
    def __enter__(self):
        """Start timing the operation."""
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log completion or failure of operation."""
        import time
        elapsed = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                f"Completed: {self.operation_name} "
                f"(took {format_time(elapsed)})"
            )
        else:
            self.logger.error(
                f"Failed: {self.operation_name} "
                f"after {format_time(elapsed)}: {exc_val}"
            )
        
        return False  # Don't suppress exceptions


def validate_file_path(path, must_exist=True, create_dir=False):
    """
    Validate a file or directory path.
    
    Args:
        path: Path to validate
        must_exist: Whether the path must exist
        create_dir: Create directory if it doesn't exist
        
    Returns:
        tuple: (is_valid, error_message)
    """
    import os
    from pathlib import Path
    
    path = Path(path)
    
    if must_exist and not path.exists():
        if create_dir and not path.suffix:  # It's a directory
            try:
                path.mkdir(parents=True, exist_ok=True)
                return True, None
            except Exception as e:
                return False, f"Cannot create directory: {e}"
        return False, f"Path does not exist: {path}"
    
    if path.exists() and path.is_file() and not path.suffix:
        return False, f"Expected directory but found file: {path}"
    
    return True, None
