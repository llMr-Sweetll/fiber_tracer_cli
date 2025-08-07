"""
Fiber Tracer Package for 3D GFRP/CFRP Composites Analysis

A comprehensive tool for processing X-ray CT images of fiber-reinforced polymers,
reconstructing 3D volumes, and extracting quantitative fiber properties.
"""

__version__ = "2.0.0"
__author__ = "Mr Sweet"
__email__ = "remember me"

from .core import FiberTracer
from .config import Config

__all__ = ['FiberTracer', 'Config']
