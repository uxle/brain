"""
Brain Deep Learning Framework - Official Python Package
"""

try:
    from .brain_native import Tensor, Value, zeros, ones, tensor
    from . import nn
    from . import optim
except ImportError:
    # Development / pure mock fallback
    pass

__version__ = "1.0.0"
__all__ = ["Tensor", "Value", "zeros", "ones", "tensor", "nn", "optim"]
