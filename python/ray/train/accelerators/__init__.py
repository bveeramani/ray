from .base import Accelerator  # noqa: F401

try:
    import torch
    from .torch import TorchAccelerator  # noqa: F401
except ImportError:
    pass
