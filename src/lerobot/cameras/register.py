# Import camera configuration classes so they register
# themselves with CameraConfig.

from .chesster.configuration_chesster import ChessterCameraConfig
from .opencv.configuration_opencv import OpenCVCameraConfig

__all__ = [
    "ChessterCameraConfig",
    "OpenCVCameraConfig",
]