from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


@CameraConfig.register_subclass("chesster")
@dataclass(kw_only=True)
class ChessterCameraConfig(OpenCVCameraConfig):
    pass