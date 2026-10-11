from typing import Any

import chess
import cv2
from numpy.typing import NDArray

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCamera
from project.vision.camera_calibration.undistort import getDistortionMaps
from project.vision.detect_aruco import ArucoDetector
from project.vision.frame_augmentation import AugmentFrame

from .configuration_chesster import ChessterCameraConfig


class ChessterCamera(OpenCVCamera):
    def __init__(self, config: ChessterCameraConfig):
        super().__init__(config)

        self._map1, self._map2 = getDistortionMaps(4)
        self.arucoDetector = ArucoDetector()
        self.corners = None
        self._move = None
        self._frame_augmentation = None

    def set_move(self, move: chess.Move | None) -> None:
        self._move = move

    def get_move(self) -> chess.Move | None:
        return self._move

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        frame = super().read(color_mode)
        return self._transform(frame)

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        frame = super().async_read(timeout_ms)
        return self._transform(frame)

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        frame = super().read_latest(max_age_ms)
        return self._transform(frame)

    def _transform(self, frame: NDArray[Any]) -> NDArray[Any]:
        frame = cv2.remap(frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)

        corners, ids = self.arucoDetector.detect(frame)

        if corners is not None and len(corners) >= 4:
            #self.arucoDetector.draw_grid(frame, corners)
            self.corners = corners

        if self.corners is not None and self._move is not None:
            self._frame_augmentation = AugmentFrame(frame, self.corners)
            frame = self._frame_augmentation.highlight_move(self._move)

        return frame
