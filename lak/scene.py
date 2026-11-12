"""The scene map: the fixed contract between perception and the controller."""
from dataclasses import dataclass
import numpy as np

OTHER, ROAD, KART, OBSTACLE = 0, 1, 2, 3
CLASS_NAMES = ["OTHER", "ROAD", "KART", "OBSTACLE"]
PALETTE = np.array([[40, 40, 48], [128, 64, 128], [0, 0, 230], [250, 170, 30]], np.uint8)


@dataclass
class SceneMap:
    classes: np.ndarray   # (H, W) uint8 in {OTHER, ROAD, KART, OBSTACLE}
    depth: np.ndarray     # (H, W) float32, metres along the optical axis

    def color(self):
        return PALETTE[self.classes]
