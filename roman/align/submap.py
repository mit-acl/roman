import numpy as np

from dataclasses import dataclass
from typing import List

from roman.object.object import Object

class Submap:
    id: int
    pose: np.array
    time: float
    objects: List[Object]