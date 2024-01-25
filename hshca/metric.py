from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import distance
from typing import Optional


class HCAMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def distance_matrix(self, vectors_1: np.ndarray,
                        vectors_2: Optional[np.ndarray] = None
                        ) -> np.ndarray:
        raise NotImplementedError


class Euclidean(HCAMetric):
    NAME = "euclidean"

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return self.NAME

    def distance_matrix(self, vectors_1: np.ndarray,
                        vectors_2: Optional[np.ndarray] = None) -> np.ndarray:
        if vectors_2 is None:
            vectors_2 = vectors_1
        return distance.cdist(vectors_1, vectors_2, metric=self.NAME)
