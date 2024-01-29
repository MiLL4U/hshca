from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import distance
from typing import Optional, Literal

SciPySupportedMetricsName = Literal[
    'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
    'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
    'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
    'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
    'sokalsneath', 'sqeuclidean', 'yule']


class HCAMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    # @abstractmethod
    # def vector_distance(self, vector_1: np.ndarray, vector_2: np.ndarray
    #                     ) -> float:
    #     raise NotImplementedError

    @abstractmethod
    def distance_matrix(self, vectors_1: np.ndarray,
                        vectors_2: Optional[np.ndarray] = None
                        ) -> np.ndarray:
        raise NotImplementedError


class SciPySupportedMetric(HCAMetric):
    def __init__(self, metric_name: SciPySupportedMetricsName) -> None:
        self.__name = metric_name

    @property
    def name(self) -> str:
        return self.__name

    def distance_matrix(self, vectors_1: np.ndarray,
                        vectors_2: Optional[np.ndarray] = None) -> np.ndarray:
        if vectors_2 is None:
            vectors_2 = vectors_1
        return distance.cdist(vectors_1, vectors_2, metric=self.name)


class Euclidean(SciPySupportedMetric):
    def __init__(self) -> None:
        super().__init__('euclidean')
