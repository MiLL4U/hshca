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

    @abstractmethod
    def distance_matrix(self, vectors_1: np.ndarray,
                        vectors_2: Optional[np.ndarray] = None
                        ) -> np.ndarray:
        raise NotImplementedError


# FIXME: cannot instantiate this class directory
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
        vectors_1 = self.cast_to_2d_array(vectors_1)
        vectors_2 = self.cast_to_2d_array(vectors_2)
        return distance.cdist(vectors_1, vectors_2, metric=self.name)

    def cast_to_2d_array(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.ndim == 2:
            return vectors
        elif vectors.ndim == 1:
            return np.array([vectors])
        else:
            raise TypeError(f"invalid dimension ({vectors.ndim})")


class Euclidean(SciPySupportedMetric):
    def __init__(self) -> None:
        super().__init__('euclidean')


class CityBlock(SciPySupportedMetric):
    def __init__(self) -> None:
        super().__init__('cityblock')
