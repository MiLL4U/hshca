from typing import Tuple

from numpy import ndarray

from hshca.linkmethod import LinkageMethod
from hshca.metric import HCAMetric

from .hca import MultiDimensionalHCA


class HyperSpectralHCA(MultiDimensionalHCA):
    def __init__(self, data: ndarray,
                 method: type[LinkageMethod],
                 spectral_metric: type[HCAMetric],
                 phys_dist_ratio: float,
                 physical_scale: Tuple[float, ...],
                 ) -> None:
        super().__init__(data, method, spectral_metric)
