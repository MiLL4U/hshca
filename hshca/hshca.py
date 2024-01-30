from typing import Optional, Tuple

import numpy as np

from hshca.linkmethod import LinkageMethod
from hshca.metric import HCAMetric

from .hca import MultiDimensionalHCA


class HyperSpectralHCA(MultiDimensionalHCA):
    DEFAULT_PHYS_DIST_FACTOR = 1.0

    def __init__(self, data: np.ndarray,
                 method: type[LinkageMethod],
                 spectral_metric: type[HCAMetric],
                 phys_dist_factor: Optional[float] = None,
                 physical_scale: Optional[Tuple[float, ...]] = None,
                 show_progress: Optional[bool] = None) -> None:
        super().__init__(data, method, spectral_metric, show_progress)
        if phys_dist_factor is None:
            phys_dist_factor = self.DEFAULT_PHYS_DIST_FACTOR
            print("INFO: phys_dist_factor is not specified")
            print("INFO: using default value (1.0)")
        if physical_scale is None:
            physical_scale = tuple(1.0 for _ in range(data.ndim - 1))
            print("INFO: physical_scale is not specified")
            print("INFO: using default value:", physical_scale)

        self.__phys_factor = phys_dist_factor
        self.__phys_scale = physical_scale

        self.__init_cluster_coordinates()

    def __init_cluster_coordinates(self) -> None:
        indices = np.indices(self.map_shape)
        transposed = indices.transpose(
            [_ for _ in range(1, len(self.map_shape) + 1)] + [0])
        reshaped = transposed.reshape((self.data_num, len(self.map_shape)))
        scaled = reshaped * self.__phys_scale
        self.__cls_coords = [coord for coord in scaled]
