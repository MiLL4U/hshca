from typing import Optional, Tuple

import numpy as np
from scipy.spatial import distance

from hshca.linkmethod import LinkageMethod
from hshca.metric import HCAMetric

from .hca import MultiDimensionalHCA


class HyperSpectralHCA(MultiDimensionalHCA):
    PHYSICAL_METRIC = 'euclidean'
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
        self.__init_physical_distance_matrix()

    def __init_cluster_coordinates(self) -> None:
        indices = np.indices(self.map_shape)
        transposed = indices.transpose(
            [_ for _ in range(1, len(self.map_shape) + 1)] + [0])
        reshaped = transposed.reshape((self.data_num, len(self.map_shape)))
        self.__cls_coords = reshaped * self.__phys_scale

    def __init_physical_distance_matrix(self) -> None:
        res = distance.cdist(
            self.__cls_coords, self.__cls_coords, self.PHYSICAL_METRIC)
        res += np.diag(np.full(res.shape[0], np.inf))
        self.__phys_dist_matrix = res

    def compute(self) -> None:
        self.__update_physical_distance_matrix()
        self.__update_mixed_dist_matrix()
        super().compute()

    def print_dist_scales(self) -> None:
        print("min (spectral, spatial):")
        print("   ", np.min(self.dist_matrix), np.min(self.__phys_dist_matrix))

        d_mat_temp = np.copy(self.dist_matrix)
        d_mat_temp[d_mat_temp == np.inf] = 0
        pd_mat_temp = np.copy(self.__phys_dist_matrix)
        pd_mat_temp[pd_mat_temp == np.inf] = 0
        print("max (spectral, spatial):")
        print("   ", np.max(d_mat_temp), np.max(pd_mat_temp))

    def search_dist_argmin(self) -> Tuple[int, int]:
        return super().search_dist_argmin()

    def __update_mixed_dist_matrix(self) -> None:
        # spectral + physical * factor
        self.__mixed_dist_matrix = self.dist_matrix + \
            self.__phys_dist_matrix * self.__phys_factor

    def __update_physical_distance_matrix(self) -> None:
        pass
