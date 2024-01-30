from typing import List, Optional, Tuple, cast

import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

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
        self.__init_physical_dist_matrix()
        self.__update_mixed_dist_matrix()

    def __init_cluster_coordinates(self) -> None:
        indices = np.indices(self.map_shape)
        transposed = indices.transpose(
            [_ for _ in range(1, len(self.map_shape) + 1)] + [0])
        reshaped = transposed.reshape((self.data_num, len(self.map_shape)))
        res = reshaped * self.__phys_scale
        self.__cls_coords = cast(
            List[List[np.ndarray]], [[row] for row in res])

    def __init_physical_dist_matrix(self) -> None:
        coords = np.array(self.__cls_coords).reshape(
            (self.data_num, len(self.map_shape)))
        res = distance.cdist(coords, coords, self.PHYSICAL_METRIC)
        res[np.tril_indices(self.data_num)] = np.inf
        self.__phys_dist_matrix = res

    def spatial_centroids(self) -> np.ndarray:
        res = [np.average(np.array(coords), axis=0)
               if coords else np.full(len(self.map_shape), np.inf)
               for coords in self.__cls_coords]
        return np.array(res)

    def compute(self) -> None:
        itr = tqdm(range(self.linkage_num)) if self.show_proress_enabled \
            else range(self.linkage_num)
        for _ in itr:
            self.pair_idx = self.search_dist_argmin()
            self.make_linkage(self.pair_idx)
            self.update_dist_matrix(self.pair_idx)

            self.__update_cls_coords(self.pair_idx)
            self.__update_physical_dist_matrix()
            self.__update_mixed_dist_matrix()

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
        res = np.unravel_index(
            np.argmin(self.__mixed_dist_matrix), self.__mixed_dist_matrix.shape)
        return cast(Tuple[int, int], res)

    def __update_physical_dist_matrix(self) -> None:
        # HACK: decrease update frequency
        spatial_centroids = self.spatial_centroids()
        res = distance.cdist(
            spatial_centroids, spatial_centroids, self.PHYSICAL_METRIC)
        res[np.tril_indices(self.data_num)] = np.inf
        self.__phys_dist_matrix = res
        self.__phys_dist_matrix[np.isnan(self.__phys_dist_matrix)] = np.inf

    def __update_cls_coords(self, linked_pair: Tuple[int, int]) -> None:
        coord_1 = np.average(self.__cls_coords[linked_pair[0]], axis=0)
        coord_2 = np.average(
            self.__cls_coords[linked_pair[1]], axis=0)  # deleted cluster

        self.__cls_coords[linked_pair[0]].append((coord_1 + coord_2) / 2)
        # delete coordinates of deleted cluster
        self.__cls_coords[linked_pair[1]] = []

    def __update_mixed_dist_matrix(self) -> None:
        # spectral + physical * factor
        self.__mixed_dist_matrix = self.dist_matrix + \
            self.__phys_dist_matrix * self.__phys_factor
