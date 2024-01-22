from typing import List, Tuple, Type, Union, cast

import numpy as np
from numpy import ndarray

from .cluster import Cluster
from .linkmethod import LinkageMethod
from .metric import HCAMetric


class HierarchicalClusterAnalysis:
    def __init__(self, data: ndarray,
                 method: Type[LinkageMethod],
                 metric: Type[HCAMetric]) -> None:
        self.__data = data
        self.__compute_dtype = data.dtype
        self.__metric = metric()
        self.__method = method(self.__metric)

        self.__init_internal_variables()

    @property
    def data(self) -> ndarray:
        return self.__data

    @property
    def data_num(self) -> int:
        return self.__data.shape[0]

    @property
    def linkage_num(self) -> int:
        return self.data_num - 1

    @property
    def distance_matrix(self) -> ndarray:
        return self.__dist_matrix

    def __init_internal_variables(self) -> None:
        # self.__repr_points = np.copy(self.__data)
        # self.__is_linkaged = np.zeros(self.data_num, dtype=bool)
        self.__clusters: List[Union[Cluster, None]] = [
            Cluster(self.data, [i]) for i in range(self.data_num)]
        self.__link_count = 0
        self.__cluster_idxs = np.arange(self.data_num)
        self.__linkage_hist = np.full(
            (self.linkage_num, 2), -1,  # REVIEW: is -1 appropriate for empty?
            dtype=int)
        self.__linkage_dist = np.full(
            self.linkage_num, np.inf,  # REVIEW: np.empty?
            dtype=self.__compute_dtype)

    def compute(self) -> None:
        self.__init_dist_matrix()

        for _ in range(self.linkage_num):
            pair_idx = self.__search_dist_argmin()
            self.__make_linkage(pair_idx)
            self.__update_dist_matrix(pair_idx)
            print(self.__dist_matrix)

    def __search_dist_argmin(self) -> Tuple[int, int]:
        # HACK: optimize search algorhythm
        res = np.unravel_index(
            np.argmin(self.__dist_matrix), self.__dist_matrix.shape)

        return cast(Tuple[int, int], res)

    def __make_linkage(self, pair_idx: Tuple[int, int]) -> None:
        # save history
        self.__linkage_hist[self.__link_count] = pair_idx
        self.__linkage_dist[self.__link_count] = self.__dist_matrix[pair_idx]

        # merge cluster
        cluster_1 = self.__clusters[pair_idx[0]]
        cluster_2 = self.__clusters[pair_idx[1]]
        if cluster_1 and cluster_2:
            cluster_1.merge(cluster_2)
        self.__clusters[pair_idx[1]] = None

        # update cluster index
        self.__cluster_idxs[pair_idx[1]] = pair_idx[0]

        # increment linkage count
        self.__link_count += 1

    def __update_dist_matrix(self, linked_pair: Tuple[int, int]) -> None:
        # fill row and column of linked node with inf
        self.__dist_matrix[linked_pair[1]] = np.inf
        self.__dist_matrix[:, linked_pair[1]] = np.inf

        # debug
        for row in self.__dist_matrix:
            print(row.round(1))

        # update distance matrix
        # TODO: implement here
        return

    def __init_dist_matrix(self) -> None:
        self.__dist_matrix = self.__metric.distance_matrix(self.__data)
        self.__dist_matrix[np.tril_indices(self.data_num)] = np.inf