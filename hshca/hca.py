from typing import List, Optional, Tuple, Type, Union, cast

import numpy as np
from numpy import ndarray
from tqdm import tqdm

from .cluster import Cluster
from .linkmethod import LinkageMethod
from .metric import HCAMetric


class HierarchicalClusterAnalysis:
    DEFAULT_SHOW_PROGRESS = False

    def __init__(self, data: ndarray,
                 method: Type[LinkageMethod],
                 metric: Type[HCAMetric],
                 show_progress: Optional[bool] = None) -> None:
        self.__data = data
        self.__compute_dtype = data.dtype
        self.__metric = metric()
        self.__method = method(self.__metric)
        self.__show_progress = show_progress if show_progress \
            else self.DEFAULT_SHOW_PROGRESS

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

    @property
    def linkage_history(self) -> ndarray:
        return self.__linkage_hist

    @property
    def linkage_distances(self) -> ndarray:
        return self.__linkage_dist

    def __init_internal_variables(self) -> None:
        # self.__repr_points = np.copy(self.__data)
        # self.__is_linkaged = np.zeros(self.data_num, dtype=bool)
        self.__clusters: List[Union[Cluster, None]] = [
            Cluster(self.data, [i]) for i in range(self.data_num)]
        self.__link_count = 0
        self.__linkage_hist = np.full(
            (self.linkage_num, 2), -1, dtype=int)   # -1: empty
        self.__linkage_dist = np.full(
            self.linkage_num, np.inf, dtype=self.__compute_dtype)

    def compute(self) -> None:
        self.__init_dist_matrix()

        for _ in tqdm(range(self.linkage_num)):
            pair_idx = self.__search_dist_argmin()
            self.__make_linkage(pair_idx)
            self.__update_dist_matrix(pair_idx)

    def get_fcluster(self, max_cluster_num: int) -> ndarray:
        max_cluster_num = min(max_cluster_num, self.data_num)
        linkage_count = self.data_num - max_cluster_num
        linkages = self.linkage_history[:linkage_count]

        clusters = cast(List[Union[List[int], None]], [[i]
                        for i in range(self.data_num)])
        for base, addition in linkages:
            clusters[base].extend(clusters[addition])
            clusters[addition] = None
        flatten = [cluster for cluster in clusters if cluster]

        res = np.empty(self.data_num, dtype=np.int32)
        for cls_idx, cluster in enumerate(flatten):
            for data_idx in cluster:
                res[data_idx] = cls_idx

        return res

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

        # increment linkage count
        self.__link_count += 1

    def __update_dist_matrix(self, linked_pair: Tuple[int, int]) -> None:
        # get distances to new cluster
        new_cluster = self.__clusters[linked_pair[0]]
        if new_cluster is None:
            raise ValueError("new cluster not exist")
        new_dist = self.__method.cluster_distance_multi(
            new_cluster, self.__clusters)

        # update distance matrix
        self.__dist_matrix[:linked_pair[0], linked_pair[0]] \
            = new_dist[:linked_pair[0]]  # vertical
        self.__dist_matrix[linked_pair[0], linked_pair[0]:] \
            = new_dist[linked_pair[0]:]  # horizontal

        # fill row and column of linked node with inf
        self.__dist_matrix[linked_pair[1]] = np.inf
        self.__dist_matrix[:, linked_pair[1]] = np.inf
        self.__dist_matrix[linked_pair[0], linked_pair[0]] = np.inf

    def __init_dist_matrix(self) -> None:
        self.__dist_matrix = self.__metric.distance_matrix(self.__data)
        self.__dist_matrix[np.tril_indices(self.data_num)] = np.inf
