from __future__ import annotations

from typing import List

from numpy import ndarray
from copy import deepcopy


class Cluster:
    def __init__(self, all_vectors: ndarray, node_idxs: List[int]) -> None:
        self.__all_vectors = all_vectors
        self.__node_idxs = node_idxs
        self.__prev_size = self.size
        # self.__repr_point = ...
        # TODO: hold representing point to improve performance?

    @property
    def node_idxs(self) -> List[int]:
        return self.__node_idxs

    @property
    def size(self) -> int:
        return len(self.__node_idxs)

    @property
    def previous_size(self) -> int:
        return self.__prev_size

    @property
    def all_vectors(self) -> ndarray:
        return self.__all_vectors

    @property
    def member_vectors(self) -> ndarray:
        return self.__all_vectors[self.__node_idxs]

    def has_same_vectors(self, vectors: ndarray) -> bool:
        return True if vectors is self.__all_vectors else False

    def merge(self, other: Cluster) -> Cluster:
        """Merge the current cluster with another cluster.

        Args:
            other (Cluster): The cluster to merge with.

        Raises:
            ValueError: If the clusters being merged have different data.

        Returns:
            Cluster: The current cluster after merging.
        """
        if not other.has_same_vectors(self.__all_vectors):
            raise ValueError("attempted to merge clusters with different data")
        self.__prev_size = self.size
        self.__node_idxs.extend(other.node_idxs)
        return self

    def merged(self, other: Cluster) -> Cluster:
        """Create a new cluster by merging the current cluster with another
        cluster.

        Args:
            other (Cluster): The cluster to merge with.

        Raises:
            ValueError: If the clusters being merged have different data.

        Returns:
            Cluster: A new cluster resulting from the merge, leaving the current
                     and the other cluster unchanged.
        """
        res = Cluster(self.all_vectors, deepcopy(self.node_idxs))
        res.merge(other)
        return res
