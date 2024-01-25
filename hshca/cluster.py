from __future__ import annotations

from typing import List

from numpy import ndarray


class Cluster:
    def __init__(self, all_vectors: ndarray, node_idxs: List[int]) -> None:
        self.__all_vectors = all_vectors
        self.__node_idxs = node_idxs
        # self.__repr_point = ...
        # TODO: hold representing point to improve performance?

    @property
    def node_idxs(self) -> List[int]:
        return self.__node_idxs

    @property
    def size(self) -> int:
        return len(self.__node_idxs)

    @property
    def vectors(self) -> ndarray:
        return self.__all_vectors[self.__node_idxs]

    def add_nodes(self, node_idxs: List[int]) -> None:
        self.__node_idxs.extend(node_idxs)  # REVIEW: need to validate?

    def merge(self, other: Cluster) -> None:
        # TODO: check if using same data in two clusters
        self.__node_idxs.extend(other.node_idxs)
