from typing import List

import ibwpy as ip
import matplotlib.pyplot as plt
import numpy as np

from hshca import HyperSpectralHCA
from hshca.linkmethod import Centroid, Ward  # noqa
from hshca.metric import CityBlock, Euclidean  # noqa

DATA_PATH = "./test/mHeLa_control_1_smth.ibw"
MAP_SHAPE = (30, 30)

METHOD = Ward
METRIC = Euclidean
SPATIAL_FACTORS = (0.00001, 0.0001, 0.0003, 0.0005, 0.0008, 0.003, 0.03)
SPATIAL_SCALE = (1.0, 1.0, 1.0)

CLUSTER_NUM = 5

ibw = ip.load(DATA_PATH)
data = ibw.array  # 4D array (x, y, z, r)

results: List[np.ndarray] = []
for factor in SPATIAL_FACTORS:
    print(factor)
    hca = HyperSpectralHCA(
        data, METHOD, METRIC,
        factor, SPATIAL_SCALE,
        show_progress=True)
    hca.compute()
    res = hca.get_cluster_map(CLUSTER_NUM).reshape(MAP_SHAPE).T
    results.append(res)

for factor, result in zip(SPATIAL_FACTORS, results):
    print(factor)
    plt.imshow(result)
    plt.show()
    plt.close()
