from os import makedirs

import matplotlib.pyplot as plt
import numpy as np

from hshca import MultiDimensionalHCA
from hshca.linkmethod import Centroid, Ward  # noqa
from hshca.metric import CityBlock, Euclidean  # noqa

DATA_PATH = "./test/test_cell.npy"
DST_PATH = "./dst/"

METHOD = Ward
METRIC = Euclidean
SPATIAL_DIST_FACTOR = 0.0003
SPATIAL_SCALE = (1.0, 1.0, 1.0)

CLUSTER_NUMS = (3, 4, 5, 6, 7, 8)

makedirs(DST_PATH, exist_ok=True)
data = np.load(DATA_PATH)  # 3D array (x, y, r)

hca = MultiDimensionalHCA(
    data, METHOD, METRIC, show_progress=True)
hca.compute()

for cluster_num in CLUSTER_NUMS:
    res = hca.get_cluster_map(cluster_num).reshape(hca.map_shape)
    plt.imshow(res)

    name = f"cls{cluster_num}_normal.png"
    plt.savefig(DST_PATH + name)
