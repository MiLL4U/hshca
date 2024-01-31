from os import makedirs

import ibwpy as ip
import matplotlib.pyplot as plt

from hshca import MultiDimensionalHCA
from hshca.linkmethod import Centroid, Ward  # noqa
from hshca.metric import CityBlock, Euclidean  # noqa

DATA_PATH = "./test/mHeLa_control_1_smth.ibw"
DST_PATH = "./dst/"
MAP_SHAPE = (30, 30)

METHOD = Ward
METRIC = Euclidean
SPATIAL_DIST_FACTOR = 0.0003
SPATIAL_SCALE = (1.0, 1.0, 1.0)

CLUSTER_NUMS = (3, 4, 5, 6, 7, 8)

makedirs(DST_PATH, exist_ok=True)

ibw = ip.load(DATA_PATH)
data = ibw.array  # 4D array (x, y, z, r)
# print(data)

hca = MultiDimensionalHCA(
    data, METHOD, METRIC, show_progress=True)
hca.compute()

for cluster_num in CLUSTER_NUMS:
    res = hca.get_cluster_map(cluster_num).reshape(MAP_SHAPE).T
    plt.imshow(res)

    name = f"cls{cluster_num}_normal.png"
    plt.savefig(DST_PATH + name)
