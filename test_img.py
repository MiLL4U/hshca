import ibwpy as ip
import matplotlib.pyplot as plt

from hshca import HyperSpectralHCA
from hshca.linkmethod import Centroid, Ward  # noqa
from hshca.metric import Euclidean, CityBlock  # noqa

DATA_PATH = "./test/mHeLa_control_1_smth.ibw"
MAP_SHAPE = (30, 30)

METHOD = Ward
METRIC = Euclidean
PHYS_DIST_FACTOR = 0.0003
PHYS_SCALE = (1.0, 1.0, 1.0)

CLUSTER_NUM = 5

ibw = ip.load(DATA_PATH)
data = ibw.array  # 4D array (x, y, z, r)
# print(data)

hca = HyperSpectralHCA(
    data, METHOD, METRIC,
    PHYS_DIST_FACTOR, PHYS_SCALE,
    show_progress=True)
hca.print_dist_scales()
hca.compute()

"""
dist = hca.linkage_distances
hist = hca.linkage_history
for d, h in zip(dist, hist):
    print(d, h)
"""

res = hca.get_fcluster(CLUSTER_NUM)

res_map = hca.get_cluster_map(CLUSTER_NUM).reshape(MAP_SHAPE).T
plt.imshow(res_map)
plt.show()
