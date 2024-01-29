import ibwpy as ip

from hshca import MultiDimensionalHCA
from hshca.linkmethod import Centroid
from hshca.metric import Euclidean

DATA_PATH = "./test/mHeLa_control_1_smth.ibw"
MAP_SHAPE = (30, 30)

METHOD = Centroid
METRIC = Euclidean
CLUSTER_NUM = 5

ibw = ip.load(DATA_PATH)
data = ibw.array  # 4D array (x, y, z, r)
# print(data)

hca = MultiDimensionalHCA(data, METHOD, METRIC)
hca.compute()

dist = hca.linkage_distances
hist = hca.linkage_history
# for d, h in zip(dist, hist):
#     print(d, h)

res = hca.get_fcluster(CLUSTER_NUM)
# print(res)

res_map = hca.get_cluster_map(CLUSTER_NUM).reshape(MAP_SHAPE)
print(res_map)
