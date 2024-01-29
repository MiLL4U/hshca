import numpy as np

from hshca import MultiDimensionalHCA
from hshca.linkmethod import Centroid
from hshca.metric import Euclidean

DATA_PATH = "./test/test_rand.npy"
METHOD = Centroid
METRIC = Euclidean
CLUSTER_NUM = 5

data = np.array(np.load(DATA_PATH))  # 3D array (x, y, r)
print(data)
print()

vec_num = data.shape[0] * data.shape[1]
data_2d = data.reshape(vec_num, data.shape[2])  # (i, r)
print(data_2d)

# hca = HierarchicalClusterAnalysis(data_2d, METHOD, METRIC)
hca = MultiDimensionalHCA(data, METHOD, METRIC)
hca.compute()

dist = hca.linkage_distances
hist = hca.linkage_history
for d, h in zip(dist, hist):
    print(d, h)

res = hca.get_fcluster(CLUSTER_NUM)
print(res)

res_map = hca.get_cluster_map(CLUSTER_NUM)
print(res_map)
