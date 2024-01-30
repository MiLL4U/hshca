import numpy as np
from scipy.spatial import distance

DATA_PATH = "./test/test_rand.npy"
METRIC = "euclidean"

data = np.array(np.load(DATA_PATH))  # 3D array (x, y, r)
print(data)

vec_num = data.shape[0] * data.shape[1]
data_2d = data.reshape(vec_num, data.shape[2])  # (i, r)
print(data_2d)

dist_matrix = distance.cdist(data_2d, data_2d, metric=METRIC)
