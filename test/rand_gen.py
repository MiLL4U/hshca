from random import randint
from random import seed
import numpy as np

X_SIZE = 3
Y_SIZE = 3
R_SIZE = 3

VALUE_RANGE = (0, 99)
SCALE = 0.1
SEED = 1

res_array = np.zeros((X_SIZE, Y_SIZE, R_SIZE))
seed(SEED)
for x in range(X_SIZE):
    for y in range(Y_SIZE):
        res_array[x, y] = np.array([
            randint(VALUE_RANGE[0], VALUE_RANGE[1]) * SCALE
            for _ in range(R_SIZE)])

print(res_array)
np.save("test_rand.npy", res_array)
