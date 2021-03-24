import numpy as np
from ex2_utils import generate_responses_1, create_epanechnik_kernel, get_patch
import random

response = generate_responses_1()
xi = np.matrix([[-2, -1, 0, 1, 2],[-2, -1, 0, 1, 2],[-2, -1, 0, 1, 2],[-2, -1, 0, 1, 2],[-2, -1, 0, 1, 2]])
yi = np.matrix([[-2,-2,-2,-2,-2], [-1,-1,-1,-1,-1], [0,0,0,0,0], [1,1,1,1,1], [2,2,2,2,2]])

num_iterations = 0
sx = random.randint(40,50)
sy = random.randint(40,50)
center = [sx, sy]


for i in range(100):
    wi, inliers = get_patch(response, center, (5, 5))
    x_k = np.sum(xi * wi) / np.sum(wi)
    y_k = np.sum(yi * wi) / np.sum(wi)
    print(x_k)
    print(y_k)
    sx = sx + x_k
    sy = sy + y_k
    center = [sx, sy]
    wi, inliers = get_patch(response, center, (5, 5))
    num_iterations+=1

    if x_k <= 0.05 and y_k <= 0.05:
        break


print(num_iterations)

