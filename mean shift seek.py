import numpy as np
from ex2_utils import generate_responses_1, create_epanechnik_kernel, get_patch
import random
import math

response = generate_responses_1()
xi = np.matrix([[-2, -1, 0, 1, 2],[-2, -1, 0, 1, 2],[-2, -1, 0, 1, 2],[-2, -1, 0, 1, 2],[-2, -1, 0, 1, 2]])
yi = np.matrix([[-2,-2,-2,-2,-2], [-1,-1,-1,-1,-1], [0,0,0,0,0], [1,1,1,1,1], [2,2,2,2,2]])
num_iterations = 0
sx = random.randint(40,50)
sy = random.randint(40,50)
center = [sx, sy]
wi, inliers = get_patch(response, center, (5,5))

for i in range(100):
    x_k = np.divide(np.sum(xi*wi), np.sum(wi))
    y_k = np.divide(np.sum(yi*wi), np.sum(wi))
    sx = math.ceil(sx + x_k)
    sy = math.ceil(sy + y_k)
    center = [sx, sy]
    wi, inliers = get_patch(response, center, (5, 5))
    num_iterations+=1
    print(sx, sy)
    print(response[center[0], center[1]])

print(num_iterations)

