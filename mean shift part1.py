import numpy as np
from ex2_utils import generate_responses_1, create_epanechnik_kernel, get_patch
import random

response = generate_responses_1()
xi = np.matrix([[-2, -1, 0, 1, 2],[-2, -1, 0, 1, 2],[-2, -1, 0, 1, 2],[-2, -1, 0, 1, 2],[-2, -1, 0, 1, 2]])
yi = np.matrix([[-2,-2,-2,-2,-2], [-1,-1,-1,-1,-1], [0,0,0,0,0], [1,1,1,1,1], [2,2,2,2,2]])

num_iterations = 0
sx = random.randint(40,60)
sy = random.randint(40,60)
center = [sx, sy]
print(center)
#60 43
#60 42
#51 42
#53 45
#53 41
#57 46
#60 50
#59 48
#52 42
#55 47
#57 41

for i in range(500): #da se slučajno ne zacikla
    wi, inliers = get_patch(response, center, (5, 5))
    x_k = np.sum(np.multiply(xi,wi)) / np.sum(wi)
    y_k = np.sum(np.multiply(yi,wi)) / np.sum(wi)
    #print(x_k)
    #print(y_k)
    #center[0] = center[0] + x_k
    #center[1] = center[1] + y_k
    sx = sx + x_k
    sy = sy + y_k
    center = [sx, sy]

    wi, inliers = get_patch(response, center, (5, 5))
    num_iterations+=1

    if x_k < 0.01 and y_k < 0.01:
        break

print(num_iterations)
print(int(center[0]), int(center[1]))
print(response[int(center[0]), int(center[1])])
print(np.max(response))


