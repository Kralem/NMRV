from ex1_utils import gausssmooth, gaussderiv, rotate_image, show_flow
import cv2
import numpy as np
import matplotlib.pyplot as plt
from of_methods import lucaskanade, hornschunck

im1 = np.random.rand(200, 200).astype(np.float32)
im2 = im1.copy()
im2 = rotate_image(im2, -1)


#im1 = cv2.imread('cporta_left.png', 0)
#im2 = cv2.imread('cporta_right.png', 0)

U_lk, V_lk = lucaskanade(im1, im2, 10)
#print(U_lk.shape)
#print(V_lk.shape)
U_hs, V_hs = hornschunck(im1, im2, 1000, 0.5)

#print(U_hs.shape)

#print(V_hs.shape)

fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2,2)
ax1_11.imshow(im1)
ax1_12.imshow(im2)
show_flow(U_lk, V_lk, ax1_21, type='angle')
show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
fig1.suptitle('Lucas-Kanade optical flow')

fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2,2)
ax2_11.imshow(im1)
ax2_12.imshow(im2)
show_flow(U_hs, V_hs, ax2_21, type='angle')
show_flow(U_hs, V_hs, ax2_22, type='field', set_aspect=True)
fig2.suptitle('Horn-Schunck optical flow')
plt.show()