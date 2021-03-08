import cv2
import numpy as np
import scipy as sp
from ex1_utils import gausssmooth, gaussderiv, rotate_image, show_flow
import matplotlib.pyplot as plt

def lucaskanade (im1 ,im2, N):
    #im1 − first image matrix (grayscale)
    #im2 − second image matrix (grayscae)
    #n − size of the neighborhood (N x N)
    #TODO : the algorithm
    matrika = np.ones((N, N)) #prvi korak, sestavimo osnovne spremenljivke
    Ix1, Iy1 = gaussderiv(im1, 1)
    Ix2, Iy2 = gaussderiv(im2, 1)
    Ix = np.add(Ix1, Ix2) / 2
    Iy = np.add(Iy1, Iy2) / 2
    It = im1 - im2
    It = gausssmooth(It, 1)
    IxIt = np.multiply(Ix, It) #drugi korak, začnemo jih kombinirati
    IyIt = np.multiply(Iy, It)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)
    IxIy = np.multiply(Ix, Iy)
    Sigmaxt = cv2.filter2D(IxIt, -1, matrika)
    Sigmayt = cv2.filter2D(IyIt, -1, matrika)
    Sigmax2 = cv2.filter2D(Ix2, -1, matrika)
    Sigmay2 = cv2.filter2D(Iy2, -1, matrika)
    Sigmaxy = cv2.filter2D(IxIy, -1, matrika)
    negy2 = Sigmay2 * -1 #tretji korak, sestavimo jih v en skupni račun
    zgornji = np.add(np.multiply(negy2, Sigmaxt), np.multiply(Sigmaxy, Sigmayt))
    spodnji = np.subtract(np.multiply(Sigmax2, Sigmay2), np.multiply(Sigmaxy, Sigmaxy))
    u = np.divide(zgornji, spodnji)
    zgornji2 = np.subtract(np.multiply(Sigmaxy, Sigmaxt), np.multiply(Sigmax2, Sigmayt))
    v = np.divide(zgornji2, spodnji)
    return u, v

im1 = np.random.rand(200, 200).astype(np.float32)
im2 = im1.copy()
im2 = rotate_image(im2, -1)

U_lk, V_lk = lucaskanade(im1, im2, 3)
#print(U_lk)
#print(V_lk)
fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2,2)
ax1_11.imshow(im1)
ax1_12.imshow(im2)
show_flow(U_lk, V_lk, ax1_21, type='angle')
show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
fig1.suptitle('Lucas-Kanade optical flow')
plt.show()