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
    spodnji[spodnji == 0 ] = 1
    u = np.divide(zgornji, spodnji)
    zgornji2 = np.subtract(np.multiply(Sigmaxy, Sigmaxt), np.multiply(Sigmax2, Sigmayt))
    v = np.divide(zgornji2, spodnji)
    return u, v

def hornschunck(im1, im2, n_iters, lmbd):
    #im1 − first image matrix (grayscale)
    #im2 − second image matrix (grayscae)
    #n_iter - number of iterations (try several hundred)
    #lmbd - parameter
    #TODO : the algorithm
    u = 0 * im1 #prvi korak, definiramo u, v in vse ostalo kar potrebujemo
    v = 0 * im1
    Ix1, Iy1 = gaussderiv(im1, 1)
    Ix2, Iy2 = gaussderiv(im2, 1)
    Ix = np.add(Ix1, Ix2) / 2
    Iy = np.add(Iy1, Iy2) / 2
    It = im1 - im2
    It = gausssmooth(It, 1)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)
    D = lmbd + np.add(Ix2, Iy2)
    kernel = np.matrix([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    for _ in range(n_iters): #drugi korak, začnemo z iteracijo in naredimo konvolucijo z laplace-om
        ua = cv2.filter2D(u, -1, kernel)
        va = cv2.filter2D(v, -1, kernel)
        P = np.add(np.add(np.multiply(Ix, ua), np.multiply(Iy, va)), It) #tretji korak, izračunamo P ter naslednji u in v znotraj iteracije
        u = np.subtract(ua, np.multiply(Ix, np.divide(P,D)))
        v = np.subtract(va, np.multiply(Iy, np.divide(P, D)))
    return u, v

im1 = np.random.rand(200, 200).astype(np.float32)
im2 = im1.copy()
im2 = rotate_image(im2, -1)


#im1 = cv2.imread('cporta_left.png', 0)
#im2 = cv2.imread('cporta_right.png', 0)

U_lk, V_lk = lucaskanade(im1, im2, 3)
print(U_lk.shape)
print(V_lk.shape)
U_hs, V_hs = hornschunck(im1, im2, 1000, 0.5)

print(U_hs.shape)
print(V_hs.shape)

#print(U_lk) #preverimo izhode, če so matrike pravilne velikosti
#print(V_lk)
#print(U_hs)
#print(V_hs)
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