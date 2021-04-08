from ex2_utils import Tracker, get_patch
from ex3_utils import create_gauss_peak, create_cosine_window
import numpy as np
import cv2

class Project3Tracker(Tracker):
    def initialize(self, image, region):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.size = (region[2], region[3])
        self.region = region
        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.G = create_gauss_peak((self.size[1], self.size[0]), 1)
        self.patch, inliers = get_patch(image, self.position, self.G.shape[::-1])
        self.cosine = create_cosine_window(self.patch.shape[::-1])
        self.patch = np.multiply(self.cosine, self.patch)
        #print(np.shape(self.patch))
        #print(np.shape(self.patch))
        #print(np.shape(self.G))
        self.H = np.divide(np.multiply(np.fft.fft2(self.G), np.conj(np.fft.fft2(self.patch))),
                           (np.multiply(np.fft.fft2(self.patch), np.conj(np.fft.fft2(self.patch))) + 0.000005))

    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        patch_n, inliers = get_patch(image, self.position, self.G.shape[::-1])
        cosine_n = create_cosine_window(patch_n.shape[::-1])
        patch_n = np.multiply(patch_n, cosine_n)
        oblika = patch_n.shape[::-1]
        H_n = np.divide(np.multiply(np.fft.fft2(self.G), np.conj(np.fft.fft2(patch_n))),
                           (np.multiply(np.fft.fft2(patch_n), np.conj(np.fft.fft2(patch_n))) + 0.000005))
        H_update = (1 - 0.9)*self.H + 0.9*H_n
        R = np.fft.ifft2(np.multiply(H_update, np.fft.fft2(patch_n)))
        R_max = np.max(R)
        pozicija = np.where(R == R_max)
        print(pozicija)
        kx = pozicija[1][0]
        ky = pozicija[0][0]
        if kx > (self.size[0] / 2):
            x = kx - self.size[0]
            kx = kx + x
        if ky > (self.size[1] / 2):
            y = kx - self.size[1]
            ky = ky + y


        return [kx, ky, self.size[0], self.size[1]]


class Project3Params():
    def __init__(self):
        self.enlarge_factor = 2

