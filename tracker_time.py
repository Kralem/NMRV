from ex2_utils import Tracker
from ex2_utils import get_patch
from ex3_utils import create_gauss_peak, create_cosine_window
import numpy as np
import cv2
import time

class Project3Tracker(Tracker):
    def name(self):
        return 'Project3Tracker'

    def initialize(self, image, region):
        self.fps = 0
        self.no_frames = 0
        t0 = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.size = (region[2], region[3])
        self.region = region
        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)

        self.G = create_gauss_peak((self.size[0], self.size[1]), 1.75)
        self.patch, inliers = get_patch(image, self.position, self.G.shape[::-1])
        self.cosine = create_cosine_window(self.patch.shape[::-1])
        self.patch = np.multiply(self.cosine, self.patch)
        #print(np.shape(self.patch))
        #print(np.shape(self.patch))
        #print(np.shape(self.G))
        self.H = np.divide(np.multiply(np.fft.fft2(self.G), np.conj(np.fft.fft2(self.patch))),
                           (np.multiply(np.fft.fft2(self.patch), np.conj(np.fft.fft2(self.patch))) + 0.000005))
        t1 = time.time()
        if (t1 - t0) != 0:
            self.fps = self.fps + (1 / (t1 - t0))
            self.no_frames += 1
            print("total time for initializing tracker", self.fps)
        self.fps = 0

    def track(self, image):
        t0 = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        oblika = self.G.shape[::-1]
        patch, inliers = get_patch(image, self.position, self.G.shape[::-1])
        patch = np.multiply(self.cosine, patch)
        R = np.real(np.fft.ifft2(np.multiply(self.H, np.fft.fft2(patch))))
        #cv2.imshow('okno',R)
        #cv2.waitKey(0)
        ind = np.unravel_index(np.argmax(R, axis=None), R.shape)

        kx = ind[1]
        ky = ind[0]
        if kx > (self.size[0] / 2):
            kx = kx - self.size[0]
        if ky > (self.size[1] / 2):
            ky = ky - self.size[1]


        self.position = (self.position[0]+kx, self.position[1]+ky)
        patch_n, inliers = get_patch(image, self.position, self.G.shape[::-1])
        patch_n = np.multiply(patch_n, self.cosine)
        H_n = np.divide(np.multiply(np.fft.fft2(self.G), np.conj(np.fft.fft2(patch_n))),
                        (np.multiply(np.fft.fft2(patch_n), np.conj(np.fft.fft2(patch_n))) + 0.000005))
        self.H = (1 - 0.06) * self.H + 0.06 * H_n

        t1 = time.time()
        if (t1 - t0) != 0:
            self.fps = self.fps + (1 / (t1 - t0))
            self.no_frames += 1
            print("total time for updating tracker", self.fps)
            print("processed frames", self.no_frames)


        return [self.position[0] - (self.size[0] /2), self.position[1] - (self.size[1] / 2), self.size[0], self.size[1]]


class Project3Params():
    def __init__(self):
        self.enlarge_factor = 2

