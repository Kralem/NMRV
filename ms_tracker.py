import numpy as np
import scipy.signal as sp
import math
from ex2_utils import Tracker, get_patch, extract_histogram, backproject_histogram, create_epanechnik_kernel


class MeanShiftTracker(Tracker):
    def initialize(self, image, region):
        self.size = (region[2], region[3])
        self.region = region
        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.kernel = create_epanechnik_kernel(region[2], region[3], 4)
        #print(np.shape(self.kernel))
        self.oblika = np.shape(self.kernel)
        self.patch, inliners = get_patch(image, self.position, (self.oblika))
        #print(np.shape(self.patch))
        self.q = extract_histogram(self.patch, 16, self.kernel)
        #print(np.shape(self.q))


    def track(self, image):
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)
        pozicija = self.position
        for i in range(20):
            patch, _ = get_patch(image, pozicija, (self.oblika[0], self.oblika[0]))
            p = extract_histogram(patch, 16)
            v = np.sqrt(self.q / (p + 0.00005))
            wi = backproject_histogram(patch, v, 16)
            dimenzije = np.shape(wi)
            shape = int(math.floor(dimenzije[0] / 2))
            xi, yi = np.meshgrid(np.arange(-shape, shape + 1), np.arange(-shape, shape + 1))
            #print(np.shape(wi))
            x_k = np.sum(xi * wi) / np.sum(wi)
            y_k = np.sum(yi* wi) / np.sum(wi)
            new_x = pozicija[0] + x_k
            new_y = pozicija [1] + y_k
            pozicija = [new_x, new_y]

            patch, _ = get_patch(image, pozicija, self.oblika)
            q = extract_histogram(patch, 16, self.kernel)
            self.q = np.add(((1 - 0.5) * self.q), (0.5 * q))

            if x_k < 0.001 and y_k < 0.001:
                break

        return [pozicija[0], pozicija[1], self.size[0], self.size[1]]

class MSParams():
    def __init__(self):
        self.enlarge_factor = 2