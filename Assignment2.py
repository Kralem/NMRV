from ex2_utils import *
from ex4_utils import gaussian_prob, sample_gauss
from kalman import NCV, NCA, RW
from gensim.matutils import hellinger


class ParticleTracker(Tracker):
    def sample_new_particles(self, weights, particles, N):
        weights_norm = weights / np.sum(weights)
        weights_cumsumed = np.cumsum(weights_norm)
        rand_samples = np.random.rand(N, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)
        particles_new = particles[sampled_idxs.flatten(), :]
        return particles_new

    def initialize(self, image, region):

        #the usual initialization
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)

        #create tracking model
        self.Fi, self.H, self.Q, self.R = NCV(100, 1)
        self.size = (region[2], region[3])

        #create particles
        self.num_particles = 100
        self.particles = sample_gauss(np.array([self.position[0], self.position[1],0,0]), self.Q, self.num_particles)
        self.weights = np.ones(self.particles.shape[0])

        #create kernel and extract histogram
        self.epanechnikov = create_epanechnik_kernel(region[2], region[3], 1)
        self.patch, inliers = get_patch(image, self.position, self.epanechnikov.shape[::-1])
        self.q = extract_histogram(self.patch, 8, self.epanechnikov)

    def track(self, image):
        self.particles = self.sample_new_particles(self.weights, self.particles, self.num_particles) #sample new
        #print(np.shape(self.particles))
        #print(type(self.particles))

        noise = sample_gauss(np.zeros(4), self.Q, self.num_particles)

        particle_position_x = []
        particle_position_y = []

        for i, particle in enumerate(self.particles): #go through particles
            p = self.particles[i]
            state = np.matmul(self.Fi, p) + noise[i]
            self.particles[i] = state

            nx = state[0]
            ny = state[1]

            patch, inliers = get_patch(image, (nx, ny), self.epanechnikov.shape[::-1]) #extract patch

            hist_pi = extract_histogram(patch, 8, self.epanechnikov) #extract histogram on patch

            dh = hellinger(hist_pi, self.q) #calculate hellinger for hists with gensim

            self.weights[i] = np.exp(-0.5 * (pow(dh, 2) / pow(2, 2)))
            #print(self.weights[i])

            particle_position_x.append(nx)
            particle_position_y.append(ny)

        nx = np.sum(self.weights * np.array(particle_position_x)) / np.sum(self.weights) #calculate new position from weights
        ny = np.sum(self.weights * np.array(particle_position_y)) / np.sum(self.weights)

        self.position = (nx, ny)
        normalized_x = self.position[0] - (self.size[0] /2)
        normalized_y = self.position[1] - (self.size[1] /2)

        patch_n, inliers = get_patch(image, self.position, self.epanechnikov.shape[::-1])

        hn = extract_histogram(patch_n, 8, self.epanechnikov)
        self.q = (1 - 0.07) * self.q + (0.07 * hn)

        return [normalized_x, normalized_y, self.size[0], self.size[1]]







class PTParams():
    def __init__(self):
        self.enlarge_factor = 2
