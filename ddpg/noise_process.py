import numpy as np

# referenced from github.com/minimalrl
class NoiseProcess:
    def __init__(self, action_space, theta,sigma,decay,min_sigma):
        action_shape     = action_space.shape
        self.theta       = theta
        self.sigma       = sigma
        self.sigma_decay = decay
        self.min_sigma   = min_sigma

        self.dt = 0.01

        self.prev_x = np.zeros(action_shape)
        self.mean   = np.zeros(action_shape)

    def sample(self):
        x = self.prev_x + self.theta * self.dt * (self.mean - self.prev_x) + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)

        self.prev_x = x
        return x

    def decay(self):
        self.sigma = max(self.min_sigma, self.sigma - self.sigma_decay)

class NormalNoiseProcess:
    def __init__(self, action_space, var, decay, min_sigma):
        action_shape = action_space.shape

        self.mean   = np.zeros(action_shape)
        self.sigma = var
        self.sigma_decay = decay
        self.min_sigma = min_sigma

    def sample(self):
        return np.random.normal(loc = self.mean, scale=self.sigma, size=self.mean.shape)

    def decay(self):
        self.sigma = max(self.min_sigma, self.sigma - self.sigma_decay)
