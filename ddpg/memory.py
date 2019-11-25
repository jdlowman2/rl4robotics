import numpy as np
import torch
from collections import namedtuple

Sequence = namedtuple("Sequence", \
                ["state", "action", "reward", "next_state", "done"])


class Memory:
    def __init__(self, size):
        self.size = size
        self.data = np.empty(size, dtype=Sequence)
        self.point_ind = 0
        self.max_entry = 0

    def push(self, sequence):
        self.data[self.point_ind] = sequence
        self.point_ind = (1+self.point_ind) % self.size
        self.max_entry = max(self.max_entry, self.point_ind)

    def sample(self, num_samples):
        # get sample of sequences
        samples = np.random.choice(self.data[:self.max_entry], num_samples)

        # convert to single sequence of samples for batch processing
        s, a, r, s1, d = [], [], [], [], []
        for sample in samples:
            s.append(sample.state)
            a.append(sample.action)
            r.append([sample.reward])
            s1.append(sample.next_state)
            d.append([sample.done])

        return Sequence(torch.tensor(s).float(),
                        torch.tensor(a).float(),
                        torch.tensor(r).float(),
                        torch.tensor(s1).float(),
                        torch.tensor(d))
