import torch
from collections import namedtuple, deque
import random

Sequence = namedtuple("Sequence", \
                ["state", "action", "reward", "next_state", "done"])

class Memory:
    def __init__(self, size):
        self.size = size
        self.data = deque(maxlen=size)
        self.max_entry = 0

    def push(self, sequence):
        self.data.append(sequence)
        self.max_entry = len(self.data)

    def sample(self, num_samples):
        samples = random.sample(self.data, num_samples)

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
