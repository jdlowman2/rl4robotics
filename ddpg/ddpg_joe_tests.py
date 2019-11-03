import numpy as np
import matplotlib.pyplot as plt
import unittest

import IPython
from ddpg_joe import *

def make_dummy_sequence():
    return Sequence(np.zeros((4, 1)), 1.0, 0.0, np.ones((4, 1)), False)

class TestMemory(unittest.TestCase):

    def test_wraparound(self):
        start_seq = make_dummy_sequence()
        s2 = Sequence(np.zeros((4, 1)), -1, 1, np.ones((4, 1)), False)
        s3 = Sequence(np.zeros((4, 1)), 0, 0, np.ones((4, 1)), False)

        mem = Memory(2)

        mem.push(start_seq)
        self.assertTrue(mem.max_entry < mem.size)
        mem.push(s2)
        self.assertTrue(mem.max_entry==mem.size-1)
        mem.push(s3)
        self.assertTrue(mem.max_entry==mem.size-1)

        self.assertTrue(mem.data[0] == s3)
        self.assertTrue(mem.data[1] == s2)

        self.assertTrue(mem.data[0].state in mem.sample(2).state or\
                        mem.data[1].state in mem.sample(2).state)

        self.assertTrue(mem.data[0].action in mem.sample(2).action or\
                        mem.data[1].action in mem.sample(2).action)

        self.assertTrue(mem.data[0].next_state in mem.sample(2).next_state or\
                        mem.data[1].next_state in mem.sample(2).next_state)

        self.assertTrue(mem.data[0].done in mem.sample(2).done or\
                        mem.data[1].done in mem.sample(2).done)

    def test_batch_torch(self):
        mem = Memory(2)
        mem.push(make_dummy_sequence())
        mem.push(make_dummy_sequence())

        batch = mem.sample(2)
        self.assertTrue(type(batch.state) == torch.Tensor)
        self.assertTrue(type(batch.next_state) == torch.Tensor)

    # def test_pytorch_memory(self):
    #     mem = Memory(2)
    #     mem.push(make_dummy_sequence())
    #     mem.push(make_dummy_sequence())

    #     IPython.embed()

    #     batch = mem.sample(2)

    #     self.assertTrue((torch_batch.state.numpy() == batch.state).all())
    #     self.assertTrue((torch_batch.action.numpy() == batch.action).all())
    #     self.assertTrue((torch_batch.next_state.numpy() == batch.next_state).all())
    #     self.assertTrue((torch_batch.reward.numpy() == batch.reward).all())
    #     self.assertTrue((torch_batch.done.numpy() == batch.done).all())

class TestNoise(unittest.TestCase):

    def test_noise(self):
        noise = NoiseProcess((2, 1))

        d = []
        for i in range(100):
            d.append(noise.sample())

        print("Shape of sample: ", noise.sample().shape)
        self.assertTrue(noise.sample().shape == (2,1))
        plt.plot([i[0] for i in d], label="action 0")
        plt.plot([i[1] for i in d], label="action 1")
        plt.legend()
        plt.xlabel("iteration number")
        plt.ylabel("noise")
        # plt.show()

class TestDDPG(unittest.TestCase):

    def test_network_inputs(self):

        dummy_action = torch.tensor(np.zeros((2,))).float()
        dummy_obs = torch.tensor(np.zeros((10,))).float()

        ddpg = DDPG(10, 2)

        action_selected = ddpg.actor(dummy_obs)
        q = ddpg.critic(dummy_obs, action_selected)


if __name__ == "__main__":
    unittest.main()
