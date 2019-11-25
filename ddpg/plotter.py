import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import matplotlib.pyplot as plt
import time

# from https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class Plotter:
    def __init__(self, env, print_data_freq):
        plt.ion()
        self.env = env
        self.print_data_freq = print_data_freq
        self.fig, self.axes = plt.subplots(3, 1, num="DDPG Training Progress")
        self.data = []
        self.actions = np.zeros((self.env.action_space.shape[0], 1))
        self.noise = []
        self.actor_loss = []
        self.critic_loss = []
        self.ax2 = None

    def plot(self, new_datum):
        start_time = time.time()
        self.data.append(new_datum)
        self.axes[0].clear()
        self.axes[1].clear()
        self.axes[2].clear()

        self.axes[0].plot(self.print_data_freq * np.arange(len(self.data)), self.data)
        self.axes[0].set_title("Avg Reward Per Episode")
        plt.pause(0.1)

        for action_ind in range(self.actions.shape[0]):
            self.axes[1].plot(self.actions[action_ind,:],
                                label="A_"+str(action_ind))
        self.axes[1].plot(self.noise, label="noise")
        self.axes[1].legend(loc='upper right')
        self.axes[1].set_title("Actions Taken in Last Episode")
        plt.pause(0.1)


        if self.ax2 is not None:
            self.ax2.clear()
        # if len(self.actor_loss) > 10:
        actor_loss = moving_average(self.actor_loss[:], 1)
        critic_loss = moving_average(self.critic_loss[:], 1)
        self.axes[2].plot(actor_loss, label="actor_loss")
        self.ax2 = self.axes[2].twinx()
        self.ax2.plot(critic_loss, 'red',label="critic_loss")
        self.ax2.legend(loc='upper right')
        self.axes[2].legend(loc='upper right')
        self.axes[2].set_title("Network MSE Loss")

        plt.tight_layout()
        plt.pause(0.1)
        plt.show()

        print("Time to plot: ", round(time.time() - start_time, 4))
