import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import matplotlib.pyplot as plt
import time
import csv

import IPython

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


# referenced from github.com/minimalrl
class NoiseProcess:
    def __init__(self, action_shape):
        self.theta = OU_NOISE_THETA
        self.sigma = OU_NOISE_SIGMA
        self.dt = 0.01
        self.prev_x = np.zeros(action_shape)
        self.mean = np.zeros(action_shape)


    def sample(self):
        x = self.prev_x + self.theta * self.dt * (self.mean - self.prev_x) + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)

        self.prev_x = x
        return x


class Actor(torch.nn.Module):
    def __init__(self, obs_size, action_space):
        super(Actor, self).__init__()
        self.layer1 = torch.nn.Linear(obs_size, 128)
        self.layer2 = torch.nn.Linear(128, 64)
        self.layer3 = torch.nn.Linear(64, action_space.shape[0])

        self.action_space = action_space

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = torch.tanh(x) * torch.from_numpy(self.action_space.high).float()
        return x

    def take_action(self, state, added_noise=None):
        state_x = torch.from_numpy(state).float()
        action = self.forward(state_x).detach().numpy()

        if added_noise is not None:
            action += added_noise

        return action


class Critic(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(Critic, self).__init__()
        self.layer1x = torch.nn.Linear(obs_size, 64)
        self.layer1a = torch.nn.Linear(action_size, 64)

        self.layer2 = torch.nn.Linear(128, 32)
        self.layer3 = torch.nn.Linear(32, 1)

    def forward(self, x, a):
        x_layer1 = F.relu(self.layer1x(x))
        a_layer1 = F.relu(self.layer1a(a))

        state_action_pair = torch.cat([x_layer1, a_layer1], dim=1)
        intermediate = F.relu(self.layer2(state_action_pair))
        q_value = self.layer3(intermediate)

        return q_value


def update_net(target_net, net, tau):
    # referenced from
    # https://stackoverflow.com/questions/49446785/how-can-i-update-the-parameters-of-a-neural-network-in-pytorch
    target_dict = target_net.state_dict()
    other_dict = net.state_dict()

    for key, val in target_dict.items():
        other_val = other_dict[key]
        averaged_val = tau * other_val + (1.0 - tau) * val
        target_dict[key].copy_(averaged_val)


class Plotter:
    def __init__(self, print_data_freq):
        plt.ion()
        self.print_data_freq = print_data_freq
        self.fig, self.ax = plt.subplots(1, 1, num="DDPG Training Progress")
        self.data = []

    def plot(self, new_datum):
        self.data.append(new_datum)
        self.ax.clear()
        self.ax.plot(self.data)
        plt.pause(0.1)
        plt.show()

class DDPG:
    def __init__(self, envname):
        self.envname = envname
        self.env = gym.make(envname)
        self.reset()

    def reset(self):
        obs_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]

        self.env.reset()

        self.actor = Actor(obs_size, self.env.action_space)
        self.critic = Critic(obs_size, action_size)

        self.target_actor = Actor(obs_size, self.env.action_space)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic = Critic(obs_size, action_size)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), LEARNING_RATE_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), LEARNING_RATE_CRITIC)

        self.memory = Memory(MEM_SIZE)

        self.data = {"loss": []}
        self.start_time = None

        self.plotter = Plotter(PRINT_DATA)

    # main training loop
    def train(self):
        score = 0.0
        self.start_time = time.time()
        for episode_num in range(MAX_EPISODES):
            noise = NoiseProcess(self.env.action_space.shape)

            state = self.env.reset()

            for t in range(MAX_STEPS_PER_EP):
                action = self.actor.take_action(state, noise.sample())
                next_state, reward, done, _ = self.env.step(action)

                self.memory.push( \
                    Sequence(state, action, reward/100.0, next_state, done))

                score += reward
                state = next_state

                if done:
                    break

            if self.memory.max_entry > MEMORY_MIN:
                for _ in range(10):
                    self.update_networks()

            if episode_num % PRINT_DATA == 0 and episode_num != 0 :
                print("Episode: ", episode_num, " / ", MAX_EPISODES,
                      " | Avg Score: ",
                      np.array(score/PRINT_DATA).round(4),
                      " | Elapsed time [s]: ",
                      round((time.time() - self.start_time), 2),
                      )

                self.plotter.plot(score/PRINT_DATA)

                score = 0.0
        self.env.close()

    # mini-batch sample and update networks
    def update_networks(self):
        batch = self.memory.sample(BATCH_SIZE)

        # reward is shape [64], target_critic output is [64, 1]
        target_q = batch.reward + GAMMA * \
                        self.target_critic(batch.next_state,
                            self.target_actor(batch.next_state))

        # update critic network
        critic_loss = F.smooth_l1_loss(self.critic(batch.state, batch.action), target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor network
        actor_loss = -self.critic(batch.state, self.actor(batch.state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        update_net(self.target_actor, self.actor, TAU)
        update_net(self.target_critic, self.critic, TAU)

        # self.data["loss"].append(loss.item())

    def demonstrate(self, save_snapshots=None):
        self.env.close()
        self.env = gym.make(self.envname)

        state = self.env.reset()
        done = False
        step_num=0
        while not done:
            self.env.render()
            action = self.actor(torch.from_numpy(state).float()).detach().numpy()
            state, reward, done, _ = self.env.step(action)
            step_num += 1

    def save_experiment(self, experiment_name):
        torch.save(self.actor.state_dict(), "experiments/" + experiment_name + "_actor")
        torch.save(self.critic.state_dict(), "experiments/" + experiment_name + "_critic")

        parameters = {
            "Environment Name": self.envname,
            "MAX_EPISODES":MAX_EPISODES,
            "MAX_STEPS_PER_EP":MAX_STEPS_PER_EP,
            "MEM_SIZE":MEM_SIZE,
            "MEMORY_MIN":MEMORY_MIN,
            "BATCH_SIZE":BATCH_SIZE,
            "GAMMA":GAMMA,
            "TAU":TAU,
            "LEARNING_RATE_ACTOR":LEARNING_RATE_ACTOR,
            "LEARNING_RATE_CRITIC":LEARNING_RATE_CRITIC,
            "OU_NOISE_THETA":OU_NOISE_THETA,
            "OU_NOISE_SIGMA":OU_NOISE_SIGMA,
            "start time": self.start_time,
            }

        with open("experiments/" + experiment_name + ".csv", "w") as file:
            w = csv.writer(file)
            for key, val in parameters.items():
                w.writerow([key, val, "\n"])

    def load_experiment(self, experiment_name):
        # NOTE: this does not load the global training parameters, so you
        # can't continue training
        actor_file = "experiments/" + experiment_name + "_actor"
        self.actor.load_state_dict(torch.load(actor_file))
        critic_file = "experiments/" + experiment_name + "_critic"
        self.critic.load_state_dict(torch.load(critic_file))

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())


## Parameters ##
MAX_EPISODES = 1000
MAX_STEPS_PER_EP = 300
MEM_SIZE = int(1E6)
MEMORY_MIN = int(2E3)
BATCH_SIZE = 64
GAMMA = 0.99    # discount factor
TAU = 0.005     # tau averaging for target network updating
LEARNING_RATE_ACTOR = 5E-4
LEARNING_RATE_CRITIC = 1E-3
OU_NOISE_THETA = 0.1
OU_NOISE_SIGMA = 0.1

PRINT_DATA = 20  # how often to print data

if __name__ == "__main__":
    # ddpg = DDPG("MountainCarContinuous-v0")
    # ddpg = DDPG("LunarLanderContinuous-v2")
    ddpg = DDPG("Pendulum-v0")
    ddpg.train()
    ddpg.demonstrate()

    IPython.embed() # use this to save or load networks
