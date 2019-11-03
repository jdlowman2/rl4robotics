import gym
import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple

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
            r.append(sample.reward)
            s1.append(sample.next_state)
            d.append(sample.done)

        return Sequence(torch.tensor(s).float(),
                        torch.tensor(a),
                        torch.tensor(r),
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
    def __init__(self, obs_size, action_size):
        super(Actor, self).__init__()
        self.layer1 = torch.nn.Linear(obs_size, 400)
        self.layer2 = torch.nn.Linear(400, 300)
        self.layer3 = torch.nn.Linear(300, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = torch.tanh(x)
        return x


class Critic(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(Critic, self).__init__()
        self.layer1x = torch.nn.Linear(obs_size, 400)
        self.layer1a = torch.nn.Linear(action_size, 400)

        self.layer2 = torch.nn.Linear(800, 1)

    def forward(self, x, a):
        x_layer1 = F.relu(self.layer1x(x))
        a_layer1 = F.relu(self.layer1a(a))

        state_action_pair = torch.cat((x_layer1, a_layer1))

        q_value = F.relu(self.layer2(state_action_pair))

        return q_value

def update_net(net, other_net):
    # referenced from
    # https://stackoverflow.com/questions/49446785/how-can-i-update-the-parameters-of-a-neural-network-in-pytorch
    state_dict = net.state_dict()
    other_state_dict = other_net.state_dict()

    i=0
    for name, param in state_dict.items():
        transformed_param = param * 0.9
        state_dict[name].copy_(transformed_param)

        i+=1

class DDPG:
    def __init__(self, obs_size, action_size):
        self.actor = Actor(obs_size, action_size)
        self.critic = Critic(obs_size, action_size)

        self.target_actor = Actor(obs_size, action_size)
        self.target_critic = Critic(obs_size, action_size)

        self.memory = Memory(MEM_SIZE)

    # main training loop
    def train(self):
        for episode_num in range(MAX_EPISODES):
            noise = NoiseProcess(env.action_space.shape)

            state = env.reset()

            for t in range(MAX_STEPS_PER_EP):
                action = self.actor(torch.from_numpy(state).float()).detach().numpy() + noise.sample()
                next_state, reward, done, _ = env.step(action)

                self.memory.push(\
                    Sequence(state, action, reward, next_state, done))

                self.update_networks()

    # mini-batch sample and update networks
    def update_networks(self):
        batch = self.memory.sample(BATCH_SIZE)
        IPython.embed()
        target_q = batch.reward + GAMMA*self.target_critic(batch.next_state, self.target_actor(batch.next_state))

        loss = 1.0/BATCH_SIZE * sum((target_q - \
                self.critic(batch.state, batch.action))**2)
        policy_grad = 1.0/BATCH_SIZE

        self.target_actor.tau_update(self.actor)
        self.target_critic.tau_update(self.critic)


## Parameters ##
MAX_EPISODES = 200
MAX_STEPS_PER_EP = 300
MEM_SIZE = int(1E6)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001
LEARNING_RATE_ACTOR = 1E-4
LEARNING_RATE_CRITIC = 1E-3

OU_NOISE_THETA = 0.15
OU_NOISE_SIGMA = 0.2

if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    ddpg = DDPG(obs_size, action_size)
    ddpg.train()
