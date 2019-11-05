import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import matplotlib.pyplot as plt

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
    def __init__(self, obs_size, action_size):
        super(Actor, self).__init__()
        self.layer1 = torch.nn.Linear(obs_size, 128)
        self.layer2 = torch.nn.Linear(128, 64)
        self.layer3 = torch.nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = torch.tanh(x)*2
        return x


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

    # target_net = TAU * net + (1 - TAU) * target_net

class DDPG:
    def __init__(self, obs_size, action_size):
        self.actor = Actor(obs_size, action_size)
        self.critic = Critic(obs_size, action_size)

        self.target_actor = Actor(obs_size, action_size)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic = Critic(obs_size, action_size)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), LEARNING_RATE_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), LEARNING_RATE_CRITIC)

        self.memory = Memory(MEM_SIZE)

        self.data = {"loss": []}

    # main training loop
    def train(self):
        score = 0.0
        for episode_num in range(MAX_EPISODES):
            noise = NoiseProcess(env.action_space.shape)

            state = env.reset()

            for t in range(MAX_STEPS_PER_EP):
                action = self.actor(torch.from_numpy(state).float()).detach().numpy() + noise.sample()
                next_state, reward, done, _ = env.step(action)

                self.memory.push( \
                    Sequence(state, action, reward/100.0, next_state, done))

                score += reward
                state = next_state

                if done:
                    break

            if self.memory.max_entry > MEMORY_MIN:
                for _ in range(10):
                    self.update_networks()

            # print average score not loss
            # if episode_num % PRINT_DATA == 0:
            #     print("Episode: ", episode_num, " / ", MAX_EPISODES,
            #             " | Avg loss this period: ",
            #                 np.array(self.data["loss"][-PRINT_DATA:]).mean().round(4))

            if episode_num % PRINT_DATA == 0 and episode_num != 0 :
                print("Episode: ", episode_num, " / ", MAX_EPISODES,
                      " | Avg Score: ",
                      np.array(score/PRINT_DATA).round(4))
                score = 0.0
        env.close()

    # mini-batch sample and update networks
    def update_networks(self):
        batch = self.memory.sample(BATCH_SIZE)


        # reward is shape [64], target_critic output is [64, 1]
        target_q = batch.reward + GAMMA * \
                        self.target_critic(batch.next_state,
                            self.target_actor(batch.next_state))

        # print(batch.state.size(), batch.action.size(), batch.reward.size())
        # print(self.critic(batch.state, batch.action).size(), target_q.size())

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
    # env = gym.make("MountainCarContinuous-v0")
    env = gym.make("Pendulum-v0")
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    ddpg = DDPG(obs_size, action_size)
    ddpg.train()
