import numpy as np
import time
import gym

from torch.distributions import Categorical
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

import sys

############################# PARAMETERS #############################

MAX_EPISODES = 10000
MAX_STEPS_PER_EP = 300
GAMMA = 0.99        # discount factor
LR_ACTOR = 5E-4
LR_CRITIC = 5E-4

PRINT_DATA = 20     # how often to print data
RENDER_GAME = False # View the Episode. 
######################################################################

use_cuda = torch.cuda.is_available()

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_size, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.log_softmax(self.layer3(x),dim=-1)
        return x

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_size, 16)
        self.dp1 = nn.Dropout(p=0.5)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dp1(x)
        x = F.relu(self.layer2(x))
        q_value = self.layer3(x)
        return q_value

class A2C:
    def __init__(self, envname):
        
        self.env = gym.make(envname)
        self.actor = Actor(self.env.observation_space.shape[0], self.env.action_space.n)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic = Critic(self.env.observation_space.shape[0])
        self.critic_optimizer = optim.Adam(self.actor.parameters(), lr=LR_CRITIC)

        if use_cuda:
            self.actor.cuda()
            self.critic.cuda()

        self.data = {"loss": []}
        self.start_time = None

    def select_action(self, state):
        log_probs = self.actor(torch.from_numpy(state).float())
        value = self.critic(torch.from_numpy(state).float())
        action = Categorical(log_probs.exp()).sample()
        return action.data.cpu().numpy(), log_probs[action], value

    def update_a2c(self, rewards, log_probs, values, state):
        

        Qval = self.critic(torch.from_numpy(state).float())
        Qval = Qval.detach().numpy()
        Qvals = np.zeros_like(values, dtype=np.float32)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval 


        Qvals = torch.from_numpy(Qvals).float()
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5*advantage.pow(2).mean()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        ## This stage is broken. 
        actor_loss.backward()
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()


    def train(self):

        score = 0.0
        rewards = []
        log_probs = []
        values = []

        print("Going to be training for a total of {} episodes".format(MAX_EPISODES))
        self.start_time = time.time()
        for e in range(MAX_EPISODES):
            state = self.env.reset()
            
            for t in range(MAX_STEPS_PER_EP):

                if RENDER_GAME:
                    self.env.render()

                action, log_prob, value = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                score += reward
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                if done:
                    break

            if (e+1) % 25 == 0:
                print("Episode: {}, reward: {}".format(e+1, sum(rewards)))

            # Update Actor - Critic 
            self.update_a2c(rewards, log_probs, values, state)

        self.env.close()

if __name__ == "__main__":
    A2C = A2C("LunarLander-v2")
    A2C.train()
