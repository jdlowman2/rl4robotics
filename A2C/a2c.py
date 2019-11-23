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
TEST_FREQUENCY = 100
TEST_EPISODES = 100
GAMMA = 0.99        # discount factor
LR_ACTOR = 5E-4
LR_CRITIC = 5E-4

PRINT_DATA = 20     # how often to print data
RENDER_GAME = False # View the Episode. 
######################################################################

use_cuda = torch.cuda.is_available()
print('use cuda : ', use_cuda)

class Plotter():
    def __init__(self):
        self.data = []

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_size, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, action_size.shape[0])

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
        
        self.envname = envname
        self.env = gym.make(envname)
        self.actor = Actor(self.env.observation_space.shape[0], self.env.action_space)
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
        action = torch.exp(log_probs).data.numpy()
        return action, log_probs.sum(dim=-1), value

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
        critic_loss = (advantage**2).mean()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
 
        actor_loss.backward(retain_graph=True)
        critic_loss.backward(retain_graph=True)

        self.actor_optimizer.step()
        self.critic_optimizer.step()


    def train(self):

        score = 0.0
        rewards = []
        log_probs = []
        values = []
        total_rewards = []

        print("Going to be training for a total of {} episodes".format(MAX_EPISODES))
        self.start_time = time.time()
        for e in range(MAX_EPISODES):
            state = self.env.reset()
            score = 0.0
            step_num = 0
            for t in range(MAX_STEPS_PER_EP):
                step_num += 1

                if RENDER_GAME and (e+1) % 25 ==0:
                    self.env.render()

                action, log_prob, value = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                score += reward
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                if done:
                    break

            total_rewards.append(score)

             # Update Actor - Critic 
            self.update_a2c(rewards, log_probs, values, state)

            if (e+1) % PRINT_DATA == 0:
                print("Episode: {}, reward: {}, steps: {}".format(e+1, total_rewards[e], step_num))

            if (e+1) % TEST_FREQUENCY == 0:
                print("-"*10 + " testing now " + "-"*10)
                mean_reward, std_reward = self.test(TEST_EPISODES,e)
                print('Mean Reward Achieved : {}, Standard Deviation : {}'.format(mean_reward, std_reward))
                print("-"*50)

        self.env.close()

    # TODO -- Finish test function to get average and std reward. 
    def test(self, num_episodes, train_episode):
        testing_rewards = []
        for e in range(TEST_EPISODES):
            state = self.env.reset()
            temp_reward = []
            for t in range(MAX_STEPS_PER_EP):
                action, _, _ = self.select_action(state)
                _, reward, done, _ = self.env.step(action)
                temp_reward.append(reward)
                if done:
                    break
            testing_rewards.append(reward)
        return np.mean(testing_rewards), np.std(testing_rewards)


    def demonstrate(self, save_snapshots=None):
        self.env = gym.make(self.envname)
        state = self.env.reset()
        while not done:
            self.env.render()
            action, log_prob, value = self.select_action(state)
            state, reward, done, _ = self.env.step(action)


    # TODO -- Figure out how to save. 
    def save_experiment(self, experiment_name):

        torch.save(self.actor,"experiments/" + experiments_name + "_actor")
        torch.save(self.critic,"experiments/" + experiments_name + "_critic")

        parameters = {
            "Environment Name": self.envname,
            "MAX_EPISODES":MAX_EPISODES,
            "MAX_STEPS_PER_EP":MAX_STEPS_PER_EP,
            "GAMMA":GAMMA,
            "TAU":TAU,
            "LEARNING_RATE_ACTOR":LR_ACTOR,
            "LEARNING_RATE_CRITIC":LR_CRITIC,
        }

        with open("experiments/" + experiment_name + ".csv", "w") as file:
            w = csv.writer(file)
            for key, val in parameters.items():
                w.writerow([key, val, "\n"])



if __name__ == "__main__":
    # A2C = A2C("MountainCarContinuous-v0")
    # A2C = A2C("Pendulum-v0")
    A2C = A2C("LunarLanderContinuous-v2")
    A2C.train()
