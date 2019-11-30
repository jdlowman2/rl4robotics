import sklearn.preprocessing
import numpy as np
import random
import time
import gym

from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

import sys, time

# LOL - Technically it works if you let it run for 1000 episodes, I'll look at it 
# more later today. 

############################# PARAMETERS #############################
MAX_EPISODES = 10000
MAX_STEPS_PER_EP = 1000
TEST_FREQUENCY = 100
TEST_EPISODES = 100
GAMMA = 0.9           # discount factor
LR = 1E-3             # Learning Rate
N_HIDDEN = 128
PRINT_DATA = 1        # how often to print data
RENDER_GAME = False   # View the Episode. 

ENVIRONMENT = "MountainCarContinuous-v0"
# ENVIRONMENT = "Pendulum-v0"
# ENVIRONMENT = "LunarLanderContinuous-v2"
######################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device: ", device)

class Plotter():
    def __init__(self):
        self.data = []

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.action_size = action_size
        self.layer1 = nn.Linear(state_size, N_HIDDEN)
        self.layer2 = nn.Linear(N_HIDDEN, N_HIDDEN)
        self.layer3 = nn.Linear(N_HIDDEN, action_size)
        self.value = nn.Linear(N_HIDDEN, 1)
        self.to(device)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        mu = 2 * torch.tanh(self.layer3(x))
        sigma = F.softplus(self.layer3(x)) + 1E-5
        n_output = self.action_size
        distribution = torch.distributions.Normal(mu.view(self.action_size,).data, sigma.view(self.action_size,).data)
        value = self.value(x)
        return distribution, value


class A2C:
    def __init__(self, envname):
        
        self.envname = envname
        self.env = gym.make(envname)
        self.model = ActorCritic(self.env.observation_space.shape[0], self.env.action_space.shape[0]).to(device)
        self.optimizer = optim.Adam(self.model.parameters(),LR)

        self.data = {"loss": []}
        self.start_time = None

    # Normalize / Standardize the inputs for faster model convergence. 
    # Randomly generate oberservations and use them to train a scaler. 
    # Referenced from - Reinforcement Learning Cookbook. 
    def initialize_scale_state(self):
        state_space_samples = np.array([self.env.observation_space.sample() for x in range(int(1E4))])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(state_space_samples)

    def scale_state(self, state):
        scaled = self.scaler.transform([state])
        return scaled[0]

    def select_action(self, state):
        dist, value = self.model(torch.Tensor(state)) 
        action = dist.sample().numpy()
        log_prob = dist.log_prob(action[0])
        return action, log_prob, value

    def update_a2c(self, rewards, log_probs, values, state):

        Qvals = []
        Qval = 0
        pw = 0
        for reward in rewards[::-1]:
            Qval += GAMMA ** pw * reward
            pw += 1
            Qvals.append(Qval)

        Qvals = Qvals[::-1]
        Qvals = torch.tensor(Qvals)
        Qvals = (Qvals - Qvals.mean()) / (Qvals.std() + 1e-9)

        loss = 0
        for log_prob, value, Qval in zip(log_probs, values, Qvals):

            advantage = Qval - value.item()
            actor_loss = -log_prob * advantage
            critic_loss = F.smooth_l1_loss(value[0], Qval)
            loss += critic_loss + actor_loss

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)  # I'm still dealing with this issue, it slows me down for . 
        self.optimizer.step()


    # Main training loop.
    def train(self):

        score = 0.0
        rewards = []
        log_probs = []
        values = []
        total_rewards = []
        self.initialize_scale_state()

        print("Going to be training for a total of {} episodes".format(MAX_EPISODES))
        self.start_time = time.time()
        for e in range(MAX_EPISODES):
            state = self.env.reset()
            score = 0.0
            step_num = 0

            # To improve exploration, take inital actions sampled from random distirbution. 
            action = torch.tensor([[2 * random.random() - 1]])
            state, reward, done, _ = self.env.step(action)

            for t in range(MAX_STEPS_PER_EP):
                
                step_num += 1

                if RENDER_GAME and (e+1) % 25 ==0:
                    self.env.render()

                state = self.scale_state(state)
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

            # if (e+1) % TEST_FREQUENCY == 0:
            #     print("-"*10 + " testing now " + "-"*10)
            #     mean_reward, std_reward = self.test(TEST_EPISODES,e)
            #     print('Mean Reward Achieved : {}, Standard Deviation : {}'.format(mean_reward, std_reward))
            #     print("-"*50)

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

        actor_path = "experiments/" + experiments_name + "_actor"
        critic_path = "experiments/" + experiments_name + "_critic"

        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

        # if you want to load the model, use something similar to the following
        # network = actor()
        # actor.load_state_dict(torch.load(file_path))

        parameters = {
            "Environment Name": self.envname,
            "MAX_EPISODES":MAX_EPISODES,
            "MAX_STEPS_PER_EP":MAX_STEPS_PER_EP,
            "GAMMA":GAMMA,
            "TAU":TAU,
            "LEARNING_RATE_ACTOR":LR_ACTOR,
            "LEARNING_RATE_CRITIC":LR_CRITIC,
        }

        parameters_path = "experiments/" + experiment_name + ".csv"
        with open(parameters_path, "w") as file:
            w = csv.writer(file)
            for key, val in parameters.items():
                w.writerow([key, val, "\n"])



if __name__ == "__main__":
    A2C = A2C(ENVIRONMENT)
    A2C.train()
