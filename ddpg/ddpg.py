import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import time
import csv
import sys

import argparse

import IPython

from memory import *
from actor_critic_networks import *
# from plotter import *


Sequence = namedtuple("Sequence", \
                ["state", "action", "reward", "next_state", "done"])


# referenced from github.com/minimalrl
class NoiseProcess:
    def __init__(self, action_shape):
        self.theta = OU_NOISE_THETA
        self.sigma = OU_NOISE_SIGMA
        self.sigma_decay = OU_NOISE_SIGMA_DECAY_PER_EPS
        self.min_sigma = MIN_OU_NOISE_SIGMA
        self.dt = 0.01
        self.prev_x = np.zeros(action_shape)
        self.mean = np.zeros(action_shape)

    def sample(self):
        x = self.prev_x + self.theta * self.dt * (self.mean - self.prev_x) + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)

        self.prev_x = x
        return x

    def decay(self):
        self.sigma = max(self.min_sigma, self.sigma - self.sigma_decay)


# Run the agent 10 times every 100 episodes
    # compute mean and standard dev across these iterations
# Total reward during training (moving average)
# Average episode length during training (moving average)
# Speed of convergence:  how many episodes to get to "success reward" amount (defined for each environment.) (moving average)
# Initialize algorithm 100 times and compare how many times it succeeds

class DDPG:
    def __init__(self, envname):
        self.envname = envname
        self.env = gym.make(envname)
        self.reset()

        self.parameters = {
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
            "L1_SIZE": L1_SIZE,
            "L2_SIZE": L2_SIZE,
            "FILL_MEM_SIZE": FILL_MEM_SIZE,
            "OU_NOISE_SIGMA_DECAY_PER_EPS":OU_NOISE_SIGMA_DECAY_PER_EPS,
            "MIN_OU_NOISE_SIGMA": MIN_OU_NOISE_SIGMA,
            }

    def reset(self):
        obs_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]

        self.env.reset()

        self.actor = Actor(obs_size, self.env.action_space, L1_SIZE, L2_SIZE)
        self.critic = Critic(obs_size, action_size, L1_SIZE, L2_SIZE)

        self.target_actor = Actor(obs_size, self.env.action_space, L1_SIZE, L2_SIZE)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic = Critic(obs_size, action_size, L1_SIZE, L2_SIZE)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), LEARNING_RATE_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), LEARNING_RATE_CRITIC)

        self.memory = Memory(MEM_SIZE)

        self.data = {"loss": []}
        self.start_time = None

    def random_fill_memory(self, num_eps):
        state = self.env.reset()
        done = False

        for episode in range(num_eps):
            for t in range(20):
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)

                if "mountain" in self.envname:
                    reward = next_state[0] + 0.5
                    if next_state[0] >= 0.5:
                        reward += 1

                self.memory.push( \
                    Sequence(state, action, reward, next_state, done))
                state = next_state

        print("Number of sequences added to memory: ", self.memory.max_entry)

    # main training loop
    def train(self):
        self.start_time = time.time()
        print("Starting job: \n", self.parameters)

        self.random_fill_memory(FILL_MEM_SIZE) # get some random transitions for the memory buffer
        episode_scores = []
        actor_loss, critic_loss = torch.tensor(1E6), torch.tensor(1E6)

        for episode_num in range(MAX_EPISODES):
            noise = NoiseProcess(self.env.action_space.shape)

            state = self.env.reset()
            done = False
            step_scores = []

            while not done:
                noise_to_add = noise.sample()
                action = self.actor.take_action(state, noise_to_add)
                next_state, reward, done, _ = self.env.step(action)

                # if "mountain" in self.envname:
                #     reward = next_state[0] + 0.5
                #     if next_state[0] >= 0.5:
                #         reward += 1

                self.memory.push( \
                    Sequence(state, action, reward, next_state, done))

                step_scores.append(float(reward))
                state = next_state

                if self.memory.max_entry > MEMORY_MIN:
                    actor_loss, critic_loss = self.update_networks()

            episode_scores.append(sum(step_scores))
            noise.decay()

            if episode_num % PRINT_DATA == 0:
                average_episode_score = sum(episode_scores[-PRINT_DATA:])/float(PRINT_DATA)
                print("\nEpisode: ", episode_num, " / ", MAX_EPISODES,
                      " | Avg Score: ",
                      np.array(average_episode_score).round(4),
                      " | Elapsed time [s]: ",
                      round((time.time() - self.start_time), 2),
                      )
                print("Actor loss: ", actor_loss.detach().numpy().round(4).item(),
                        "critic_loss: ", critic_loss.detach().numpy().round(4).item())
                print("Noise sigma: ", noise.sigma)

            if episode_num % SAVE_FREQ == 0:
                self.save_experiment("eps_"+str(episode_num) + "_of_"+str(MAX_EPISODES))
                average, variance = self.compute_average_metric()
                print("\nAverage metric at iteration ", episode_num)
                print("Average: ", average, " variance: ", variance)

        print("Finished training. Training time: ",
                    round((time.time() - self.start_time), 2) )
        print("Episode Scores: \n", episode_scores)
        self.env.close()

    # mini-batch sample and update networks
    def update_networks(self):
        batch = self.memory.sample(BATCH_SIZE)

        # modes for batch norm layers
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actor_a       = self.target_actor(batch.state)
        critic_q             = self.critic(batch.state, batch.action)
        target_critic_next_q = self.target_critic(batch.next_state, target_actor_a)

        target_q = batch.reward + GAMMA * torch.mul(target_critic_next_q, batch.done.float())

        # update critic network
        self.critic.train()
        self.critic_optimizer.zero_grad()
        critic_loss = F.mse_loss(critic_q, target_q)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.eval()
        self.actor_optimizer.zero_grad()
        actor_a = self.actor(batch.state)
        self.actor.train()
        actor_loss = -self.critic(batch.state, actor_a).mean() # gradient ascent for highest Q value
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor.eval()

        # soft update
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        return actor_loss, critic_loss

    def compute_average_metric(self):
        num_to_test = 10
        rewards = np.zeros(num_to_test)

        for demo_ind in range(num_to_test):
            rewards[demo_ind] = self.demonstrate(render=False)

        print("Evaluation over ", num_to_test, "episodes.",
                                " Mean: ", rewards.mean(),
                                " | Variance: ", rewards.var())

        return rewards.mean(), rewards.var()

    def demonstrate(self, render=True):
        state = self.env.reset()
        done = False
        rewards = 0.0

        while not done:
            if render:
                self.env.render()
            action = self.actor.take_action(state, None)
            state, reward, done, _ = self.env.step(action)
            rewards += reward

        self.env.reset()
        return rewards

    def save_experiment(self, experiment_name="experiment"):
        t = time.localtime()
        suffix = "_" + self.envname[0:3] +"_"+ str(t.tm_mon) + "_" + str(t.tm_mday) + "_" + \
                str(t.tm_hour) + "_" + str(t.tm_min)
        experiment_name = experiment_name + suffix

        torch.save(self.actor.state_dict(), "experiments/" + experiment_name + "_actor")
        torch.save(self.critic.state_dict(), "experiments/" + experiment_name + "_critic")

        with open("experiments/" + experiment_name + ".csv", "w") as file:
            w = csv.writer(file)
            for key, val in self.parameters.items():
                w.writerow([key, val, "\n"])

        # self.plotter.fig.savefig("experiments/" + experiment_name + \
        #     "_training_curve" + ".png", dpi=200)

    def load_experiment(self, experiment_name):
        # NOTE: this does not load the global training parameters, so you
        # can't continue training
        # actor_file = "experiments/" + experiment_name + "_actor"
        # self.actor.load_state_dict(torch.load(actor_file))
        critic_file = "experiments/" + experiment_name + "_critic"
        self.critic.load_state_dict(torch.load(critic_file))

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())


## Parameters ##
MAX_EPISODES = 50000
MAX_STEPS_PER_EP = 300
MEM_SIZE = int(1E6)
MEMORY_MIN = int(2E3)
BATCH_SIZE = 64
GAMMA = 0.99    # discount factor
TAU = 0.001     # tau averaging for target network updating
LEARNING_RATE_ACTOR = 1E-4
LEARNING_RATE_CRITIC = 1E-3

L1_SIZE = 400
L2_SIZE = 300
FILL_MEM_SIZE = 20

OU_NOISE_THETA = 0.15
OU_NOISE_SIGMA = 0.2 # sigma decaying over time?
OU_NOISE_SIGMA_DECAY_PER_EPS = 0.0 #OU_NOISE_SIGMA / MAX_EPISODES # reach .5 Sigma at halfway
MIN_OU_NOISE_SIGMA = OU_NOISE_SIGMA

PRINT_DATA = 100  # how often to print data
SAVE_FREQ = int(round(MAX_EPISODES/10)) # How often to save networks
DEMONSTRATE_INTERVAL = 100000*PRINT_DATA

def run_batch_experiments(num_exp_per_env, envnames):
    for envname in envnames:
        for exp_num in range(num_exp_per_env):
            ddpg = DDPG(envname)
            ddpg.train()
            ddpg.save_experiment("experiment_"+str(exp_num))
            plt.close("all")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="LunarLanderContinuous-v2", help="Environment name")
    parser.add_argument("--max_episodes", type=int, default=MAX_EPISODES, help="total number of episodes to train")
    opt = parser.parse_args()
    print(opt)

    MAX_EPISODES = opt.max_episodes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is ", device)
    # # ddpg = DDPG("MountainCarContinuous-v0")
    # # ddpg = DDPG("LunarLanderContinuous-v2")
    # ddpg = DDPG("Pendulum-v0")
    run_batch_experiments(1, [opt.env_name])
    # IPython.embed() # use this to save or load networks
