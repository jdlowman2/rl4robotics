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
from pathlib import Path

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
        self.mean   = np.zeros(action_shape)

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
    def __init__(self, opt):
        self.opt = opt
        self.start_time = time.time()
        self.parameters = {
            "Environment Name"            : opt.env_name,
            "MAX_EPISODES"                : MAX_EPISODES,
            "MEM_SIZE"                    : MEM_SIZE,
            "MEMORY_MIN"                  : MEMORY_MIN,
            "BATCH_SIZE"                  : BATCH_SIZE,
            "GAMMA"                       : GAMMA,
            "TAU"                         : TAU,
            "LEARNING_RATE_ACTOR"         : LEARNING_RATE_ACTOR,
            "LEARNING_RATE_CRITIC"        : LEARNING_RATE_CRITIC,
            "OU_NOISE_THETA"              : OU_NOISE_THETA,
            "OU_NOISE_SIGMA"              : OU_NOISE_SIGMA,
            "start time"                  : self.start_time,
            "L1_SIZE"                     : L1_SIZE,
            "L2_SIZE"                     : L2_SIZE,
            "FILL_MEM_SIZE"               : FILL_MEM_SIZE,
            "OU_NOISE_SIGMA_DECAY_PER_EPS":OU_NOISE_SIGMA_DECAY_PER_EPS,
            "MIN_OU_NOISE_SIGMA"          : MIN_OU_NOISE_SIGMA,
            "LastMeanError"               : 1E6,
            "LastVarError"                : 1E6,
            }

        self.reset()

    def reset(self):
        self.envname = self.parameters["Environment Name"]
        self.env = gym.make(self.envname)
        self.env.reset()

        t = time.localtime()
        if not self.opt.load_from:
            self.name_suffix = "_" + self.envname[0:3] +"_"+ str(t.tm_mon) + "_" + str(t.tm_mday) + "_" + \
                    str(t.tm_hour) + "_" + str(t.tm_min)
        else:
            self.name_suffix = self.opt.load_from

        obs_size    = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]

        self.actor        = Actor(obs_size, self.env.action_space, L1_SIZE, L2_SIZE)
        self.critic       = Critic(obs_size, action_size, L1_SIZE, L2_SIZE)

        self.target_actor = Actor(obs_size, self.env.action_space, L1_SIZE, L2_SIZE)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic= Critic(obs_size, action_size, L1_SIZE, L2_SIZE)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), LEARNING_RATE_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), LEARNING_RATE_CRITIC)

        self.memory = Memory(MEM_SIZE)

        self.start_time = time.time()
        self.solved = None

    def random_fill_memory(self, num_eps):
        for episode in range(num_eps):
            state = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)

                self.memory.push( \
                    Sequence(state, action, reward, next_state, done))
                state = next_state

        print("Number of sequences added to memory: ", self.memory.max_entry)

    # main training loop
    def train(self):
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
                print("\nAverage metric at iteration ", episode_num)
                average, variance = self.compute_average_metric()
                self.save_experiment("eps_"+str(episode_num) + "_of_"+str(MAX_EPISODES))
                self.check_if_solved(average, episode_num)

        print("Finished training. Training time: ",
                    round((time.time() - self.start_time), 2) )
        print("Episode Scores: \n", episode_scores)
        self.env.close()

    def check_if_solved(self, average, episode_num):
        if "mountaincar" in self.envname.lower() and average > 90.0:
            if self.solved is None:
                self.solved = episode_num
                print(self.envname, "solved after ", self.solved)

        elif "lunarlander" in self.envname.lower() and average > 200.0:
            if self.solved is None:
                self.solved = episode_num
                print(self.envname, "solved after ", self.solved)

    # mini-batch sample and update networks
    def update_networks(self):
        batch = self.memory.sample(BATCH_SIZE)

        # modes for batch norm layers
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actor_a       = self.target_actor(batch.next_state) # This should be batch.next_state
        critic_q             = self.critic(batch.state, batch.action)
        target_critic_next_q = self.target_critic(batch.next_state, target_actor_a)

        target_q = batch.reward + GAMMA * torch.mul(target_critic_next_q, (~batch.done).float())

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

        print("Evaluation over ", num_to_test, "episodes.\n\t",
                                " Mean: ", rewards.mean(),
                                " | Variance: ", rewards.var())
        self.parameters["LastMeanError"] = rewards.mean()
        self.parameters["LastVarError"] = rewards.var()

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

    def save_experiment(self, experiment_name=None):
        experiment_name = experiment_name + "_" + self.opt.exp_name + self.name_suffix

        torch.save(self.actor.state_dict(), "experiments/" + experiment_name + "actor")
        torch.save(self.critic.state_dict(), "experiments/" + experiment_name + "critic")

        with open("experiments/" + experiment_name + ".csv", "w") as file:
            w = csv.writer(file)
            for key, val in self.parameters.items():
                w.writerow([key, val, "\n"])

    def load_experiment(self):
        # NOTE: this does not load the global training parameters, so you
        # can't continue training

        self.params = read_params_from_file(self.opt)
        self.reset()
        actor_file = Path("experiments/" + self.opt.load_from + "actor")
        self.actor.load_state_dict(torch.load(actor_file))
        critic_file = Path("experiments/" + self.opt.load_from + "critic")
        self.critic.load_state_dict(torch.load(critic_file))

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())


def read_params_from_file(opt):
    load_from = opt.load_from
    print("Filename: \n\t", "experiments/" + load_from + ".csv")
    with open("experiments/" + load_from + ".csv", "r") as file:
        reader = csv.reader(file)
        params = dict()
        for row in reader:
            try:
                params[row[0]] = int(row[1])
            except ValueError:
                try:
                    params[row[0]] = float(row[1])
                except ValueError:
                    params[row[0]] = row[1]

        return params


if __name__ == "__main__":
    ## Parameters ##
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name"          , type=str, default="LunarLanderContinuous-v2", help="Environment name")
    parser.add_argument("--exp_name"          , type=str, default="experiment_", help="Experiment name")
    parser.add_argument("--load_from"         , type=str, default="", help="Train the networks or just load a file")

    parser.add_argument("--max_episodes"      , type=int, default=50000, help="total number of episodes to train")
    parser.add_argument("--mem_min"           , type=int, default=int(2E3), help="minimum size of replay memory before updating actor and critic networks")
    parser.add_argument("--mem_size"          , type=int, default=int(1E6), help="total size of replay memory")
    parser.add_argument("--fill_mem_size"     , type=int, default=100, help="number of episodes to run with random action to fill replay memory")
    parser.add_argument("--batch_size"        , type=int, default=64, help="batch size when sampling from replay memory")

    parser.add_argument("--gamma"             , type=float, default=0.99, help="discount factor for future rewards")
    parser.add_argument("--tau"               , type=float, default=0.001, help="tau averaging for target network updating")

    parser.add_argument("--lr_actor"          , type=float, default=1E-4, help="")
    parser.add_argument("--lr_critic"         , type=float, default=1E-3, help="")
    parser.add_argument("--l1_size"           , type=int, default=400, help="")
    parser.add_argument("--l2_size"           , type=int, default=300, help="")

    parser.add_argument("--ou_noise_theta"    , type=float, default=0.15, help="")
    parser.add_argument("--ou_noise_sigma"    , type=float, default=0.2, help="")
    parser.add_argument("--ou_noise_decay"    , type=float, default=0.0, help="")
    parser.add_argument("--min_ou_noise_sigma", type=float, default=0.15, help="")

    parser.add_argument("--save_freq", type=int, default=5000, help="")

    opt = parser.parse_args()
    print(opt)

    MAX_EPISODES                = opt.max_episodes
    MEM_SIZE                    = opt.mem_size
    MEMORY_MIN                  = opt.mem_min
    FILL_MEM_SIZE               = opt.fill_mem_size
    BATCH_SIZE                  = opt.batch_size

    GAMMA                       = opt.gamma
    TAU                         = opt.tau

    LEARNING_RATE_ACTOR         = opt.lr_actor
    LEARNING_RATE_CRITIC        = opt.lr_critic
    L1_SIZE                     = opt.l1_size
    L2_SIZE                     = opt.l2_size

    OU_NOISE_THETA              = opt.ou_noise_theta
    OU_NOISE_SIGMA              = opt.ou_noise_sigma
    OU_NOISE_SIGMA_DECAY_PER_EPS= opt.ou_noise_decay
    MIN_OU_NOISE_SIGMA          = opt.min_ou_noise_sigma

    PRINT_DATA                  = 50  # how often to print data
    SAVE_FREQ                   = opt.save_freq # How often to save networks
    DEMONSTRATE_INTERVAL        = 100000*PRINT_DATA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is ", device)

    ddpg = DDPG(opt)

    if not opt.load_from:
        ddpg.train()
        ddpg.save_experiment("finished_" + opt.exp_name + str(exp_num))

    else:
        # Use this to save or load networks. Assumes you are loading from experiments/ subdirectory.
        # Example Usage:
        # $ python ddpg.py --load_from quicklunarlander/finished_quick0_quick_Lun_11_28_15_27
        ddpg.load_experiment()
        IPython.embed()
