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
import os

import IPython

from memory import *
from actor_critic_networks import *
from noise_process import *
from read_params_from_file import *
# import sklearn.preprocessing

Sequence = namedtuple("Sequence", \
                ["state", "action", "reward", "next_state", "done"])


def adjust_mountain_reward(state, reward):
    # From https://medium.com/@ts1829/solving-mountain-car-with-q-learning-b77bf71b1de2
    reward = state[0] + 0.5 # Adjust reward based on car position
    if state[0] >= 0.5: # Adjust reward for task completion
        reward += 1

    return reward

    # # https://github.com/Pechckin/MountainCar/blob/master/MountainCarContinuous-v0.py
    # reward = reward + 100 * self.opt.gamma * (abs(next_state[1]) - abs(state[1]))

class DDPG:
    def __init__(self, opt):
        self.opt = opt
        self.start_time = time.time()
        self.training_timesteps = 0
        self.last_mean = 1E6
        self.last_var = 1E6
        self.update_params()
        self.reset()

    def update_params(self):
        self.parameters = {
            "Environment Name"            : opt.env_name,
            "MAX_EPISODES"                : self.opt.max_episodes,
            "MEM_SIZE"                    : self.opt.mem_size,
            "MEMORY_MIN"                  : self.opt.mem_min,
            "BATCH_SIZE"                  : self.opt.batch_size,
            "GAMMA"                       : self.opt.gamma,
            "TAU"                         : self.opt.tau,
            "LEARNING_RATE_ACTOR"         : self.opt.lr_actor,
            "LEARNING_RATE_CRITIC"        : self.opt.lr_critic,
            "OU_NOISE_THETA"              : self.opt.ou_noise_theta,
            "OU_NOISE_SIGMA"              : self.opt.ou_noise_sigma,
            "start time"                  : self.start_time,
            "L1_SIZE"                     : self.opt.l1_size,
            "L2_SIZE"                     : self.opt.l2_size,
            "OU_NOISE_SIGMA_DECAY_PER_EPS": self.opt.ou_noise_decay,
            "MIN_OU_NOISE_SIGMA"          : self.opt.min_ou_noise_sigma,
            "LastMeanError"               : self.last_mean,
            "LastVarError"                : self.last_var,
            "Training Timesteps"          : self.training_timesteps
            }


    def reset(self):
        self.envname = self.parameters["Environment Name"]
        self.env = gym.make(self.parameters["Environment Name"])
        self.env.reset()

        t = time.localtime()
        if not self.opt.load_from:
            self.name_suffix = "_" + self.env.spec.id[0:3] +"_"+ str(t.tm_mon) + "_" + str(t.tm_mday) + "_" + \
                    str(t.tm_hour) + "_" + str(t.tm_min)
        else:
            self.name_suffix = self.opt.load_from

        obs_size    = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]

        self.actor        = Actor(obs_size, self.env.action_space, self.opt.l1_size, self.opt.l2_size)
        self.critic       = Critic(obs_size, action_size, self.opt.l1_size, self.opt.l2_size)

        self.target_actor = Actor(obs_size, self.env.action_space, self.opt.l1_size, self.opt.l2_size)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic= Critic(obs_size, action_size, self.opt.l1_size, self.opt.l2_size)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), self.opt.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), self.opt.lr_critic, weight_decay=0.01)

        self.memory = Memory(self.opt.mem_size)

        self.start_time = time.time()
        self.solved = None
        self.training_timesteps = 0

        if opt.noise_type == "ou":
            self.noise = NoiseProcess(self.env.action_space,
                                        self.opt.ou_noise_theta,
                                        self.opt.ou_noise_sigma,
                                        self.opt.ou_noise_decay,
                                        self.opt.min_ou_noise_sigma)
        elif opt.noise_type == "normal":
            self.noise = NormalNoiseProcess(self.env.action_space,
                                             self.opt.normal_noise_var,
                                             self.opt.normal_noise_decay,
                                             self.opt.min_normal_noise)
        else:
            raise("Invalid noise type provided")

        self.folder_name = opt.exp_name + self.name_suffix
        # self.initialize_scale_state()

    # def initialize_scale_state(self):
    #     state_space_samples = np.array([self.env.observation_space.sample() for x in range(int(1E4))])
    #     self.scaler = sklearn.preprocessing.StandardScaler().fit(state_space_samples)

    # def scale_state(self, state):
    #     scaled = self.scaler.transform([state])
    #     return scaled[0]

    def fill_memory(self):
        steps = 0
        while steps < self.opt.mem_min:
            state = self.env.reset()
            done = False

            ep_steps = 0
            while not done and ep_steps < self.env._max_episode_steps:
                ep_steps += 1
                noise_to_add = self.noise.sample()
                action = self.actor.take_action(state, noise_to_add)
                next_state, reward, done, _ = self.env.step(action)

                # Reward shaping for mountain car
                if "mountain" in self.env.spec.id.lower():
                    reward = adjust_mountain_reward(state, reward)

                self.memory.push( \
                    Sequence(state, action, reward, next_state, done))

                state = next_state
                steps += 1

    # main training loop
    def train(self):
        print("Starting job: \n", self.parameters)

        episode_scores = []
        actor_loss, critic_loss = torch.tensor(1E6), torch.tensor(1E6)

        self.fill_memory()
        self.training_timesteps = 0

        for episode_num in range(self.opt.max_episodes):

            # state = self.scale_state(self.env.reset())
            state = self.env.reset()
            done = False
            step_scores = []

            ep_steps = 0
            while not done and ep_steps < self.env._max_episode_steps:
                ep_steps += 1
                self.training_timesteps += 1
                noise_to_add = self.noise.sample()
                action = self.actor.take_action(state, noise_to_add)
                next_state, reward, done, _ = self.env.step(action)

                # Referenced from Reinforcement Learning Cookbook
                # next_state = self.scale_state(next_state)

                # Reward shaping for mountain car
                if "mountain" in self.env.spec.id.lower():
                    reward = adjust_mountain_reward(state, reward)

                step_scores.append(float(reward))

                self.memory.push( \
                    Sequence(state, action, reward, next_state, done))

                state = next_state

                if self.memory.max_entry > self.opt.mem_min:
                    actor_loss, critic_loss = self.update_networks()

            episode_scores.append(sum(step_scores))
            self.noise.decay()

            if episode_num % self.opt.print_data == 0:
                average_episode_score = sum(episode_scores[-self.opt.print_data:])/float(self.opt.print_data)
                print("\nEpisode: ", episode_num, " / ", self.opt.max_episodes,
                      " | Avg Score: ",
                      np.array(average_episode_score).round(4),
                      " | Elapsed time [s]: ",
                      round((time.time() - self.start_time), 2),
                      )
                print("Actor loss: ", actor_loss.detach().numpy().round(4).item(),
                        "critic_loss: ", critic_loss.detach().numpy().round(4).item())
                print("Noise sigma: ", self.noise.sigma)

            if episode_num % self.opt.save_freq == 0:
                print("\nAverage metric at iteration ", episode_num)
                average, variance = self.compute_average_metric()
                self.save_experiment("eps_"+str(episode_num) + "_of_"+str(self.opt.max_episodes))
                self.check_if_solved(average, episode_num)

                if "mountain" in self.env.spec.id.lower() and abs(average) < 1E-12:
                    return False

        print("Finished training. Training time: ",
                    round((time.time() - self.start_time), 2) )
        print("Episode Scores: \n", episode_scores)
        self.env.close()
        ddpg.save_experiment("eps_"+str(episode_num) + "_of_"+str(self.opt.max_episodes),
                                save_critic=True)

        return True

    def check_if_solved(self, average, episode_num):
        if "mountaincar" in self.env.spec.id.lower() and average > 90.0:
            if self.solved is None:
                self.solved = episode_num
                print(self.env.spec.id, "solved after ", self.solved)

        elif "lunarlander" in self.env.spec.id.lower() and average > 200.0:
            if self.solved is None:
                self.solved = episode_num
                print(self.env.spec.id, "solved after ", self.solved)

    # mini-batch sample and update networks
    def update_networks(self):
        batch = self.memory.sample(self.opt.batch_size)

        target_q = batch.reward + self.opt.gamma * torch.mul(\
                            self.target_critic(batch.next_state,
                                self.target_actor(batch.next_state)), (~batch.done).float()).detach()

        self.critic_optimizer.zero_grad()
        critic_q = self.critic(batch.state, batch.action)
        critic_loss = F.mse_loss(critic_q, target_q)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_a = self.actor(batch.state)
        actor_loss = -self.critic(batch.state, actor_a).mean() # gradient ascent for highest Q value
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft update
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.opt.tau * param.data + (1 - self.opt.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.opt.tau * param.data + (1 - self.opt.tau) * target_param.data)

        return actor_loss, critic_loss

    def compute_average_metric(self):
        num_to_test = 10
        rewards = np.zeros(num_to_test)

        for demo_ind in range(num_to_test):
            rewards[demo_ind] = self.demonstrate(render=False)

        print("Evaluation over ", num_to_test, "episodes.\n\t",
                                " Mean: ", rewards.mean(),
                                " | Variance: ", rewards.var())
        self.last_mean = rewards.mean()
        self.last_var = rewards.var()

        return rewards.mean(), rewards.var()

    def demonstrate(self, render=True):
        state = self.env.reset()
        # state = self.scale_state(self.env.reset())
        done = False
        rewards = 0.0

        ep_steps = 0
        while not done and ep_steps < self.env._max_episode_steps:
            ep_steps += 1
            if render:
                self.env.render()
            action = self.actor.take_action(state, None)
            next_state, reward, done, _ = self.env.step(action)
            # next_state = self.scale_state(next_state)
            rewards += reward # this will be the OpenAI defined reward

            state = next_state

        self.env.reset()
        return rewards

    def save_experiment(self, experiment_name, save_critic=False):
        self.update_params()
        experiment_name = experiment_name + "_" + self.opt.exp_name + self.name_suffix

        if self.folder_name not in os.listdir("experiments/"):
            os.mkdir("experiments/" + self.folder_name)
            print("made directory: ")
        save_location = "experiments/" + self.folder_name + "/" + experiment_name

        torch.save(self.actor.state_dict(), save_location + "actor")
        if save_critic:
            torch.save(self.critic.state_dict(), save_location + "critic")

        with open(save_location  + ".csv", "w") as file:
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


if __name__ == "__main__":
    ## Parameters ##
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name"          , type=str, default="LunarLanderContinuous-v2", help="Environment name")
    parser.add_argument("--exp_name"          , type=str, default="experiment_", help="Experiment name")
    parser.add_argument("--load_from"         , type=str, default="", help="Train the networks or just load a file")

    parser.add_argument("--max_episodes"      , type=int, default=50000, help="total number of episodes to train")
    parser.add_argument("--mem_min"           , type=int, default=int(2E3), help="minimum size of replay memory before updating actor and critic networks")
    parser.add_argument("--mem_size"          , type=int, default=int(1E6), help="total size of replay memory")
    parser.add_argument("--batch_size"        , type=int, default=64, help="batch size when sampling from replay memory")

    parser.add_argument("--gamma"             , type=float, default=0.99, help="discount factor for future rewards")
    parser.add_argument("--tau"               , type=float, default=0.001, help="tau averaging for target network updating")

    parser.add_argument("--lr_actor"          , type=float, default=1E-4, help="")
    parser.add_argument("--lr_critic"         , type=float, default=1E-3, help="")
    parser.add_argument("--l1_size"           , type=int, default=400, help="")
    parser.add_argument("--l2_size"           , type=int, default=300, help="")

    parser.add_argument("--noise_type"        , type=str, default="ou", help="")

    parser.add_argument("--ou_noise_theta"    , type=float, default=0.15, help="")
    parser.add_argument("--ou_noise_sigma"    , type=float, default=0.2, help="")
    parser.add_argument("--ou_noise_decay"    , type=float, default=0.0, help="")
    parser.add_argument("--min_ou_noise_sigma", type=float, default=0.15, help="")

    parser.add_argument("--normal_noise_var"  , type=float, default=0.2, help="")
    parser.add_argument("--normal_noise_decay", type=float, default=0.0, help="")
    parser.add_argument("--min_normal_noise"  , type=float, default=0.2, help="")

    parser.add_argument("--save_freq", type=int, default=5000, help="")
    parser.add_argument("--print_data", type=int, default=50, help="")
    parser.add_argument("--demonstrate_freq", type=int, default=5000, help="")

    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is ", device)

    ddpg = DDPG(opt)

    if not opt.load_from:
        ddpg.train()

    else:
        # Use this to save or load networks. Assumes you are loading from experiments/ subdirectory.
        # Example Usage:
        # $ python ddpg.py --load_from quicklunarlander/finished_quick0_quick_Lun_11_28_15_27
        ddpg.load_experiment()
        IPython.embed()
