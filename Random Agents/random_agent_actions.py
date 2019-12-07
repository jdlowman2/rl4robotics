"""
Random agent numpy array
Every 10 Episodes for 50,000 episodes
Save to numpy array

Plot of
    Training rewards
    Test rewards
    Random agent rewards
"""

import gym
import numpy as np
from pathlib import Path
import torch
from matplotlib import pyplot as plt
import argparse


def save_rewards(num_intervals):
    environments = {"pen": "Pendulum-v0",
                    "lun": "LunarLanderContinuous-v2",
                    "mou": "MountainCarContinuous-v0",
                    "bip": "BipedalWalker-v2",
                    }

    # Change these values----------------
    save_intervals = list(range(0, num_intervals, 10))
    test_episodes_per_interval = 25 # number of test episodes per interval. The mean and stdev will be computed over these
    #------------------------------------

    for envname in list(environments.values()):
        env = gym.make(envname)
        random_agent_rewards = np.zeros((len(save_intervals), test_episodes_per_interval))

        for save_ind, save_num in enumerate(save_intervals):
            for ind, episode in enumerate(list(range(test_episodes_per_interval))):
                state = env.reset()
                done = False
                total_reward = 0.0

                while not done:
                    action = env.action_space.sample()
                    state, r, done, _ = env.step(action)
                    total_reward += r

                random_agent_rewards[save_ind, ind] = total_reward

            print("\nTest across ", test_episodes_per_interval, "after training for ", save_num, "episodes.")
            print("\tMean: ", random_agent_rewards[save_ind, :].mean(), " | Stdev: ", np.std(random_agent_rewards[save_ind, :]))

        env.close()

        save_filename = envname.split("-")[0] + "_random_agent"
        np.save(save_filename, random_agent_rewards)


def load_random_rewards(filename):
    random_agent_rewards = np.load(filename)
    mean = random_agent_rewards.mean(axis=1)
    stdevs = random_agent_rewards.std(axis=1)

    info = {"data": random_agent_rewards, "mean": mean, "stdevs": stdevs}

    return info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_or_save", type=str, default="")
    parser.add_argument("--load_filename", type=str, default="")
    parser.add_argument("--num_intervals", type=int, default=25000)
    opt = parser.parse_args()

    if "load" in opt.load_or_save.lower():
        load_random_rewards(opt.load_filename)
    else:
        save_rewards(opt.num_intervals)

