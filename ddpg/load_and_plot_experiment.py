import gym
from actor_critic_networks import Actor
import numpy as np
from pathlib import Path
import torch
from matplotlib import pyplot as plt
import os

import IPython

# Change these values----------------
foldername = "experiments/" + "Lander Converge/"
save_intervals = list(range(0, 25001, 1000))
render = False
#------------------------------------

environments = {"pen": "Pendulum-v0",
                    "lun": "LunarLanderContinuous-v2",
                    "mou": "MountainCarContinuous-v0",
                    }
save_filenames = [file for file in os.listdir(foldername) if "actor" in file]

# assert(len(save_intervals) == len(save_filenames))
file = save_filenames[0].split('_')
total_num = file[3]
exp_name = file[4]
envname = environments[file[5].lower()]
start_time = "_".join(file[-4:]).strip("actor")

episodes_per_file = 100 # number of episodes to run. The mean and stdev will be computed over these
L1_SIZE = 400
L2_SIZE = 300

env = gym.make(envname)

save_num_rewards = np.zeros((episodes_per_file, len(save_intervals)))

for save_ind, save_num in enumerate(save_intervals):
    experiment_path = Path(foldername + "eps_"+\
                        str(save_num) + "_of_"+total_num+\
                        "_" + exp_name + "_" + envname[:3] +"_"+\
                        start_time + "actor")
    print("Loading from ", experiment_path)

    actor = Actor(env.observation_space.shape[0],
                        env.action_space, L1_SIZE, L2_SIZE)

    actor.load_state_dict(torch.load(experiment_path))

    for ind, episode in enumerate(list(range(episodes_per_file))):
        # print("Reset")
        state = env.reset()
        done = False
        rewards = 0.0

        while not done:
            if render:
                env.render()
            action = actor.take_action(state, None)
            state, reward, done, _ = env.step(action)
            rewards += reward
            # print("reward: ", reward, " action: ", action)

        save_num_rewards[ind, save_ind] = rewards

    assert(np.nonzero(save_num_rewards[:, save_ind])[0].shape[0] == save_num_rewards[:, save_ind].shape[0])

    print("\nTest across ", episodes_per_file, "after training for ", save_num, "episodes.")
    print("\tMean: ", save_num_rewards[:, save_ind].mean(), " | Stdev: ", np.std(save_num_rewards[:, save_ind]))
    print("\n")

env.close()

means = [row.mean() for row in save_num_rewards.T]
stdevs = [np.std(row) for row in save_num_rewards.T]

plt.errorbar(save_intervals, means, yerr=stdevs,
            label=envname, fmt="-o", linewidth=2.0, ecolor="red", capsize=10.0)
plt.xlabel("Number of Training Episodes")
plt.ylabel("Rewards Per Episode for 100 Episodes")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig(foldername +  foldername.split("/")[-2].lower().replace(" ", "_") + "_training_eval"+ \
        ".png", dpi=200)
plt.close()

IPython.embed()
