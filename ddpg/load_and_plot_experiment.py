import gym
from actor_critic_networks import Actor
import numpy as np
from pathlib import Path
import torch
from matplotlib import pyplot as plt
import os
import time

import IPython

# import sys
# if "win" in sys.platform: # My Local Windows Machine
#     sys.path.append("C:/Users/jdlow/Box/Fall 2019/ROB 590 RL for Driving/collision_environment/gym-lane-change/gym_lane_change/envs/")
# else: # Great Lakes Linux platfrom
#     sys.path.append("collision_environment/gym-lane-change/gym_lane_change/envs/")
# from lane_change_env import LaneChangeEnv

# Change foldername
foldername = "experiments/" + "p_defaults_Pen_12_4_17_37/"

render = False
episodes_per_file = 25 # number of episodes to run. The mean and stdev will be computed over these
underscore_in_experiment_name = True # eg. m2_report
ylimits = None

# save_intervals = list(range(0, 2501, 100))
saved_episode_nums = sorted([int(file.split("_")[1]) for file in os.listdir(foldername) if "actor" in file])
max_episode_num = max(saved_episode_nums)
IPython.embed()
step_size = saved_episode_nums[1] - saved_episode_nums[0]
save_intervals = list(range(0, max_episode_num+1, int(step_size)))


environments = {"pen": "Pendulum-v0",
                    "lun": "LunarLanderContinuous-v2",
                    "mou": "MountainCarContinuous-v0",
                    "lan": "LaneChangeEnv",
                    "bip": "BipedalWalker-v2",
                    }
save_filenames = [file for file in os.listdir(foldername) if "actor" in file]

# assert(len(save_intervals) == len(save_filenames))
file = save_filenames[0].split('_')
total_num = file[3]
if underscore_in_experiment_name:
    exp_name = "_".join(file[4:6])
    envname = environments[file[6].lower()]
else:
    exp_name = file[4]
    envname = environments[file[5].lower()]

start_time = "_".join(file[-4:]).strip("actor")

L1_SIZE = 400
L2_SIZE = 300

if envname.lower() == "lanechangeenv":
    env = LaneChangeEnv()
else:
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

    t1 = time.time()
    try:
        actor.load_state_dict(torch.load(experiment_path))
    except FileNotFoundError:
        print("\nFile not found: ", experiment_path)
        print("Saving results so far\n")
        break
    t2 = time.time()
    print("Time to load actor ", round(t2 - t1, 3))

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
    print("Time to test ", round(time.time() - t2, 3))


    assert(np.nonzero(save_num_rewards[:, save_ind])[0].shape[0] == save_num_rewards[:, save_ind].shape[0])

    print("\nTest across ", episodes_per_file, "after training for ", save_num, "episodes.")
    print("\tMean: ", save_num_rewards[:, save_ind].mean(), " | Stdev: ", np.std(save_num_rewards[:, save_ind]))
    print("\n")

env.close()

save_file = foldername + "/" + "eval_rewards_eps_by_intervals"
np.save(save_file, save_num_rewards)
print("Results saved to ", save_file)

means = [row.mean() for row in save_num_rewards.T]
stdevs = [np.std(row) for row in save_num_rewards.T]

plt.errorbar(save_intervals, means, yerr=stdevs,
            label=envname, fmt="-o", linewidth=2.0, ecolor="red", capsize=10.0)
plt.xlabel("Number of Training Episodes")
plt.ylabel("Rewards Per Episode for 100 Episodes")
if ylimits is not None:
    plt.ylim(ylimits)
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig(foldername +  foldername.split("/")[-2].lower().replace(" ", "_") + "_training_eval"+ \
        ".png", dpi=200)
plt.close()

# IPython.embed()
