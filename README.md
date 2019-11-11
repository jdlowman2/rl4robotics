# rl4robotics

EECS 545 Machine Learning Course Project Fall 2019
Reinforcement learning for robotics


1) Deep Deterministic Policy Gradient
To see an already trained network perform, in ddpg/ddpg.py, change MAX_EPISODES = 0. Run the script from within the ddpg folder. At the interactive prompt, enter the following commands:

    ddpg.load_experiment(<experiment_name>) # e.g. "lunarlander_11_11"
    ddpg.demonstrate()
