import numpy as np
import os
import IPython

exp_directory = "experiment__Lun_12_4_15_25"

files = os.listdir(exp_directory)
os.chdir(exp_directory)

train_filename     = [f for f in files if "train_rewards" in f][0]
mean_test_filename = [f for f in files if "test_rewards_mean" in f][0]
var_test_filename  = [f for f in files if "test_rewards_var" in f][0]

train     = np.load(train_filename)
mean_test = np.load(mean_test_filename)
var_test  = np.load(var_test_filename)
std_test  = np.sqrt(var_test)

print("train shape    : ", train.shape)
print("test mean shape: ", mean_test.shape)
print("test var shape : ", var_test.shape)
print("test std shape : ", std_test.shape)

# IPython.embed()
