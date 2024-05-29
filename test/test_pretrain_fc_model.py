import argparse
from mpi4py import MPI
from motion_imitation.envs import env_builder as env_builder
import os
import numpy as np
from tqdm import tqdm
import pickle  
import os
import pickle
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


ENABLE_ENV_RANDOMIZER = True

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
arg_parser.add_argument("--mode", dest="mode", type=str, default="test")
arg_parser.add_argument("--motion_file", dest="motion_file", type=str, default="motion_imitation/data/motions/dog_pace.txt")
arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=True)
arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=None)
arg_parser.add_argument("--model_file", dest="model_file", type=str, default="")
arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int, default=0) # save intermediate model every n policy steps

args = arg_parser.parse_args()

num_procs = MPI.COMM_WORLD.Get_size()
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

enable_env_rand = ENABLE_ENV_RANDOMIZER and (args.mode != "test")

env = env_builder.build_imitation_env(motion_files=[args.motion_file],
                                        num_parallel_envs=num_procs,
                                        mode=args.mode,
                                        enable_randomizer=enable_env_rand,
                                        enable_rendering=args.visualize)


o = env.reset()
env.render(mode='rgb_array')


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return self.fc4(x)

test_model = Net(160, 12)
# TODO: update path
# test_model.load_state_dict(torch.load('PretrainModel/predict_model_05-28_10-01-35.pkl'))
test_model.load_state_dict(torch.load('PretrainModel/predict_model_05-29_10-19-05.pkl'))

# print(o)
o = torch.tensor(o, dtype=torch.float32)
i = 0
i_list = []
# while len(i_list) < 100:
while True:
    i += 1
    action = test_model(o)
    action = action.detach().numpy()
    # print(action)
    o, r, d, _ = env.step(action)
    o = torch.tensor(o, dtype=torch.float32)
    env.render(mode='rgb_array')
    if d:
        print(i)
        i_list.append(i)
        env.reset()
        i = 0
env.close()
print(f"\naverage step:{sum(i_list) / len(i_list)}\n")

