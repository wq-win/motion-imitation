import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from mpi4py import MPI
from motion_imitation.envs import env_builder as env_builder
import os
import numpy as np
from tqdm import tqdm
import pickle  
from scipy.spatial.transform import Rotation as R
from DynamicSynapse.dynamicsynapse1 import DynamicSynapse
import matplotlib.pyplot as plt
from pretrain.pretrain_fc_velocity import Net

BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ENABLE_ENV_RANDOMIZER = True
num_procs = MPI.COMM_WORLD.Get_size()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
arg_parser.add_argument("--motion_file", dest="motion_file", type=str, default="motion_imitation/data/motions/dog_pace.txt")
arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=True)
arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=None)
arg_parser.add_argument("--model_file", dest="model_file", type=str, default="")
arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int, default=0) # save intermediate model every n policy steps

args = arg_parser.parse_args()

enable_env_rand = ENABLE_ENV_RANDOMIZER and (args.mode != "test")

def closure(r):
    def a():
        return r-0.1
    return a

if __name__ == "__main__":
    env = env_builder.build_imitation_env(motion_files=[args.motion_file],
                                        num_parallel_envs=num_procs,
                                        mode=args.mode,
                                        enable_randomizer=enable_env_rand,
                                        enable_rendering=args.visualize)
    
    net = Net()
    net.to(DEVICE)
    state_dict = torch.load("PretrainModel/predict_model_05-23_16-39-37.pkl", map_location="cuda:0")
    net.load_state_dict(state_dict=state_dict)
    # net.to(DEVICE)  

    net.dynamic_optimizer = DynamicSynapse(net.parameters(), lr=0.002, amp=0.002, period=700, dt=17)


    o = env.reset()
    env.render(mode='rgb_array')
    while True:
        action = net(torch.from_numpy(o).float().to(DEVICE))
        o,r,d,_ = env.step(action.cpu().detach().numpy())   
        net.dynamic_optimizer.step(closure=closure(r), dt=17)
        if d:
            env.reset() 
    env.close()