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
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print(parentdir)
os.sys.path.insert(0, parentdir)
from pretrain import pretrain_fc_deep 

ENABLE_ENV_RANDOMIZER = True
motion_file = "motion_imitation/data/motions/dog_pace.txt"
num_procs = MPI.COMM_WORLD.Get_size()
mode = "test"
enable_env_rand = ENABLE_ENV_RANDOMIZER and (mode != "test")
visualize = True
def main(test_episodes = 'inf'):
    with open('dataset/o_a_collect_nums_1000.pkl', 'rb') as f:
            allresult = pickle.load(f)
    o = np.array(allresult['o'], dtype=float)

    a = np.array(allresult['a'])
    env = env_builder.build_imitation_env(motion_files=[motion_file],
                                            num_parallel_envs=num_procs,
                                            mode=mode,
                                            enable_randomizer=enable_env_rand,
                                            enable_rendering=visualize)


    test_model = pretrain_fc_deep.Net(160, 12)
    test_model.load_state_dict(torch.load('pretrain_model/predict_model_06-03_14-59-13.pkl', map_location=torch.device('cpu')))
    
    o = env.reset()
    o = torch.tensor(o, dtype=torch.float32)
    env.render(mode='rgb_array')
    
    i = 0
    i_list = []
    if test_episodes == 'inf':
        while True:
            action = test_model(o)
            action = action.detach().numpy()
            # print(action)
            o, r, d, _ = env.step(action)
            o = torch.tensor(o, dtype=torch.float32)
            if d:
                env.reset()
    else:
        while len(i_list) < int(test_episodes):
            i += 1
            action = test_model(o)
            action = action.detach().numpy()
            # print(action)
            o, r, d, _ = env.step(action)
            o = torch.tensor(o, dtype=torch.float32)
            if d:
                print(i)
                i_list.append(i)
                env.reset()
                i = 0
    env.close()
    print(f"\naverage step:{sum(i_list) / len(i_list)}\n")
if __name__ == '__main__':
    # main('inf')
    main(10)