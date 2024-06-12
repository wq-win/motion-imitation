import pickle
import numpy as np
import torch
from motion_imitation.envs import env_builder
from mpi4py import MPI
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print(parentdir)
os.sys.path.insert(0, parentdir)
from pretrain import pretrain_o_ma


ENABLE_ENV_RANDOMIZER = True
motion_file = "motion_imitation/data/motions/dog_pace.txt"
num_procs = MPI.COMM_WORLD.Get_size()
mode = "test"
enable_env_rand = ENABLE_ENV_RANDOMIZER and (mode != "test")
visualize = True
def main():

    env = env_builder.build_imitation_env(motion_files=[motion_file],
                                            num_parallel_envs=num_procs,
                                            mode=mode,
                                            enable_randomizer=enable_env_rand,
                                            enable_rendering=visualize)
    
    test_model = pretrain_o_ma.Net(12, 12)
    test_model.load_state_dict(torch.load('pretrain_model/o_ma_model_06-11_22-37-34.pkl', map_location=torch.device('cpu')))
    o = env.reset()
    o = torch.tensor(o, dtype=torch.float32)
    env.render(mode='rgb_array')

    i = 0
    while True:
        action = test_model(o[48:60])
        action = action.detach().numpy()
        o, r, d, _ = env.step(action)
        o = torch.tensor(o, dtype=torch.float32)
        if d:
            print(i)
            env.reset()            
            i = 0
        i += 1
    env.close()
if __name__ == '__main__':
    main()