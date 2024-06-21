import copy
import pickle
from matplotlib import pyplot as plt
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
from pretrain import pretrain_save_data_V1


TIMESTEP = 1 / 30
DISPLACEMENT_RATE = 1

def oma_to_pma(oma):
    oma[np.array([0, 6])] = -oma[np.array([0, 6])]
    oma[np.array([1, 4, 7, 10])] -= 0.6
    oma[np.array([2, 5, 8, 11])] -= -0.66
    return oma

def main():
    env = env_builder.build_imitation_env(motion_files=["motion_imitation/data/motions/dog_pace.txt"],
                                        num_parallel_envs=MPI.COMM_WORLD.Get_size(),
                                        mode="test",
                                        enable_randomizer=False,
                                        enable_rendering=True)
    o = env.reset()
    o = torch.tensor(o, dtype=torch.float32)
    test_model = pretrain_save_data_V1.Net(12, 12)
    test_model.load_state_dict(torch.load('pretrain_model/save_data_V5_model_06_21_11_06_43.pkl', map_location=torch.device('cpu')))
    oma = o[48:60]
    pma = oma_to_pma(oma)
    displacement = test_model(pma)
    # displacement[np.array([0, 6])] = -displacement[np.array([0, 6])]
    action = oma + displacement * TIMESTEP * DISPLACEMENT_RATE

    while True:
        o, r, d, _ = env.step()
        

    
if __name__ == '__main__':
    main()