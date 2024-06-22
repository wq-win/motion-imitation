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
    pma = copy.deepcopy(oma)
    pma[np.array([0, 6])] = -oma[np.array([0, 6])]
    pma[np.array([1, 4, 7, 10])] -= 0.6
    pma[np.array([2, 5, 8, 11])] -= -0.66
    return pma


def oma_to_right_action(oma):
    action = copy.deepcopy(oma)
    action[np.array([0, 3, 6, 9])] = -oma[np.array([0, 3, 6, 9])]
    action[np.array([1, 4, 7, 10])] -= 0.67
    action[np.array([2, 5, 8, 11])] -= -1.25
    return action


def error_between_target_and_result(o):
    """
    target motorangle is o[12:24]=env.step input action
    result motorangle is o[48:60]=current observation motorangle
    """
    error = o[12:24] - o[48:60]
    return error


def main():
    test_model = pretrain_save_data_V1.Net(12, 12)
    test_model.load_state_dict(torch.load('pretrain_model/save_data_V5_model_06_21_11_06_43.pkl', map_location=torch.device('cpu')))

    env = env_builder.build_imitation_env(motion_files=["motion_imitation/data/motions/dog_pace.txt"],
                                        num_parallel_envs=MPI.COMM_WORLD.Get_size(),
                                        mode="test",
                                        enable_randomizer=False,
                                        enable_rendering=True)
    o = env.reset()
    
    while True:
          
        oma = o[48:60]
        pma = oma_to_pma(oma)
        pma = torch.tensor(pma, dtype=torch.float32)
        displacement = test_model(pma)
        

        displacement = test_model(pma+displacement*TIMESTEP)
        displacement = displacement.detach().numpy()
        displacement[np.array([0, 6])] = -displacement[np.array([0, 6])]
        next_oma = oma + displacement * TIMESTEP * DISPLACEMENT_RATE + error_between_target_and_result(o) * 1
        action = oma_to_right_action(next_oma)
        
        o, r, d, _ = env.step(action)
        # if d:
        #     o = env.reset()
                 

    
if __name__ == '__main__':
    main()