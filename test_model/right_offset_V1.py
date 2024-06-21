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


def oma_to_pma(oma):
    oma[np.array([0, 6])] = -oma[np.array([0,6])]
    oma[np.array([1, 4, 7, 10])] -= 0.6
    oma[np.array([2, 5, 8, 11])] -= -0.66
    pma = oma
    return pma


def oma_to_right_action(oma):
    oma[np.array([0, 3, 6, 9])] = -oma[np.array([0, 3, 6, 9])]  # 0,6 -= -0.00325897 ; 3,9 -= 0.0034551
    oma[np.array([1, 4, 7, 10])] -= 0.67  # 0.6742553
    oma[np.array([2, 5, 8, 11])] -= -1.25 # -1.25115246
    right_action = oma
    return right_action

def offset_to_right_action(oma):
    oma[np.array([0, 6])] -= -0.00325897
    oma[np.array([3, 9])] -= 0.0034551
    oma[np.array([1, 4, 7, 10])] -= 0.6742553
    oma[np.array([2, 5, 8, 11])] -= -1.25115246
    return oma

def test_to_right_action(oma):
    oma[np.array([1, 4, 7, 10])] -= 0.67
    oma[np.array([2, 5, 8, 11])] -= -1.25
    return oma
def main():
    env = env_builder.build_imitation_env(motion_files=["motion_imitation/data/motions/dog_pace.txt"],
                                        num_parallel_envs=MPI.COMM_WORLD.Get_size(),
                                        mode="test",
                                        enable_randomizer=False,
                                        enable_rendering=True)
    o = env.reset()
    oma = o[48:60]
    # oma = oma_to_right_action(oma)
    # oma = offset_to_right_action(oma)
    oma = test_to_right_action(oma)
    i = 0
    while True:
        # oma *= 0
        o, r, d, _ = env.step(oma)
        oma = o[48:60]
        
        # oma = oma_to_right_action(oma)
        # oma = offset_to_right_action(oma)
        oma = test_to_right_action(oma)
        print(oma)
        if i == 6000:
            env.reset()
            i = 0
        i += 1
        print(i)
    
if __name__ == '__main__':
    main()