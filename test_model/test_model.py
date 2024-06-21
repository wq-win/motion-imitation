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


def oa_to_a(oa):
    oa[np.array([1, 4, 7, 10])] -= 0.67
    oa[np.array([2, 5, 8, 11])] -= -1.25
    return oa

def main():
    env = env_builder.build_imitation_env(motion_files=["motion_imitation/data/motions/dog_pace.txt"],
                                        num_parallel_envs=MPI.COMM_WORLD.Get_size(),
                                        mode="test",
                                        enable_randomizer=False,
                                        enable_rendering=True)
    o = env.reset()
    oa = o[12:24]
    oa = oa_to_a(oa)
    i = 0
    while True:
        o, r, d, _ = env.step(oa)
        oa = o[12:24]
        print(o[48:60])
        oa = oa_to_a(oa)
        if i == 500:
            env.reset()
            i = 0
        i += 1
        print(i)
    
if __name__ == '__main__':
    main()