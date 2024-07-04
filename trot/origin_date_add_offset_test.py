import find_oma_trot_offset, find_a_oma_offset

import numpy as np
from mpi4py import MPI

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from motion_imitation.envs import env_builder


trot_array = find_oma_trot_offset.trot_array
oma = find_oma_trot_offset.trot_to_oma(trot_array)
action_array = find_a_oma_offset.oma_to_a(oma)
action_array[:, np.array([3, 9])] = -action_array[:, np.array([3, 9])]


def main():   
    env = env_builder.build_imitation_env(motion_files=["motion_imitation/data/motions/dog_trot.txt"],
                                        num_parallel_envs=MPI.COMM_WORLD.Get_size(),
                                        mode="test",
                                        enable_randomizer=False,
                                        enable_rendering=True)
    o = env.reset()
    i = 0
    next_index = 1
    while True:
        action = action_array[i]
        o, r, d, _ = env.step(action)
        i += next_index
        i = i % action.shape[0]
        # if d:
        #     env.reset()
  
    env.close()
if __name__ == '__main__':
    main()