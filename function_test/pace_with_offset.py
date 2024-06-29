import numpy as np
from mpi4py import MPI

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from motion_imitation.envs import env_builder
from find_offset import all_offset

pace_fix = all_offset.pace_fix

def main():   
    env = env_builder.build_imitation_env(motion_files=["motion_imitation/data/motions/dog_pace.txt"],
                                        num_parallel_envs=MPI.COMM_WORLD.Get_size(),
                                        mode="test",
                                        enable_randomizer=False,
                                        enable_rendering=True)

    pma = np.array(pace_fix)[:, 7:]
    oma = all_offset.pma_to_oma(pma)
    action_array = all_offset.oma_to_right_action(oma)    
    
    env.reset()
    i = 38
    next_index = 1
    while True:
        action = action_array[i]
        o, r, d, _ = env.step(action)
        i += next_index
        i = i % action.shape[0]
  
    env.close()
if __name__ == '__main__':
    main()