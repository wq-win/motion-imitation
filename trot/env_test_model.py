import Net_train, find_a_oma_offset

import os
import inspect

import numpy as np
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from matplotlib import pyplot as plt
import torch
from motion_imitation.envs import env_builder
from mpi4py import MPI



TIMESTEP = 1 / 30
DISPLACEMENT_RATE = 1
def main():
    test_model = Net_train.Net(12, 12)
    test_model.load_state_dict(torch.load(os.path.join(parentdir,'trot/model/trot_model_07_04_13_01.pkl'), map_location=torch.device('cpu')))

    env = env_builder.build_imitation_env(motion_files=[os.path.join(parentdir,"motion_imitation/data/motions/dog_trot.txt")],
                                        num_parallel_envs=MPI.COMM_WORLD.Get_size(),
                                        mode="test",
                                        enable_randomizer=False,
                                        enable_rendering=True)
    
    o = env.reset()
    while True:
        oma = o[48:60]
        # oma_list.append(oma)
        oma = torch.tensor(oma, dtype=torch.float32)
        for _ in range(1):
            displacement = test_model(oma)
            displacement = test_model(oma + displacement * TIMESTEP)  # 这一行执行，相当于迭代两步
        displacement = displacement.detach().numpy()
        oma = oma.detach().numpy()
            
        next_oma = oma + displacement * TIMESTEP * DISPLACEMENT_RATE 
        action = find_a_oma_offset.oma_to_a(next_oma)
        action[np.array([3, 9])] = -action[np.array([3, 9])]

        o, r, d, info = env.step(action)
        
        # if d:
        #     o = env.reset()
    env.close()
    
if __name__ == "__main__":
    main()