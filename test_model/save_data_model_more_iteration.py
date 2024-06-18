import copy
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
from pretrain import pretrain_save_data_V1

TIMESTEP = 1 / 30
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
    
    test_model = pretrain_save_data_V1.Net(12, 12)
    test_model.load_state_dict(torch.load('pretrain_model/save_data_V5_model_06_18_18_47_48.pkl', map_location=torch.device('cpu')))
    o = env.reset()
    o = torch.tensor(o, dtype=torch.float32)
    env.render(mode='rgb_array')
    oma = o[48:60]

    pma = copy.deepcopy(oma)
    # oma to pma
    pma[np.array([0, 6])] = -oma[np.array([0, 6])]
    pma[np.array([1, 4, 7, 10])] -= 0.6
    pma[np.array([2, 5, 8, 11])] -= -0.66

    iter_times = 32
    for _ in range(iter_times):
        displacement = test_model(pma)
        # displacement = 0
        pma += displacement * TIMESTEP

    action_o = pma
    action_o[np.array([0, 3, 6, 9])] = -action_o[np.array([0, 3, 6, 9])]
    action_o[np.array([1, 4, 7, 10])] -= 0.67
    action_o[np.array([2, 5, 8, 11])] -= -1.25
    action = action_o

    i = 0
    while True: 
        action = action.detach().numpy()
        o, r, d, _ = env.step(action)
        o = torch.tensor(o, dtype=torch.float32)
        action_o = o[48:60]

        pma = copy.deepcopy(oma)
        # oma to pma
        pma[np.array([0,6])] = -oma[np.array([0,6])]
        pma[np.array([1, 4, 7, 10])] -= 0.6
        pma[np.array([2, 5, 8, 11])] -= -0.66

        for _ in range(iter_times):
            displacement = test_model(pma)
            # displacement = 0
            action_o = oma + displacement * TIMESTEP

        action_o[np.array([0, 3, 6, 9])] = -action_o[np.array([0, 3, 6, 9])]
        action_o[np.array([1, 4, 7, 10])] -= 0.67
        action_o[np.array([2, 5, 8, 11])] -= -1.25
        action = action_o
        if d:
            print(i)
            env.reset()            
            i = 0
        i += 1
    env.close()
if __name__ == '__main__':
    main()