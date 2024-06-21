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
    test_model.load_state_dict(torch.load('pretrain_model/save_data_V5_model_06_18_15_05_05.pkl', map_location=torch.device('cpu')))
    o = env.reset()
    o = torch.tensor(o, dtype=torch.float32)
    env.render(mode='rgb_array')
    oma = o[48:60]
    iter_times = 32
    for _ in range(iter_times):
        displacement = test_model(oma)
        # displacement = 0
        oma += displacement * TIMESTEP

    oma[np.array([0, 6])] = -oma[np.array([0, 6])]
    oma[np.array([1, 4, 7, 10])] += 0.6
    oma[np.array([2, 5, 8, 11])] += -0.66

    oma[np.array([0, 3, 6, 9])] = -oma[np.array([0, 3, 6, 9])]
    oma[np.array([1, 4, 7, 10])] -= 0.67
    oma[np.array([2, 5, 8, 11])] -= -1.25
    action = oma

    i = 0
    done_times = 0
    sum_step = 0
    while True: 
        action = action.detach().numpy()
        o, r, d, _ = env.step(action)
        o = torch.tensor(o, dtype=torch.float32)
        oma = o[48:60]
        iter_times = 32
        for _ in range(iter_times):
            displacement = test_model(oma)
            # displacement = 0
            oma += displacement * TIMESTEP

        oma[np.array([0, 6])] = -oma[np.array([0, 6])]
        oma[np.array([1, 4, 7, 10])] += 0.6
        oma[np.array([2, 5, 8, 11])] += -0.66

        oma[np.array([0, 3, 6, 9])] = -oma[np.array([0, 3, 6, 9])]
        oma[np.array([1, 4, 7, 10])] -= 0.67
        oma[np.array([2, 5, 8, 11])] -= -1.25
        action = oma
        i += 1
        if d:
            done_times += 1
            sum_step += i
            print(i)
            env.reset()            
            i = 0
        
        # if done_times == 100:
        #     print(f'\naverage_episode_step:{sum_step / done_times}')
        #     break
    env.close()
if __name__ == '__main__':
    main()