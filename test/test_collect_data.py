import os
import inspect
import pickle
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print(parentdir)
os.sys.path.insert(0, parentdir)
import numpy as np
from motion_imitation.envs import env_builder
from mpi4py import MPI

with open('dataset/save_data_V06_26_2_1000.pkl', 'rb') as f:
            allresult = pickle.load(f)

input_array = np.array(allresult['input'], dtype=float)
# output = np.array(allresult['output'])


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
    p_ma = input_array
    
    # pma to oma
    p_ma[:, np.array([0, 6])] = -p_ma[:, np.array([0, 6])]
    p_ma[:, np.array([1, 4, 7, 10])] += 0.6
    p_ma[:, np.array([2, 5, 8, 11])] += -0.66
    
    # oma to a
    p_ma[:, np.array([0, 3, 6, 9])] = -p_ma[:, np.array([0, 3, 6, 9])]
    p_ma[:, np.array([1, 4, 7, 10])] -= 0.67
    p_ma[:, np.array([2, 5, 8, 11])] -= -1.25
    
    env.reset()
    # env.render(mode='rgb_array')

    i = 0
    next_index = 1
    while True:
        i = i % p_ma.shape[0]
        action = p_ma[i]
        o, r, d, _ = env.step(action)
        i += next_index
    env.close()
if __name__ == '__main__':
    main()