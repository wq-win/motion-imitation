import pickle

import numpy as np
from motion_imitation.envs import env_builder
from mpi4py import MPI
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print(parentdir)
os.sys.path.insert(0, parentdir)
from pretrain import pretrain_fc_deep 


ENABLE_ENV_RANDOMIZER = True
motion_file = "motion_imitation/data/motions/dog_pace.txt"
num_procs = MPI.COMM_WORLD.Get_size()
mode = "test"
enable_env_rand = ENABLE_ENV_RANDOMIZER and (mode != "test")
visualize = True
def main():
    with open('dataset/o_a_collect_nums_1000.pkl', 'rb') as f:
            allresult = pickle.load(f)
    o = np.array(allresult['o'], dtype=float)

    a = np.array(allresult['a'])
    env = env_builder.build_imitation_env(motion_files=[motion_file],
                                            num_parallel_envs=num_procs,
                                            mode=mode,
                                            enable_randomizer=enable_env_rand,
                                            enable_rendering=visualize)
    
    oma = o[:, 48:60]
    env.reset()
    env.render(mode='rgb_array')
    # print(len(a), len(a[0]))  

    i = 0
    while True:
        oma[i][np.array([0, 3, 6, 9])] = -oma[i][np.array([0, 3, 6, 9])]
        oma[i][np.array([1, 4, 7, 10])] -= 0.67
        oma[i][np.array([2, 5, 8, 11])] -= -1.25  # -.66
        o, r, d, _ = env.step(oma[i])
        # print(ma[i])
        # if d:
        #     print(i)
        #     env.reset()            
            # i = 0
        i += 1
        if i % 600 == 0:
            print('one episode')
    env.close()
if __name__ == '__main__':
    main()