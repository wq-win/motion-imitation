# The contents of this file can be changed at will.
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
from pretrain import pretrain_oma_data_Net
from find_offset import all_offset


TIMESTEP = 1 / 30
DISPLACEMENT_RATE = 1

action_list = []
without_error_action_list = []
oma_list = []

def ploter(action_list, without_error_action_list, oma_list):
    
    plt.figure()
    for i in range(12):
        plt.subplot(4, 3, i+1)
        plt.plot(range(len(action_list[:, i]),), action_list[:, i], label=f'action:{i}', linestyle='-')
        plt.plot(range(len(without_error_action_list[:, i]),), without_error_action_list[:, i], label=f'without_error_action:{i}', linestyle='-.')
        plt.plot(range(len(oma_list[:, i]),), oma_list[:, i], label=f'oma:{i}', linestyle='--')
        plt.legend()
    plt.show()   
    
    ax = plt.figure().add_subplot(projection='3d')

    x1= action_list[:, 0]
    y1= action_list[:, 1]
    z1= action_list[:, 2]
    ax.plot(x1, y1, z1, label='action')
    
    x2= without_error_action_list[:, 0]
    y2= without_error_action_list[:, 1]
    z2= without_error_action_list[:, 2]
    ax.plot(x2, y2, z2, label='without_error_action')
    
    x3= oma_list[:, 0]
    y3= oma_list[:, 1]
    z3= oma_list[:, 2]
    ax.plot(x3, y3, z3, label='oma')
    
    ax.legend()
    plt.show()

def main():
    test_model = pretrain_oma_data_Net.Net(12, 12)
    test_model.load_state_dict(torch.load(os.path.join(parentdir,'pretrain_model/oma_model_06_26_16_51.pkl'), map_location=torch.device('cpu')))

    env = env_builder.build_imitation_env(motion_files=[os.path.join(parentdir,"motion_imitation/data/motions/dog_pace.txt")],
                                        num_parallel_envs=MPI.COMM_WORLD.Get_size(),
                                        mode="test",
                                        enable_randomizer=False,
                                        enable_rendering=True)

    o = env.reset()
    i = 1000
    while True:
        oma = o[48:60]
        oma_list.append(oma)
        oma = torch.tensor(oma, dtype=torch.float32)
        for _ in range(1):
            displacement = test_model(oma)
            displacement = test_model(oma + displacement * TIMESTEP)  # 这一行执行，相当于迭代两步
        displacement = displacement.detach().numpy()
        oma = oma.detach().numpy()
            
        without_error_oma = oma + displacement * TIMESTEP * DISPLACEMENT_RATE
        without_error_oma_action = all_offset.oma_to_right_action(without_error_oma)  
        without_error_action_list.append(without_error_oma_action)
            
        next_oma = oma + displacement * TIMESTEP * DISPLACEMENT_RATE + all_offset.error_between_target_and_result(o, True) * 1
        # next_oma = oma + displacement * TIMESTEP * DISPLACEMENT_RATE
        action = all_offset.oma_to_right_action(next_oma)  

        action_list.append(action)        
        o, r, d, info = env.step(action)
        
        # if d:
        #     o = env.reset()
        if i == 0:
            break
        i -= 1
        
    env.close()

if __name__ == "__main__":
    main()
    action_array = np.array(action_list)
    without_error_action_array = np.array(without_error_action_list)
    oma_array = np.array(oma_list)
    
    
    action_array = all_offset.a_to_oa(action_array)
    without_error_action_array = all_offset.a_to_oa(without_error_action_array)

    
    ploter(action_array, without_error_action_array, oma_array)