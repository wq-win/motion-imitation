import os
import inspect
import pickle
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import numpy as np
from matplotlib import pyplot as plt
import torch
from motion_imitation.envs import env_builder
from mpi4py import MPI
from pretrain import pretrain_oma_data_Net
from find_offset import all_offset
from collect_test import test_oma
from collect import collect_pma_data

TIMESTEP = 1 / 30
DISPLACEMENT_RATE = 1

action_list = []
without_error_action_list = []
oma_list = []

def ploter(position1, direction1, position2, direction2, x=0, y=1, z=2, u=0, v=1, w=2):
    ax = plt.figure().add_subplot(projection='3d')
    X1 = position1[:, x]
    Y1 = position1[:, y]
    Z1 = position1[:, z]
    U1 = direction1[:, u]
    V1 = direction1[:, v]
    W1 = direction1[:, w]
    color_array1 = np.ones((position1.shape[0],3))
    color_array1[:, 1] = np.linspace(0.8, 0, color_array1.shape[0])
    ax.quiver(X1, Y1, Z1, U1, V1, W1, color=color_array1, length=TIMESTEP, normalize=False, label='oma')
    
    X2 = position2[:, x]
    Y2 = position2[:, y]
    Z2 = position2[:, z]
    U2 = direction2[:, u]
    V2 = direction2[:, v]
    W2 = direction2[:, w]
    color_array2 = np.ones((position2.shape[0],3))
    color_array2[:, 0] = np.linspace(0.8, 0, color_array2.shape[0])
    ax.quiver(X2, Y2, Z2, U2, V2, W2, color=color_array2, length=TIMESTEP, normalize=False, label='ppo')
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()
    ax.set_title('our model with ppo')
    test_oma.set_axes_equal(ax)
    plt.show()

def main():
    test_model = pretrain_oma_data_Net.Net(12, 12)
    test_model.load_state_dict(torch.load(os.path.join(parentdir,'pretrain_model/oma_model_06_26_16_51.pkl'), map_location=torch.device('cpu')))

    env = env_builder.build_imitation_env(motion_files=[os.path.join(parentdir,"motion_imitation/data/motions/dog_pace.txt")],
                                        num_parallel_envs=MPI.COMM_WORLD.Get_size(),
                                        mode="test",
                                        enable_randomizer=False,
                                        enable_rendering=False)

    o = env.reset()
    i = 600
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
        print(i,end='\r')
        
    env.close()

if __name__ == "__main__":
    main()
    action_array = np.array(action_list)
    without_error_action_array = np.array(without_error_action_list)
    oma_array = np.array(oma_list)
    
    oma_v, _, _ = collect_pma_data.calculate_ring_velocity(oma_array)
    
    action_array = all_offset.a_to_oa(action_array)
    without_error_action_array = all_offset.a_to_oa(without_error_action_array)

    with open('function_test/ppo_track.pkl', 'rb') as f:
        allresult = pickle.load(f)
    p2 = np.array(allresult['input'])
    d2 = np.array(allresult['output'])    
    ploter(oma_array, oma_v, p2, d2)