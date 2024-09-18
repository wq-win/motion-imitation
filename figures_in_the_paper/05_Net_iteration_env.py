import copy
import data
import augmentation_data
import numpy as np
import torch
from mpi4py import MPI
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from motion_imitation.envs import env_builder
from pretrain import trot_Net_train


TIMESTEP = data.TIMESTEP
position_list = []
direction_list = []
color_list = []

if __name__ == '__main__':
    trot = data.trot_array
    ma_v, _, _ = augmentation_data.calculate_ring_velocity(trot)
    position_list.append(trot)
    direction_list.append(ma_v)
    color_list.append(np.zeros((len(trot), 3)))
    
    env = env_builder.build_imitation_env(
        motion_files=["motion_imitation/data/motions/dog_trot.txt"],
        num_parallel_envs=MPI.COMM_WORLD.Get_size(),
        mode="test",
        enable_randomizer=False,
        enable_rendering=False,
    )
    o = env.reset()
    point = o[48:60]
    test_model = trot_Net_train.Net(12, 12)
    test_model.load_state_dict(torch.load('pretrain_model/trot_data_model_07_04_21_11_10.pkl', map_location=torch.device('cpu')))

    for _ in range(20):
        position_list.append(copy.deepcopy(point))
        point_tensor = torch.tensor(point, dtype=torch.float32)
        displacement = test_model(point_tensor)
        displacement = displacement.detach().numpy()
        point += displacement * TIMESTEP
        direction_list.append(displacement)
    env.close()
    color_nums = len(position_list) - 1   
    color = np.ones((color_nums, 3))
    color[:, 0] = np.linspace(0.8, 0, color_nums)
    color_list.append(color)
        
    
    position_array = np.vstack(position_list)
    direction_array = np.vstack(direction_list)
    color_array = np.vstack(color_list)
    print(position_array.shape[0], direction_array.shape[0], color_array.shape[0])
    data.ploter(position_array, direction_array, color_array=color_array)
    


