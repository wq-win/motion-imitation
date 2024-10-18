import copy

import numpy as np
import data
import augmentation_data
import torch
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from pretrain import trot_Net_train


TIMESTEP = augmentation_data.TIMESTEP
position_list = []
direction_list = []
color_list = []

if __name__ == '__main__':
    data.set_rand_seed(1)
    trot = data.trot_array
    ma_v, _, _ = augmentation_data.calculate_ring_velocity(trot)
    position_list.append(trot)
    direction_list.append(ma_v)
    color_list.append(np.zeros((len(trot), 3)))
    
    point = augmentation_data.sample_random_point(trot)
    # point = augmentation_data.sample_random_point_pi()        
    test_model = trot_Net_train.Net(12, 12)
    test_model.load_state_dict(torch.load('pretrain_model/trot_data_model_07_04_21_11_10.pkl', map_location=torch.device('cpu')))

    for _ in range(50):
        position_list.append(copy.deepcopy(point))
        point_tensor = torch.tensor(point, dtype=torch.float32)
        displacement = test_model(point_tensor)
        displacement = displacement.detach().numpy()
        point += displacement * TIMESTEP
        direction_list.append(displacement)
    
    color_nums = len(position_list) - 1   
    color = np.ones((color_nums, 3))
    color[:, 0] = np.linspace(0.8, 0, color_nums)
    color_list.append(color)
    
    position_array = np.vstack(position_list)
    direction_array = np.vstack(direction_list)
    color_array = np.vstack(color_list)
    print(position_array.shape[0], direction_array.shape[0], color_array.shape[0])
    data.ploter(position_array, direction_array, color_array=color_array)
