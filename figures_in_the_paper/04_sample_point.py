import copy
from matplotlib import pyplot as plt
import numpy as np
import data
import augmentation_data


TIMESTEP = augmentation_data.TIMESTEP
SAMPLE_NUM = 100
ITER_TIMES = 2
position_list = []
direction_list = []
color_list = []

if __name__ == "__main__":
    data.set_rand_seed(1)
    trot = data.trot_array
    ma_v, _, _ = augmentation_data.calculate_ring_velocity(trot)
    position_list.append(trot)
    direction_list.append(ma_v)
    color_list.append(np.zeros((len(trot), 3)))
    
    for j in range(SAMPLE_NUM):
        point = augmentation_data.sample_random_point(trot)
        # point = augmentation_data.sample_random_point_pi()
        for i in range(ITER_TIMES):
            position_list.append(copy.deepcopy(point))
            normal_direction = augmentation_data.calculate_point_normal_direction(trot, point, if_next=False)
            normal_displacement = augmentation_data.repulse(normal_direction, trot, point)
            tangent_displacement = augmentation_data.calculate_point_tangent_velocity(trot, point)
            
            displacement =  tangent_displacement + normal_displacement 
            displacement = augmentation_data.calculate_point_displacement(trot, point, displacement)
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