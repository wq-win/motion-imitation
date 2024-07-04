import numpy as np
import torch
import Net_train 
import data_augmentation 
import data_augmentation_plot

TIMESTEP = 1 / 30
input_list = []
output_list = []

test_model = Net_train.Net(12, 12)
test_model.load_state_dict(torch.load('trot/model/tort_model_07_04_13_01.pkl', map_location=torch.device('cpu')))

point = data_augmentation.sample_random_point_pi()

for i in range(100):
    
    input_list.append(point)
    point = torch.tensor(point, dtype=torch.float32)
    displacment = test_model(point)
    # 迭代2次
    # displacment = test_model(point + displacment * TIMESTEP)
    point += displacment * TIMESTEP
    displacment = displacment.detach().numpy()
    output_list.append(displacment)
    point = point.detach().numpy()

input_array = np.vstack(input_list)
output_array = np.vstack(output_list)
data_augmentation_plot.ploter(input_array, output_array)