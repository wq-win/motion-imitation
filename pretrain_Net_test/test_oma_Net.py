import copy
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import numpy as np
import torch
from pretrain import pretrain_oma_data_Net
from collect import collect_oma_data 
from collect_test import test_oma

TIMESTEP = 1 / 30
input_list = []
output_list = []

test_model = pretrain_oma_data_Net.Net(12, 12)
test_model.load_state_dict(torch.load('pretrain_model/oma_model_06_26_16_51.pkl', map_location=torch.device('cpu')))

point = collect_oma_data.sample_random_point_pi()

for i in range(100):
    
    input_list.append(point)
    point = torch.tensor(point, dtype=torch.float32)
    displacment = test_model(point)
    # 迭代2次
    displacment = test_model(point + displacment * TIMESTEP)
    point += displacment * TIMESTEP
    displacment = displacment.detach().numpy()
    output_list.append(displacment)
    point = point.detach().numpy()

input_array = np.vstack(input_list)
output_array = np.vstack(output_list)
test_oma.ploter(input_array, output_array)