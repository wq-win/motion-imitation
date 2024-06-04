import os
import torch
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print(parentdir)
os.sys.path.insert(0, parentdir)
from pretrain import pretrain_fc_deep 
test_model = pretrain_fc_deep.Net(160, 12)
test_model.load_state_dict(torch.load('pretrain_model/predict_model_06-03_13-21-39.pkl', map_location=torch.device('cpu')))
# print(test_model)    

