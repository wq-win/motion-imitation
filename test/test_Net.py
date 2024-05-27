import pickle
import os, sys
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
from pretrain.pretrain_fc_refactor import Net


with open('pretrain/dataset/oa_1000episode.pkl', 'rb') as f:
        allresult = pickle.load(f)

o = np.array(allresult['o'], dtype=float)
a = np.array(allresult['a'])

o = torch.tensor(o, dtype=torch.float32)
a = torch.tensor(a, dtype=torch.float32)

test_model = Net(160, 12)
# TODO: update path
test_model.load_state_dict(torch.load('PretrainModel/predict_model_05-27_17-34-59.pkl'))
print(o[0])
print(test_model(o[0]))
print(a[0])