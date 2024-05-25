import torch
from pretrain.pretrain_fc_refactor import Net, input_dim, output_dim, o, a


test_model = Net(input_dim, output_dim)
# TODO: update path
test_model.load_state_dict(torch.load('PretrainModel/predict_model'))
test_model.eval()
# input_test = o
# test_model(o)
input_test = o[0]
print(test_model(o[0]), a[0])