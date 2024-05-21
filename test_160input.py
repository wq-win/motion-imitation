import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from mpi4py import MPI
from motion_imitation.envs import env_builder as env_builder
import os
import numpy as np
from tqdm import tqdm
import pickle  
from scipy.spatial.transform import Rotation as R
from DynamicSynapse2D import DynamicSynapseArray
# 使用pickle从文件加载数组  
import matplotlib.pyplot as plt

BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(160, 256)
      self.dropout1 = nn.Dropout(0.25)
      self.fc2 = nn.Linear(256, 64)
      self.dropout2 = nn.Dropout(0.25)
      self.fc3 = nn.Linear(64,12)
    #   self.dropout1 = nn.Dropout2d(0.25)
    #   self.fc2 = nn.Linear(1000, 12)
    
    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      x = F.relu(x)
      x = self.fc3(x)
      # Use the rectified-linear activation function over x
      output = torch.tanh(x)
    #   x = self.dropout1(x)
    #   x = self.fc2(x)
    #   x = F.tanh(x)
    #   output = self.fc2(x)
      return output
    
if __name__ == "__main__":
    loss_list = []
    episode = 100
    epoch = 50
    with open('oa.pkl', 'rb') as f:  
        allresult = pickle.load(f) 

    o = np.array(allresult['o'], dtype=float)
    a = np.array(allresult['a'])
    print(len(o),len(o[0]), len(a),len(a[0]))

    net = Net()
    net.to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    for ep in range(episode):
        sample_index = np.random.choice(range(len(o)), BATCH_SIZE)
        input_ = torch.from_numpy(o[sample_index]).float().to(DEVICE)
        label_ = torch.from_numpy(a[sample_index]).to(DEVICE)
        for e in range(epoch):
            running_loss = 0.0
            for i, data in enumerate(range(BATCH_SIZE), 0):
                # inputs, labels = data
                optimizer.zero_grad()
                outputs = net(input_[i])
                loss = criterion(outputs, label_[i])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            loss_list.append(running_loss / BATCH_SIZE)
            print(f"Episode {ep+1},Epoch {e+1}, Loss: {running_loss / BATCH_SIZE}")

plt.plot(range(len(loss_list)), loss_list)
plt.show()

torch.save(net.state_dict(), "predict_model.pkl")

# ENABLE_ENV_RANDOMIZER = True

# arg_parser = argparse.ArgumentParser()
# arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
# arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
# arg_parser.add_argument("--motion_file", dest="motion_file", type=str, default="motion_imitation/data/motions/dog_pace.txt")
# arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=True)
# arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
# arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=None)
# arg_parser.add_argument("--model_file", dest="model_file", type=str, default="")
# arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
# arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int, default=0) # save intermediate model every n policy steps

# args = arg_parser.parse_args()

# num_procs = MPI.COMM_WORLD.Get_size()
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


# # 使用pickle从文件加载数组  
# with open('weight_dataV.pkl', 'rb') as f:  
#     allresult = pickle.load(f)  

# weights_PN2KC_bool = allresult['weights_PN2KC_bool']
# num_dim_KC_activated = allresult['num_dim_KC_activated']
# KCtoMBONweight = allresult['KCtoMBONweight']

# # weights_PN2KC_bool = allresult['PNtoKCweight']
# # num_dim_KC_activated = allresult['activate_KC_dims']
# # KCtoMBONweight = allresult['KCtoMBONweight']

# # PN dims > 2
# # KC = PN @ PNtoKCweight
# # sorted_indices = np.argsort(KC, axis=1)[:, ::-1]  
# # inactivate_indices = sorted_indices[:, activate_KC_dims:]  
# # KC[np.arange(KC.shape[0])[:, None], inactivate_indices] = 0
# # MBON = KC @ KCtoMBONweight
# # print(MBON)

# enable_env_rand = ENABLE_ENV_RANDOMIZER and (args.mode != "test")

# env = env_builder.build_imitation_env(motion_files=[args.motion_file],
#                                         num_parallel_envs=num_procs,
#                                         mode=args.mode,
#                                         enable_randomizer=enable_env_rand,
#                                         enable_rendering=args.visualize)

# # print(env.observation_space, env.action_space)  # Box(160,) Box(12,)

# o = env.reset()
# env.render(mode='rgb_array')
# while True:
#     action = net(torch.from_numpy(o).float())
#     o,r,d,_ = env.step(action.detach().numpy())   
#     if d:
#         env.reset() 
# env.close()