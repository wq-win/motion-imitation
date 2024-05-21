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