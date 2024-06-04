import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

NOWTIME = time.strftime("%m-%d_%H-%M-%S", time.localtime())
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return self.fc4(x)


if __name__ == "__main__":
    print(DEVICE)
    # update the data path
    with open('dataset/o_a_collect_nums_1000.pkl', 'rb') as f:
            allresult = pickle.load(f)

    o = np.array(allresult['o'], dtype=float)
    a = np.array(allresult['a'])
    print(len(o), len(o[0]), len(a), len(a[0]))  
    input_dim = len(o[0])
    output_dim = len(a[0])

    loss_list = list()
    episode = 5
    epoch = 5
    lambda1 = 0.2
    learning_rate = 1e-4
    
    assert len(o) == len(a), 'train data error!'
    o = torch.tensor(o, dtype=torch.float32).to(DEVICE)
    a = torch.tensor(a, dtype=torch.float32).to(DEVICE)

    train_dataset = TensorDataset(o, a)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Net(input_dim, output_dim)
    model.to(DEVICE)
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    nweights = 0
    for name, weights in model.named_parameters():
        if 'bias' not in name:
            nweights = nweights + weights.numel()
    print(f'Total number of weights in the model = {nweights}')
    model.train()
    for ep in range(episode):
        for e in range(epoch):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader, 0):                   
                outputs = model(inputs)
                loss = criterion(outputs, labels)              
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # print(i)
                if i % BATCH_SIZE == BATCH_SIZE - 1:
                    loss = running_loss / 10
                    print(f"Episode: {ep + 1},Epoch: {e + 1},i: {i},Loss: {loss}")
                    loss_list.append(loss)
                    running_loss = 0.0

    plt.plot(range(len(loss_list)), loss_list)
    plt.show()
    file_path = f"pretrain_model/predict_model_{NOWTIME}.pkl"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
      os.makedirs(directory)
    torch.save(model.state_dict(), file_path)