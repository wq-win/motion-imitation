import numpy as np
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


NOWTIME = time.strftime("%m_%d_%H_%M_%S", time.localtime())
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc4(x)
      

if __name__ == "__main__":
    print(DEVICE)
    # update the data path
    with open('dataset/add_pace_data_3000_10.pkl', 'rb') as f:
            allresult = pickle.load(f)

    input = np.array(allresult['input'])
    output = np.array(allresult['output'])
    print(len(input), len(input[0]), len(output), len(output[0]))  
    
    input_dim = len(input[0])
    output_dim = len(output[0])

    losses_list = list()
    episode = 10
    epoch = 10
    lambda1 = 0.2
    learning_rate = 1e-4

    input = torch.tensor(input, dtype=torch.float32).to(DEVICE)
    output = torch.tensor(output, dtype=torch.float32).to(DEVICE)
    train_dataset = TensorDataset(input, output)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Net(input_dim, output_dim)
    model.to(DEVICE)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for ep in range(episode):
        for e in range(epoch):
            epoch_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader, 0):                   
                outputs = model(inputs)
                loss = criterion(outputs, labels)              
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                # print(i)
                if i % BATCH_SIZE == BATCH_SIZE - 1:
                    average_loss = epoch_loss / BATCH_SIZE  
                    print(f"Episode: {ep + 1}, Epoch: {e + 1}, i: {i}, Loss: {average_loss:.4f}")
                    losses_list.append(average_loss)
                    epoch_loss = 0.0

    plt.plot(range(len(losses_list)), losses_list)
    plt.savefig(f'result/loss/pace_data_loss_{NOWTIME}.png', dpi=300)
    plt.show()
    file_path = f"pretrain_model/pace_data_model_{NOWTIME}.pkl"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), file_path)