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
import tqdm


NOWTIME = time.strftime("%m_%d_%H_%M_%S", time.localtime())
BATCH_SIZE = 32
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
    with open('dataset/augmentation_pace_data_100000_100.pkl', 'rb') as f:
            allresult = pickle.load(f)

    input = np.array(allresult['input'])
    output = np.array(allresult['output'])
    print(len(input), len(input[0]), len(output), len(output[0]))  
    
    input_dim = len(input[0])
    output_dim = len(output[0])

    losses_list = list()
    episode = 2
    epoch = 2
    lambda1 = 0.2
    learning_rate = 1e-4

    input = torch.tensor(input, dtype=torch.float32).to(DEVICE)
    output = torch.tensor(output, dtype=torch.float32).to(DEVICE)
    train_dataset = TensorDataset(input, output)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Net(input_dim, output_dim)
    model.to(DEVICE)
    criterion = nn.MSELoss()
    nweights = 0
    for name, weights in model.named_parameters():
        if 'bias' not in name:
            nweights = nweights + weights.numel()
    print(f'Total number of weights in the model = {nweights}')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for ep in range(episode):
        for e in range(epoch):
            epoch_loss = 0.0
            for i, (inputs, labels) in tqdm.tqdm(enumerate(train_loader, 0)):                   
                outputs = model(inputs)
                loss = criterion(outputs, labels)              
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                # print(i)
                if i % BATCH_SIZE == BATCH_SIZE - 1:
                    average_loss = epoch_loss / BATCH_SIZE  
                    losses_list.append(average_loss)
                    epoch_loss = 0.0
            print(f"Episode: {ep + 1}, Epoch: {e + 1}, Loss: {average_loss}")

    plt.plot(range(len(losses_list)), losses_list)
    plt.title(f'weights_nums:{nweights}')
    plt.savefig(f'result/loss/pace_data_loss_{NOWTIME}.png', dpi=300)
    plt.show()
    file_path = f"pretrain_model/pace_data_model_{NOWTIME}.pkl"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), file_path)