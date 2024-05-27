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
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
with open('pretrain/dataset/oa.pkl', 'rb') as f:
        allresult = pickle.load(f)

o = np.array(allresult['o'], dtype=float)
a = np.array(allresult['a'])
# print(len(o), len(o[0]), len(a), len(a[0]))  
input_dim = o[0]
output_dim = a[0]

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10000)
        # self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(10000, 1000)
        # self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(1000, output_dim)
    #   self.dropout1 = nn.Dropout2d(0.25)
    #   self.fc2 = nn.Linear(1000, 12)

    # x represents our data
    def forward(self, x):
        # Pass data through conv1
        x = self.fc1(x)
        # x = F.relu(x)
        x = self.fc2(x)
        # x = F.relu(x)
        x = self.fc3(x)
        # Use the rectified-linear activation function over x
        # output = torch.tanh(x)
        output = x
    #   x = self.dropout1(x)
    #   x = self.fc2(x)
    #   x = F.tanh(x)
    #   output = self.fc2(x)
        return output


if __name__ == "__main__":
    loss_list = list()
    episode = 100
    epoch = 50
    lambda1 = 0.2
    learning_rate = 1e-4
    
    assert len(o) == len(a), 'train data error!'
    o = torch.from_numpy(o)
    a = torch.from_numpy(a)
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

    for ep in range(episode):
        model.train()
        for e in range(epoch):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader, 0):                
                L1_term = torch.tensor(0., requires_grad=True)
                for name, weights in model.named_parameters():
                    if 'bias' not in name:
                        weights_sum = torch.sum(torch.abs(weights))
                        L1_term +=  weights_sum
                L1_term /= nweights
                
                outputs = model(inputs)
                loss = criterion(outputs, inputs) + L1_term * lambda1               
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if i % BATCH_SIZE == BATCH_SIZE - 1:
                    loss = running_loss / BATCH_SIZE
                    print(f"Episode {ep + 1},Epoch {e + 1}, i {i},Loss: {loss}")
                    loss_list.append(loss)
                    running_loss = 0.0

    plt.plot(range(len(loss_list)), loss_list)
    plt.show()

    torch.save(model.state_dict(),
               "PretrainModel/predict_model_{}.pkl".format(NOWTIME))
