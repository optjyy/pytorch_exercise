import itertools
# from Ipython.display import Image
# from Ipython import display
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

cuda = torch.device('cuda')

class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.conv1 = nn.Conv2d(1,32,5,padding=2)
        self.conv2 = nn.Conv2d(32,64,5,padding=2)
        self.fc1 = nn.Linear(64*7*7, 1024)
        self.fc2 = nn.Linear(1024,10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = mymodel()
model = model.cuda()

batch_size = 50
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=1000
)

optimizer = optim.Adam(model.parameters(), lr = 0.0001)

model.train()

train_loss = []
train_accu = []
i = 0
for epoch in range(15):
    for data, target in train_loader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward() # loss 계산
        # print(loss)
        train_loss.append(loss.data)
        optimizer.step()
        prediction = output.data.max(1)[1]
        accuracy = (float)(prediction.eq(target.data).sum())/batch_size*100
        # print("prediction.eq(target.data) : {}".format(prediction.eq(target.data)))
        # print("prediction.eq(target.data).sum() : {}".format(prediction.eq(target.data).sum()))

        train_accu.append(accuracy)
        if i % 1000 == 0:
            print('Train Step : {}\tLoss : {}\tAccuracy : {:.3f}'.format(i, loss.data, accuracy))
        i += 1

plt.plot(np.arange(len(train_loss)), train_loss)
plt.plot(np.arange(len(train_accu)), train_accu)
        
model.eval()
correct = 0
for data, target in test_loader:
    data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda())
    output = model(data)
    prediction = output.data.max(1)[1] # max return max value and column index


    correct += prediction.eq(target.data).sum()

print('\tTest set : Accuracy : {:.2f}%'.format(100. * correct.data/ len(test_loader.dataset)))
