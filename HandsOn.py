import torchvision as tv                                       
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torchvision as tv
import operator
data = tv.datasets.FashionMNIST(root = "data", download=True)  

#import matplotlib.pyplot as plt                                
#for image, label in data:                                      
#    plt.imshow(image)                                          
#    plt.show()                                                 


data = tv.datasets.FashionMNIST(root="data", download=True)
batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(len(data))), 100, False)   
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, 5)
        self.conv2 = nn.Conv2d(100, 50, 5)
        self.conv3 = nn.Conv2d(50, 5, 5)
        self.fc1 = nn.Linear(5 * 256, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.fc1(x.flatten(start_dim = 1)))
        return self.fc2(x)

model = Model()
model.cuda()
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())

for e in range(100):
    for idx in batch_sampler:
        batch = operator.itemgetter(*idx)(data)                    
        x = []                                                     
        y_true = []                                                
        for image, label in batch:                                 
            x.append(np.asarray(image))                            
            y_true.append(label)                                   
        x = np.stack(x)[:, np.newaxis]                             
        x_tensor = torch.tensor(x, dtype=torch.float).cuda()
        y_true = np.stack(y_true)                                  
        y_true_tensor = torch.tensor(y_true, dtype=torch.long)     
        y_pred = model(x_tensor.cuda())
        l = loss(y_pred, y_true_tensor.cuda())
        optim.zero_grad()
        l.backward()
        optim.step()
        print(float(l))