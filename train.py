import torch as t
import torchvision as tv
from data import get_train_dataset, get_validation_dataset
from stopping import EarlyStoppingCallback
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from model.resnet import Resnet

#Hyperparameters
learning_rate = 0.00001
stop_patience = 1
batch_size = 50
#weight_decay = 0.01

# set up data loading for the training and validation set using t.utils.data.DataLoader and the methods implemented in data.py
train_dataset = get_train_dataset()
val_dataset = get_validation_dataset()
pos_weight_train = train_dataset.pos_weight()
pos_weight_val = val_dataset.pos_weight()

train_dl = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dl = t.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

# set up your model
net = Resnet()

# set up loss (you can find preimplemented loss functions in t.nn) use the pos_weight parameter to ease convergence
# set up optimizer (see t.optim); 
# initialize the early stopping callback implemented in stopping.py and create a object of type Trainer
#criterion = t.nn.BCELoss(weight=train_dataset.pos_weight())
criterion = t.nn.BCEWithLogitsLoss(pos_weight=pos_weight_train)
optimizer = t.optim.Adam(net.parameters(), lr=learning_rate)
es_cb = EarlyStoppingCallback(patience=stop_patience)
trainer = Trainer(net, criterion, optimizer, train_dl, val_dl, cuda=True, early_stopping_cb=es_cb)

# go, go, go... call fit on trainer
res = trainer.fit()

# plot the results
plt.figure()
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')