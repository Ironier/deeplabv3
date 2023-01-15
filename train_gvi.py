from datasets import DatasetTrain, DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)


import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("/root/deeplabv3/model")
from gvi import GVI, DIV

sys.path.append("/root/deeplabv3/utils")
from utils import add_weight_decay


num_epochs = 1000
learning_rate = 0.001
network = DIV()
params = add_weight_decay(network, l2_value=1e-6) #对参数进行预处理
optimizer = torch.optim.Adam(params, lr=learning_rate)
loss_fn = nn.MSELoss()

for epoch in range(num_epochs):
    a = np.random.randn(1024,1)
    b = np.random.randn(1024,1)
    c = a/b
    cat = np.concatenate([a,b],1)

    inputs = Variable(torch.FloatTensor(cat)).cuda()
    labels = Variable(torch.FloatTensor(c)).cuda()

    outputs = network(inputs)
    loss = loss_fn(outputs, labels)
    loss_value = loss.data.cpu().numpy()

    # optimization step:
    optimizer.zero_grad() # (reset gradients)
    loss.backward() # (compute gradients)
    optimizer.step() # (perform optimization step)

    checkpoint_path = "/root/gvi/div_epoch_" + str(epoch+1) + ".pth"
    torch.save(network.state_dict(), checkpoint_path)
