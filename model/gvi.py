import torch
import torch.nn as nn
import torch.nn.functional as F

class DIV(nn.Module): #定义除法模块
    def __init__(self):
        self.dense1 = nn.Linear(2,128,bias=True)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(128,64,bias=True)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(64,32,bias=True)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(32,1,bias=True)

    def forward(self,x):
        output = self.dense1(x)
        output = self.relu1(output)
        output = self.dense2(output)
        output = self.relu2(output)
        output = self.dense3(output)
        output = self.relu3(output)
        output = self.out(output)
        return output

class GVI(nn.Module):
    def __init__(self):
        self.linear1 = nn.Linear(3,1,bias=False)
        self.linear2 = nn.Linear(3,1,bias=False)
        self.div = DIV()
        self.relu = nn.ReLU()

    def forward(self,x):
        output = torch.cat([self.linear1(x), self.linear2(x)],1)
        output = self.div(output)
        output = self.relu(output)
        return output