import argparse
import os
import time
import shutil
from collections import OrderedDict
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# required to download pretrained model
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import warnings
warnings.filterwarnings("ignore")

def relu(v):
    return np.maximum(0, v)

def load_model():
    class MLP(nn.Module):
        def __init__(self, input_dims, n_hiddens, n_class, display=False):
            super(MLP, self).__init__()
            assert isinstance(input_dims, int), 'Please provide int for input_dims'
            self.input_dims = input_dims
            current_dims = input_dims
            layers = OrderedDict()

            if isinstance(n_hiddens, int):
                n_hiddens = [n_hiddens]
            else:
                n_hiddens = list(n_hiddens)
            for i, n_hidden in enumerate(n_hiddens):
                layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
                layers['relu{}'.format(i+1)] = nn.ReLU()
    #             layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
                current_dims = n_hidden
            layers['out'] = nn.Linear(current_dims, n_class)

            self.model= nn.Sequential(layers)
            self.layers = layers
            if display:
                print(self.model)

        def forward(self, input):
            input = input.view(input.size(0), -1)
            assert input.size(1) == self.input_dims
            return self.model.forward(input)

    def mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=False):
        model_urls = {
            'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth'
        }
        model = MLP(input_dims, n_hiddens, n_class)
        m = model_zoo.load_url(model_urls['mnist'], map_location=torch.device('cpu'))
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        model.load_state_dict(state_dict)
        return model


    def extract_W_B():
        model = mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=True)
        W = []
        B = []
        for i in range(3):
            layer_name = f'fc{i+1}'
            if i == 2:
                layer_name = 'out'
            layer = dict(model.layers)[layer_name]
            W.append(layer.weight.detach().numpy().T)
            B.append(layer.bias.detach().numpy())
        return W, B

    return extract_W_B()

def load_data():
    mn = datasets.MNIST(root='tmp/public_dataset/pytorch/mnist-data', train=False, download=True)
    x = list()
    y = list()
    for image, label in mn:
        x.append(np.array(image))
        y.append(label)
    x = np.array(x) / 255
    y = np.array(y)
    return x, y