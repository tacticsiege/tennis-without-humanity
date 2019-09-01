import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """ Actor model for agent policy (pi). """
    def __init__(self, state_size, action_size, seed, layers=[64, 64], batch_norm=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            layers [(int)]: Number of nodes in each hidden layer
            batch_norm: Use batch normalization, default: False
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.batch_norm = batch_norm

        # build network
        self.input_layer = nn.Linear(state_size, layers[0])

        self.hidden = nn.ModuleList()
        for i in range(len(layers)-1):
            self.hidden.append(nn.Linear(layers[i], layers[i+1]))
        
        self.out_layer = nn.Linear(layers[-1], action_size)

        if self.batch_norm:
            self.bn_layer = nn.BatchNorm1d(layers[0])

        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        for layer in self.hidden:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.out_layer.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, state):
        """ Forward pass to map state -> action values."""
        x = self.input_layer(state)
        if self.batch_norm:
            x = self.bn_layer(x)
        x = F.relu(x)

        for layer in self.hidden:
            x = F.relu(layer(x))

        return torch.tanh(self.out_layer(x))

class Critic(nn.Module):
    """ Critic model for agent action-value function (Q^pi). """
    def __init__(self, state_size, action_size, seed, layers=[64, 64], batch_norm=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.batch_norm = batch_norm

        # build network
        self.input_layer = nn.Linear(state_size, layers[0])
        if self.batch_norm:
            self.bn_layer = nn.BatchNorm1d(layers[0])

        self.hidden = nn.ModuleList()
        # concat output from fully connected state input layer with actions for hidden layer input
        self.hidden.append(nn.Linear(layers[0]+action_size, layers[1]))

        for i in range(1, len(layers)-1):
            self.hidden.append(nn.Linear(layers[i], layers[i+1]))

        # output single Q-value
        self.out_layer = nn.Linear(layers[-1], 1)

        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        for layer in self.hidden:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.out_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """ Forward pass to map (state, action) -> Q-value """
        xs = self.input_layer(state)
        if self.batch_norm:
            xs = self.bn_layer(xs)
        xs = F.relu(xs)

        # concat fully connected state output with actions for hidden layer input
        x = torch.cat((xs, action), dim=1)

        for layer in self.hidden:
            x = F.relu(layer(x))

        # return single Q value without activation (logits)
        return self.out_layer(x)