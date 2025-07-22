import torch
import os, sys, pdb 

class ValueModel(torch.nn.Module):
    def __init__(self, state_hdim, action_dim): 
        super(ValueModel, self).__init__()
        self.state_hdim = state_hdim 
        self.action_dim = action_dim 
        self.hdim = 128 
        self.layer_1 = torch.nn.Linear(state_hdim + action_dim, self.hdim) 
        self.layer_1_act = torch.nn.Tanh() 
        self.layer_2 = torch.nn.Linear(self.hdim, 1)
    
    def forward(self, states, action):
        #states = states.view((1, -1))
        #action = action.view((1, -1))
        
        x = torch.cat((states, action), dim = 1)
        
        x = self.layer_1(x) 
        x = self.layer_1_act(x) 
        x = self.layer_2(x) 
        return x 
    