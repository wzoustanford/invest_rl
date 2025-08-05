import torch
import os, sys, pdb 

class ValueModel(torch.nn.Module):
    def __init__(self, state_hdim, action_dim): 
        super(ValueModel, self).__init__()
        self.state_hdim = state_hdim 
        self.action_dim = action_dim 
        self.hdim = 512 
        self.hdim2 = 256 
        self.layer_1 = torch.nn.Linear(state_hdim + action_dim, self.hdim) 
        self.layer_1_act = torch.nn.Softplus() #torch.nn.Tanh() 
        self.layer_2 = torch.nn.Linear(self.hdim, self.hdim2)
        self.layer_2_act = torch.nn.Softplus()
        self.layer_3 = torch.nn.Linear(self.hdim2, 1)
    
    def forward(self, states, action):
        #states = states.view((1, -1))
        #action = action.view((1, -1))
        
        x = torch.cat((states, action), dim = 1)
        
        x = self.layer_1(x) 
        x = self.layer_1_act(x) 
        x = self.layer_2(x) 
        x = self.layer_2_act(x)
        x = self.layer_3(x)
        return x 
    