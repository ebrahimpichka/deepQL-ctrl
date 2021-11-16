import torch as T
import torch.nn.functional as F 
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os




class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, l1_dim, l2_dim, num_actions,
                                    name,
                                    chkpt_dir):
        super(DeepQNetwork, self).__init__()

        self.lr = lr
        self.input_dims = input_dims
        self.l1_dim = l1_dim
        self.l2_dim = l2_dim
        self.num_actions = num_actions

        self.checkpoint_dir = chkpt_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(self.input_dims[0], self.l1_dim)
        self.fc2 = nn.Linear(self.l1_dim, self.l2_dim)
        self.fc3 = nn.Linear(self.l2_dim, self.num_actions)

        self.optimizer = optim.Adam(self.parameters(), self.lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        act_val = self.fc3(x)

        return(act_val)

    def save_checkpoint(self):
        
        T.save(self.state_dict(), self.checkpoint_file)
        print('Checkpoint Saved!')

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        print('Checkpoint Loaded!')