import torch, pdb, pickle 
import torch.distributions as dd

class PolicyModel(torch.nn.Module):
    def __init__(self, dropout_ratio=0.0, num_conv_filters = 32, hidden_dim=47, shuffle_dict=None, num_tickers = -1, device=torch.device('cuda')): #num_conv_filters = 16, hidden_dim=32):
        super(PolicyModel, self).__init__()
        self.num_conv_filters = num_conv_filters
        self.hidden_dim = hidden_dim
        self.adaptive_max_pool_output = 10 
        
        # Define the layers of the model
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=num_conv_filters, kernel_size=3),
            torch.nn.Softplus(), #Tanh(), #ReLU(),
            torch.nn.AdaptiveMaxPool1d(output_size=self.adaptive_max_pool_output),
            torch.nn.Flatten(1, 2)
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(num_conv_filters * self.adaptive_max_pool_output, hidden_dim),
            torch.nn.Tanh(),
        )
        self.fc1_dropout = torch.nn.Dropout(p=dropout_ratio)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=0)

        self.shuffle_dict = shuffle_dict
        self.num_tickers = num_tickers
        self.device = device
        
        # Define the log of the standard deviation as an nn.Parameter 
        # Using log_std helps ensure std remains positive after optimization 
        std = torch.nn.Parameter(0.5 / self.num_tickers * torch.ones((self.num_tickers, 1))) 

        # Define the mean as an nn.Parameter
        mean = torch.nn.Parameter(1.0 / self.num_tickers * torch.ones((self.num_tickers, 1))) 

        self.noise_dist = dd.Normal(loc=mean, scale=std)


    def forward(self, x, tickers, return_acts = False):
        #x = torch.nn.functional.layer_norm(x, x.shape[1:])
        x = torch.unsqueeze(x, 1)  # Add a channel dimension

        ## normalization: division by max method 
        ma, idx = torch.max(x, dim=2)
        ma = ma.unsqueeze(2)
        x = x / ma 
        #x = torch.nn.functional.normalize(x, p=1.0, dim=2) # l1 norm method 

        x = self.conv1(x)
        x = self.fc1(x)
        x = self.fc1_dropout(x)
        acts = x # for the RL shuffled outputs, we will use max-pooling to obtain input for the value function, so the order doesn't matter 
        x = self.fc2(x)

        """
        mask = []
        for t in tickers: 
            if t in self.shuffle_dict:
                mask.append(True)
            else: 
                mask.append(False)
        mask = torch.Tensor(mask)
        pdb.set_trace()
        x = x[mask]
        tickers = tickers[mask]
        """
        
        if self.shuffle_dict is not None: 
            indices = []
            for t in tickers: 
                indices.append(self.shuffle_dict[t])
            indices = torch.Tensor(indices).to(int) 
            
            base_frame = -1.0 * float("Inf") * torch.ones((self.num_tickers, x.shape[1])).to(self.device)
            base_frame[indices] = x 
            x = base_frame 
        
        x = self.sm(x)

        e = self.noise_dist.rsample((1,)).to(self.device).squeeze(0) 
        # note the squeeze op in the last line to fix sample shape 
        x = x + e 

        x = self.sm(x) 

        if return_acts is True: 
            return x, acts
        else: 
            return x 
