import torch, pdb

class IIMODEL(torch.nn.Module):
    def __init__(self, dropout_ratio=0.0, num_conv_filters = 64, hidden_dim=256):
        super(IIMODEL, self).__init__()
        self.num_conv_filters = num_conv_filters
        self.hidden_dim = hidden_dim
        self.adaptive_max_pool_output = 10 
        
        # Define the layers of the model
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=num_conv_filters, kernel_size=3),
            torch.nn.ReLU(),
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

    def forward(self, x):
        #x = torch.nn.functional.layer_norm(x, x.shape[1:])
        x = torch.unsqueeze(x, 1)  # Add a channel dimension
        x = self.conv1(x)
        x = self.fc1(x)
        x = self.fc1_dropout(x)
        x = self.fc2(x)
        x = self.sm(x)
        return x

class IIMODELMARGIN(torch.nn.Module):
    def __init__(self, mask, dropout_ratio=0.0, num_conv_filters = 64, hidden_dim=256):
        super(IIMODELMARGIN, self).__init__()
        self.num_conv_filters = num_conv_filters
        self.hidden_dim = hidden_dim
        self.adaptive_max_pool_output = 10 
        
        # Define the layers of the model
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=num_conv_filters, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(output_size=self.adaptive_max_pool_output),
            torch.nn.Flatten(1, 2)
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(num_conv_filters * self.adaptive_max_pool_output, hidden_dim),
            torch.nn.Tanh(),
        )
        self.fc1_dropout = torch.nn.Dropout(p=dropout_ratio)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.fc2_margin = torch.nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=0)
        self.sm_margin = torch.nn.Softmax(dim=0)
        self.b = 0.5 #torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.mask = mask # this is a tensor with a list of booleans the same length as the number of stocks 

    def forward(self, x):
        device = x.device
        #x = torch.nn.functional.layer_norm(x, x.shape[1:])
        x = torch.unsqueeze(x, 1)  # Add a channel dimension
        x = self.conv1(x)
        x = self.fc1(x)
        x = self.fc1_dropout(x)
        y = x[self.mask.bool()]
        x = self.fc2(x)
        y = self.fc2_margin(y)
        x = self.sm(x)
        y = self.sm_margin(y)
        y_samesize = torch.zeros(x.shape).to(device)
        y_samesize[self.mask.bool()] = y
        scores = x - self.b * y_samesize
        #if self.b > 0.99: 
        #    self.b.requires_grad = False
        #print(f'b value: {self.b}')
        scores = scores / (1 - self.b)
        return scores

class IIMODELWITHNEWS(torch.nn.Module):
    def __init__(self, dropout_ratio=0.0, num_conv_filters = 64, hidden_dim=256):
        super(IIMODELWITHNEWS, self).__init__()
        self.num_conv_filters = num_conv_filters
        self.hidden_dim = hidden_dim
        self.adaptive_max_pool_output = 10 
        
        # Define the layers of the model
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=num_conv_filters, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(output_size=self.adaptive_max_pool_output),
            torch.nn.Flatten(1, 2)
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(num_conv_filters * self.adaptive_max_pool_output, hidden_dim),
            torch.nn.Tanh(),
        )

        self.fc_news = torch.nn.Sequential(
            torch.nn.Linear(3072, hidden_dim), 
            torch.nn.Tanh(),
        )

        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            torch.nn.Tanh(),
        )
        self.fc2_dropout = torch.nn.Dropout(p=dropout_ratio)

        self.fc3 = torch.nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, x, nf):
        #x = torch.nn.functional.layer_norm(x, x.shape[1:])
        x = torch.unsqueeze(x, 1)  # Add a channel dimension
        x = self.conv1(x)
        x = self.fc1(x)

        nf = self.fc_news(nf)

        x = torch.concat([x, nf], dim=1)

        x = self.fc2(x)
        x = self.fc2_dropout(x)
        x = self.fc3(x)
        x = self.sm(x)

        return x

class IIMODELCLASSIC(torch.nn.Module):
    """
    the classic model with around ~20% annual returns on 25d aggregate over 4 years 
    """
    def __init__(self, dropout_ratio=0.0, num_conv_filters = 64, hidden_dim=256):
        super(IIMODELCLASSIC, self).__init__()
        self.num_conv_filters = num_conv_filters
        self.hidden_dim = hidden_dim
        self.adaptive_max_pool_output = 10 
        
        # Define the layers of the model
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=num_conv_filters, kernel_size=3),
            torch.nn.ReLU(),
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

    def forward(self, x):
        #x = torch.nn.functional.layer_norm(x, x.shape[1:])
        x = torch.unsqueeze(x, 1)  # Add a channel dimension
        x = self.conv1(x)
        x = self.fc1(x)
        x = self.fc1_dropout(x)
        x = self.fc2(x)
        x = self.sm(x)
        return x
