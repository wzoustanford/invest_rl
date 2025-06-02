import torch, pdb

class IIMODEL(torch.nn.Module):
    def __init__(self, dropout_ratio=0.0, num_conv_filters = 32, hidden_dim=48): #num_conv_filters = 16, hidden_dim=32):
        super(IIMODEL, self).__init__()
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

    def forward(self, x):
        #x = torch.nn.functional.layer_norm(x, x.shape[1:])
        x = torch.unsqueeze(x, 1)  # Add a channel dimension

        # testing normalization 
        ## division by max method: 
        ma, idx = torch.max(x, dim=2)
        ma = ma.unsqueeze(2)
        x = x / ma 
        #x = torch.nn.functional.normalize(x, p=1.0, dim=2)
        
        x = self.conv1(x)
        x = self.fc1(x)
        x = self.fc1_dropout(x)
        x = self.fc2(x)
        x = self.sm(x)
        return x

class IIMODELWITHNEWS1CNORM(torch.nn.Module):
    def __init__(self, dropout_ratio=0.0, num_conv_filters = 32, hidden_dim=48): #num_conv_filters = 16, hidden_dim=32):
        super(IIMODELWITHNEWS, self).__init__()
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
        #self.fc1_dropout = torch.nn.Dropout(p=dropout_ratio)
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
        
        #self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, x, nf):
        #x = torch.nn.functional.layer_norm(x, x.shape[1:])
        x = torch.unsqueeze(x, 1)  # Add a channel dimension

        # testing normalization 
        ## division by max method: 
        ma, idx = torch.max(x, dim=2)
        ma = ma.unsqueeze(2)
        x = x / ma 
        #x = torch.nn.functional.normalize(x, p=1.0, dim=2)
        
        x = self.conv1(x)
        x = self.fc1(x)
        #x = self.fc1_dropout(x)
        nf = self.fc_news(nf)
        
        x = torch.concat([x, nf], dim=1)
        
        x = self.fc2(x)
        x = self.fc2_dropout(x)
        acts = x 
        x = self.fc3(x)
        x = self.sm(x)
        
        return x, acts

class IIMODEL2CLarge(torch.nn.Module):
    def __init__(self, dropout_ratio=0.0, num_conv_filters = 64, hidden_dim=256): #num_conv_filters = 32, hidden_dim=48): #num_conv_filters = 16, hidden_dim=32):
        super(IIMODEL, self).__init__()
        self.num_conv_filters = num_conv_filters
        self.hidden_dim = hidden_dim
        self.adaptive_max_pool_output = 20 
        
        # Define the layers of the model
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=num_conv_filters, kernel_size=3),
            torch.nn.Softplus(), #Tanh(), #ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=2),
            torch.nn.Conv1d(in_channels=num_conv_filters, out_channels=num_conv_filters, kernel_size=3),
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

        # testing normalization 
        ## division by max method: 
        ma, idx = torch.max(x, dim=2)
        ma = ma.unsqueeze(2)
        x = x / ma 
        #x = torch.nn.functional.normalize(x, p=1.0, dim=2)
        
        x = self.conv1(x)
        x = self.fc1(x)
        x = self.fc1_dropout(x)
        x = self.fc2(x)
        x = self.sm(x)
        return x
    
class IIMODELLARGER(torch.nn.Module):
    def __init__(self, dropout_ratio=0.0, num_conv_filters = 64, hidden_dim=256):
        super(IIMODEL, self).__init__()
        self.num_conv_filters = num_conv_filters
        self.hidden_dim = hidden_dim
        self.adaptive_max_pool_output = 10 
        
        # Define the layers of the model
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=num_conv_filters, kernel_size=3),
            torch.nn.Softplus(), #ReLU(),
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
        scores = x
        short_scores = - self.b * y_samesize
        #if self.b > 0.99: 
        #    self.b.requires_grad = False
        #print(f'b value: {self.b}')
        return scores, short_scores

class IIMODELWITHNEWS(torch.nn.Module):
    def __init__(self, dropout_ratio=0.0, num_conv_filters = 64, hidden_dim=256):
        super(IIMODELWITHNEWS, self).__init__()
        self.num_conv_filters = num_conv_filters
        self.hidden_dim = hidden_dim
        self.adaptive_max_pool_output = 10 
        
        # Define the layers of the model
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=num_conv_filters, kernel_size=3),
            torch.nn.ReLU(), #torch.nn.Softplus(beta=1), #
            torch.nn.AdaptiveMaxPool1d(output_size=self.adaptive_max_pool_output),
            torch.nn.Flatten(1, 2)
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(num_conv_filters * self.adaptive_max_pool_output, hidden_dim),
            torch.nn.Tanh(),
        )

        self.ln_news = torch.nn.LayerNorm(normalized_shape=3072, elementwise_affine=False)
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
        acts = x 
        x = self.fc3(x)
        x = self.sm(x)

        return x, acts 

class SELLACTIONMODEL(torch.nn.Module):
    def __init__(self):
        super(SELLACTIONMODEL, self).__init__()
        self.acts_hidden_dim = 256 
        self.hidden_dim = 64 
        self.sell_days_conv_num_filters = 32 
        self.conv_sell_days = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=2, out_channels=self.sell_days_conv_num_filters, kernel_size=1), 
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(output_size=1),
            torch.nn.Flatten(1, 2)
        )
        self.sell_portfolio_series_conv_num_filters = 4
        self.conv_sell_portfolio_series = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=2, out_channels=self.sell_portfolio_series_conv_num_filters, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(output_size=1),
            torch.nn.Flatten(1, 2)
        )
        self.additional_features = self.sell_days_conv_num_filters + self.sell_portfolio_series_conv_num_filters
        self.fc = torch.nn.Linear(self.acts_hidden_dim + self.additional_features, self.hidden_dim)
        self.fc_act = torch.nn.Tanh()
        self.sm_fc = torch.nn.Linear(self.hidden_dim, 4) ## 5 days, buy 1st day, 4 days could sell 
        self.sm_act = torch.nn.Softmax(dim=1)
    
    def forward(self, acts, sell_day_features, portfolio_return_series): 
        acts, _ = torch.max(acts, dim=0) ## max pool across all stocks price series signals 
        x = self.conv_sell_days(sell_day_features) 
        x, _ = torch.max(x, dim=0) ## max pool across all stocks sell days price series signals 
        x = torch.cat((acts.unsqueeze(0), x.unsqueeze(0)), dim = 1) 
        y = self.conv_sell_portfolio_series(portfolio_return_series) ## combine with portfolio returns with separate convnet 
        x = torch.cat((x, y), dim = 1) 
        x = self.fc(x) 
        x = self.fc_act(x) 
        x = self.sm_fc(x) 
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

class IIMODELWITHNEWSADDTIONALNORM(torch.nn.Module):
    def __init__(self, dropout_ratio=0.0, num_conv_filters = 64, hidden_dim=256):
        super(IIMODELWITHNEWSADDTIONALNORM, self).__init__()
        self.num_conv_filters = num_conv_filters
        self.hidden_dim = hidden_dim
        self.adaptive_max_pool_output = 10 
        
        num_conv_filters_norm_path = 32 

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

        self.conv1_np = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=num_conv_filters_norm_path, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(output_size=self.adaptive_max_pool_output),
            torch.nn.Flatten(1, 2)
        )
        self.fc1_np = torch.nn.Sequential(
            torch.nn.Linear(num_conv_filters_norm_path * self.adaptive_max_pool_output, hidden_dim),
            torch.nn.Tanh(),
        )

        self.ln_news = torch.nn.LayerNorm(normalized_shape=3072, elementwise_affine=False)
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

        ## division by max method: 
        #ma, idx = torch.max(x, dim=2)
        #ma = ma.unsqueeze(2)
        #y = x / ma 

        y = torch.nn.functional.normalize(x, p=1.0, dim=2)
        y = self.conv1_np(y)
        y = self.fc1_np(y)

        x = self.conv1(x)
        x = self.fc1(x)

        ## simply sum the unnormalized net output and normalized net output 
        x = x + y 

        nf = self.fc_news(nf)
        
        x = torch.concat([x, nf], dim=1)

        x = self.fc2(x)
        x = self.fc2_dropout(x)
        x = self.fc3(x)
        x = self.sm(x)

        return x
    
