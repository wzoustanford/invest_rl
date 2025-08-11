"""
LSTM Sequential Model Architecture
Combines convolutional feature extraction from IIMODEL with LSTM for sequential processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFeatureExtractor(nn.Module):
    """
    Convolutional feature extractor based on IIMODEL architecture.
    Extracts features without the final portfolio weight output.
    Includes hidden layer mapping to 47 dimensions exactly as in IIMODEL.
    """
    def __init__(self, num_conv_filters=32, hidden_dim=47, dropout_ratio=0.0):
        super(ConvFeatureExtractor, self).__init__()
        self.num_conv_filters = num_conv_filters
        self.hidden_dim = hidden_dim
        self.adaptive_max_pool_output = 10
        
        # Convolutional layers from IIMODEL
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=num_conv_filters, kernel_size=3),
            nn.Softplus(),
            nn.AdaptiveMaxPool1d(output_size=self.adaptive_max_pool_output),
            nn.Flatten(1, 2)
        )
        
        # Hidden layer exactly as in IIMODEL (maps 320 -> 47)
        self.fc1 = nn.Sequential(
            nn.Linear(num_conv_filters * self.adaptive_max_pool_output, hidden_dim),
            nn.Tanh(),
        )
        
        self.fc1_dropout = nn.Dropout(p=dropout_ratio)
        
    def forward(self, x):
        """
        Extract convolutional features from input.
        
        Args:
            x: Input tensor of shape (batch_size, feature_dim)
            
        Returns:
            Hidden layer features of shape (batch_size, hidden_dim)
        """
        # Add channel dimension
        x = torch.unsqueeze(x, 1)  # (batch_size, 1, feature_dim)
        
        # Normalize by max value (as in IIMODEL)
        ma, idx = torch.max(x, dim=2, keepdim=True)
        x = x / (ma + 1e-10)
        
        # Apply convolution
        x = self.conv1(x)  # (batch_size, num_conv_filters * adaptive_max_pool_output)
        
        # Apply hidden layer (exactly as in IIMODEL)
        x = self.fc1(x)  # (batch_size, hidden_dim)
        x = self.fc1_dropout(x)
        
        return x


class LSTMSequentialModel(nn.Module):
    """
    LSTM-based sequential model for portfolio optimization.
    Combines convolutional feature extraction with LSTM for temporal modeling.
    """
    def __init__(self, 
                 num_conv_filters=32,
                 hidden_dim=47,
                 lstm_hidden_dim=64,
                 num_lstm_layers=2,
                 dropout_ratio=0.0,
                 num_timesteps=7):
        """
        Initialize the LSTM Sequential Model.
        
        Args:
            num_conv_filters: Number of convolutional filters
            hidden_dim: Hidden dimension after conv layers (from IIMODEL)
            lstm_hidden_dim: Hidden dimension of LSTM
            num_lstm_layers: Number of stacked LSTM layers
            dropout_ratio: Dropout probability
            num_timesteps: Number of sequential timesteps to process
        """
        super(LSTMSequentialModel, self).__init__()
        
        self.num_conv_filters = num_conv_filters
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.num_timesteps = num_timesteps
        
        # Convolutional feature extractor with hidden layer
        self.conv_extractor = ConvFeatureExtractor(
            num_conv_filters=num_conv_filters,
            hidden_dim=hidden_dim,
            dropout_ratio=dropout_ratio
        )
        
        # LSTM layers now take hidden_dim (47) as input
        self.lstm = nn.LSTM(
            input_size=hidden_dim,  # Changed from conv_output_dim to hidden_dim
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_ratio if num_lstm_layers > 1 else 0
        )
        
        # Output projection layer
        self.output_projection = nn.Linear(lstm_hidden_dim, 1)
        
        # Softmax for portfolio weights
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x_sequence, return_all_timesteps=True):
        """
        Forward pass through the model.
        
        Args:
            x_sequence: List or tensor of shape (num_timesteps, num_stocks, feature_dim)
                       or list of tensors each of shape (num_stocks, feature_dim)
            return_all_timesteps: If True, return weights for all timesteps; 
                                 if False, return only final timestep
                                 
        Returns:
            If return_all_timesteps=True: List of portfolio weights for each timestep
            If return_all_timesteps=False: Portfolio weights for final timestep
            Each weight tensor has shape (num_stocks, 1)
        """
        # Handle both list and tensor inputs
        if isinstance(x_sequence, list):
            num_timesteps = len(x_sequence)
            num_stocks = x_sequence[0].shape[0]
        else:
            num_timesteps, num_stocks, _ = x_sequence.shape
            x_sequence = [x_sequence[t] for t in range(num_timesteps)]
        
        # Process each timestep through conv extractor
        conv_features = []
        for t in range(num_timesteps):
            features = self.conv_extractor(x_sequence[t])  # (num_stocks, hidden_dim)
            conv_features.append(features)
        
        # Stack features for LSTM processing
        lstm_input = torch.stack(conv_features, dim=1)  # (num_stocks, num_timesteps, hidden_dim)
        
        # Process through LSTM
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)  # lstm_out: (num_stocks, num_timesteps, lstm_hidden_dim)
        
        # Generate portfolio weights for each timestep
        all_weights = []
        
        if return_all_timesteps:
            for t in range(num_timesteps):
                # Get LSTM output for timestep t
                lstm_t = lstm_out[:, t, :]  # (num_stocks, lstm_hidden_dim)
                
                # Project to single value
                logits = self.output_projection(lstm_t)  # (num_stocks, 1)
                
                # Apply softmax to get portfolio weights
                weights = self.softmax(logits)
                all_weights.append(weights)
            
            return all_weights
        else:
            # Only return final timestep weights
            lstm_final = lstm_out[:, -1, :]  # (num_stocks, lstm_hidden_dim)
            logits = self.output_projection(lstm_final)  # (num_stocks, 1)
            weights = self.softmax(logits)
            return weights
    
    def forward_with_hidden(self, x_sequence):
        """
        Forward pass that also returns hidden states.
        Useful for analysis and debugging.
        
        Args:
            x_sequence: List or tensor of input sequences
            
        Returns:
            Tuple of (weights_list, lstm_hidden_states, conv_features)
        """
        # Handle input format
        if isinstance(x_sequence, list):
            num_timesteps = len(x_sequence)
            num_stocks = x_sequence[0].shape[0]
        else:
            num_timesteps, num_stocks, _ = x_sequence.shape
            x_sequence = [x_sequence[t] for t in range(num_timesteps)]
        
        # Process through conv extractor
        conv_features = []
        for t in range(num_timesteps):
            features = self.conv_extractor(x_sequence[t])
            conv_features.append(features)
        
        # Stack for LSTM
        lstm_input = torch.stack(conv_features, dim=1)
        
        # Process through LSTM
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # Generate weights for all timesteps
        all_weights = []
        for t in range(num_timesteps):
            lstm_t = lstm_out[:, t, :]
            logits = self.output_projection(lstm_t)
            weights = self.softmax(logits)
            all_weights.append(weights)
        
        return all_weights, lstm_out, conv_features


class LSTMSequentialTrainer:
    """
    Training utilities for the LSTM Sequential Model with Sharpe ratio optimization.
    """
    def __init__(self, model, device='cuda', gamma=0.3, learning_rate=0.001):
        """
        Initialize the trainer.
        
        Args:
            model: LSTMSequentialModel instance
            device: Device to run training on
            gamma: Discount factor for temporal Sharpe ratios
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def compute_sharpe_loss(self, weights_list, price_series_list):
        """
        Compute the negative discounted Sharpe ratio loss.
        
        Args:
            weights_list: List of portfolio weights for each timestep
            price_series_list: List of price series for each timestep
                              Each has shape (num_stocks, num_days)
                              
        Returns:
            Scalar loss (negative discounted Sharpe ratio)
        """
        total_loss = torch.tensor(0.0).to(self.device)
        num_timesteps = len(weights_list)
        
        for t in range(num_timesteps):
            weights = weights_list[t]  # (num_stocks, 1)
            series = price_series_list[t]  # (num_stocks, num_days)
            
            # Calculate shares based on initial prices
            initial_prices = series[:, 0:1] + 1e-10
            shares = weights / initial_prices
            
            # Calculate returns
            final_prices = series[:, -1:]
            total_return = torch.sum((final_prices - initial_prices) * shares)
            
            # Calculate daily returns for Sharpe ratio
            daily_returns = []
            for day in range(1, series.shape[1]):
                daily_ret = torch.sum((series[:, day:day+1] - series[:, day-1:day]) * shares)
                daily_returns.append(daily_ret)
            
            if len(daily_returns) > 0:
                returns_tensor = torch.stack(daily_returns)
                sharpe = total_return / (torch.std(returns_tensor) + 1e-10)
            else:
                sharpe = total_return
            
            # Apply gamma discounting (earlier timesteps have higher weight)
            discount = self.gamma ** t
            total_loss = total_loss - sharpe * discount
        
        return total_loss
    
    def train_step(self, data_sequence):
        """
        Perform a single training step.
        
        Args:
            data_sequence: List of dictionaries containing 'features' and 'price_series'
                          for each timestep
                          
        Returns:
            Loss value for this step
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Extract features and price series
        features_list = []
        price_series_list = []
        
        for data in data_sequence:
            features_list.append(data['features'].to(self.device))
            price_series_list.append(data['price_series'].to(self.device))
        
        # Forward pass
        weights_list = self.model(features_list, return_all_timesteps=True)
        
        # Compute loss
        loss = self.compute_sharpe_loss(weights_list, price_series_list)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent explosion (using conventional max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data_sequence):
        """
        Evaluate the model on a sequence.
        
        Args:
            data_sequence: List of dictionaries containing test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Extract features and price series
            features_list = []
            price_series_list = []
            
            for data in data_sequence:
                features_list.append(data['features'].to(self.device))
                price_series_list.append(data['price_series'].to(self.device))
            
            # Forward pass
            weights_list = self.model(features_list, return_all_timesteps=True)
            
            # Calculate returns for each timestep
            timestep_returns = []
            timestep_sharpes = []
            
            for t in range(len(weights_list)):
                weights = weights_list[t]
                series = price_series_list[t]
                
                initial_prices = series[:, 0:1] + 1e-10
                shares = weights / initial_prices
                final_prices = series[:, -1:]
                
                total_return = torch.sum((final_prices - initial_prices) * shares).item()
                
                # Calculate Sharpe
                daily_returns = []
                for day in range(1, series.shape[1]):
                    daily_ret = torch.sum((series[:, day:day+1] - series[:, day-1:day]) * shares).item()
                    daily_returns.append(daily_ret)
                
                if len(daily_returns) > 0:
                    sharpe = total_return / (torch.std(torch.tensor(daily_returns)).item() + 1e-10)
                else:
                    sharpe = 0.0
                
                timestep_returns.append(total_return)
                timestep_sharpes.append(sharpe)
            
            # Calculate discounted metrics
            discounted_return = sum(r * (self.gamma ** t) for t, r in enumerate(timestep_returns))
            discounted_sharpe = sum(s * (self.gamma ** t) for t, s in enumerate(timestep_sharpes))
            
        return {
            'timestep_returns': timestep_returns,
            'timestep_sharpes': timestep_sharpes,
            'discounted_return': discounted_return,
            'discounted_sharpe': discounted_sharpe,
            'avg_return': sum(timestep_returns) / len(timestep_returns),
            'avg_sharpe': sum(timestep_sharpes) / len(timestep_sharpes)
        }


if __name__ == "__main__":
    # Test the model
    print("Testing LSTM Sequential Model...")
    
    # Create model
    model = LSTMSequentialModel(
        num_conv_filters=32,
        hidden_dim=47,  # Matches IIMODEL hidden layer
        lstm_hidden_dim=64,
        num_lstm_layers=2,
        dropout_ratio=0.1,
        num_timesteps=7
    )
    
    # Test forward pass
    batch_size = 100  # number of stocks
    feature_dim = 240
    num_timesteps = 7
    
    # Create dummy input
    x_sequence = [torch.randn(batch_size, feature_dim) for _ in range(num_timesteps)]
    
    # Forward pass
    weights = model(x_sequence, return_all_timesteps=True)
    
    print(f"Model created successfully")
    print(f"Input shape: {num_timesteps} timesteps x ({batch_size}, {feature_dim})")
    print(f"Output: {len(weights)} timesteps, each with shape {weights[0].shape}")
    print(f"Weights sum for first timestep: {weights[0].sum().item():.4f}")
    
    # Test trainer
    trainer = LSTMSequentialTrainer(model, device='cpu', gamma=0.3)
    print("\nTrainer created successfully")