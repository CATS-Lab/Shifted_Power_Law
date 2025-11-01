import torch.nn as nn
import torch


# Predict the longitudinal and lateral acceleration and corresponding standrad deviation of the ego vehicle using LSTM
class ego_acc_LSTM_dist(nn.Module):
    def __init__(self, num_feature = 5, hidden_size = 128, num_layers = 2, output_size = 4, NUMDIR = 2):
        super(ego_acc_LSTM_dist, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.NUMDIR = NUMDIR

        self.lstm = nn.LSTM(num_feature, hidden_size, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, output_size * self.NUMDIR)
        self.fc_sigma = nn.Linear(hidden_size, output_size * self.NUMDIR)


    def forward(self, x): # (B, num_feature, input_size)
        batch_size = x.shape[0]
        # Conver Nan to 0
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  # LSTM layer
        x = out[:, -1, :]
        
        # Mean value
        mu = self.fc_mu(x)  # (B, output_size*2)
        mu = mu.view(batch_size, -1, self.NUMDIR)
        
        # Standard deviation
        sigma = self.fc_sigma(x)  # (B, output_size*2)
        sigma = torch.exp(sigma) + 1e-9 # ensure positiveness
        sigma = sigma.view(batch_size, -1, self.NUMDIR)

        return [mu, sigma]