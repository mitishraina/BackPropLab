import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = self.num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
        
    def forward(self, x):
        # initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # forward propogate lstm
        out, _ = self.lstm(x, (h0, c0))
        
        # take output from last time step
        out = out[:, -1, :]
        
        # final classification layer
        out = self.fc(out)
        return out