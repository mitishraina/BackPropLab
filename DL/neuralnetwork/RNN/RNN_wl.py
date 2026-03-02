import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # fully connected layer for final output
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        # forward propogate RNN
        out, _ = self.rnn(x, h0)
        
        # output shape: (batch_size, seq_len, hidden_size)
        # we take the last time step output
        out = out[:, -1, :]
        
        # pass thru fully connected layer
        out = self.fc(out)
        
        return out