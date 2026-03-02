import torch
import torch.nn as nn

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        
        # Update gate
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Reset gate
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Candidate hidden state
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h_prev):
        z = torch.sigmoid(self.W_z(x) + self.U_z(h_prev))
        r = torch.sigmoid(self.W_r(x) + self.U_r(h_prev))
        
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(r * h_prev))
        h = (1 - z) * h_prev + z * h_tilde
        
        return h


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)

    def forward(self, x, h0=None):
        # x: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h0
        
        outputs = []
        
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
            outputs.append(h.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)
        
        return outputs, h
    
    
# import torch
# import torch.nn as nn

# class GRUModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1):
#         super(GRUModel, self).__init__()
        
#         self.gru = nn.GRU(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True
#         )

#     def forward(self, x, h0=None):
#         # x: (batch_size, seq_len, input_size)
        
#         if h0 is None:
#             output, h_n = self.gru(x)
#         else:
#             output, h_n = self.gru(x, h0)
            
#         return output, h_n


# # Example usage
# if __name__ == "__main__":
#     batch_size = 4
#     seq_len = 10
#     input_size = 8
#     hidden_size = 16

#     x = torch.randn(batch_size, seq_len, input_size)

#     model = GRUModel(input_size, hidden_size)
#     output, h_n = model(x)

#     print("Output shape:", output.shape)  # (4, 10, 16)
#     print("Hidden state shape:", h_n.shape)  # (1, 4, 16)