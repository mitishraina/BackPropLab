import torch
import torch.nn as nn
import torch.optim as optim

# 1. Create synthetic dataset
X = torch.randn(1000, 20)
y = torch.randint(0, 2, (1000,))


# 2. Define neural network
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        return self.model(x)
    
model = NN()

# 3. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. training loop
epochs = 20
batch_size = 32

for epoch in range(epochs):
    permutation = torch.randperm(X.size()[0])
    
    epoch_loss = 0.0
    
    for i in range(0, X.size()[0], batch_size):
        indices = permutation[i: i+batch_size]
        batch_x, batch_y = X[indices], y[indices]
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    print(f"epoch [{epoch+1}/{epochs}], loss: {epoch_loss:.4f}")
print("Training complete!")