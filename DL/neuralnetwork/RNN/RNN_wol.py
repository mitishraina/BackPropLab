from .model import RNN
import numpy as np

def generate_data(n_samples=500, seq_len=10):
    X = np.random.randint(0, 2, size=(n_samples, seq_len, 1))
    y = (np.sum(X, axis=1) > (seq_len // 2)).astype(int).flatten()
    return X.astype(float), y

X, y = generate_data()

model = RNN(
    input_size=1,
    hidden_size=16,
    output_size=2,
    learning_rate=0.01
)

for epoch in range(50):
    logits = model.forward(X)
    loss = model.compute_loss(logits, y)
    model.backward(y)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")