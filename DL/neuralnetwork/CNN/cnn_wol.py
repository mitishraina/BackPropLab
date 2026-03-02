# this is built from model.py in this directory

from .model import Conv2D, ReLU, MaxPool2D, Linear, SoftmaxCrossEntropy

class SimpleCNN:
    def __init__(self):
        self.conv = Conv2D(1, 4, 3)
        self.relu = ReLU()
        self.pool = MaxPool2D(2)
        self.fc = Linear(4 * 13 * 13, 10)
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, X):
        out = self.conv.forward(X)
        out = self.relu.forward(out)
        out = self.pool.forward(out)

        self.flatten = out.reshape(out.shape[0], -1)
        logits = self.fc.forward(self.flatten)

        return logits

    def backward(self, d_out, lr):
        d_out = self.fc.backward(d_out, lr)
        d_out = d_out.reshape(-1, 4, 13, 13)

        d_out = self.pool.backward(d_out)
        d_out = self.relu.backward(d_out)
        self.conv.backward(d_out, lr)