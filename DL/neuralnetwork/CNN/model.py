import numpy as np


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.W = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * 0.01
        self.b = np.zeros((out_channels, 1))

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        K = self.kernel_size

        out_h = H - K + 1
        out_w = W - K + 1

        self.out = np.zeros((N, self.out_channels, out_h, out_w))

        for n in range(N):
            for f in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        region = X[n, :, i:i+K, j:j+K]
                        self.out[n, f, i, j] = (
                            np.sum(region * self.W[f]) + self.b[f]
                        )

        return self.out

    def backward(self, d_out, lr):
        N, C, H, W = self.X.shape
        K = self.kernel_size

        dX = np.zeros_like(self.X)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        for n in range(N):
            for f in range(self.out_channels):
                for i in range(H - K + 1):
                    for j in range(W - K + 1):
                        region = self.X[n, :, i:i+K, j:j+K]

                        dW[f] += d_out[n, f, i, j] * region
                        dX[n, :, i:i+K, j:j+K] += (
                            d_out[n, f, i, j] * self.W[f]
                        )
                        db[f] += d_out[n, f, i, j]

        self.W -= lr * dW
        self.b -= lr * db

        return dX
    
class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, d_out):
        return d_out * (self.X > 0)
    
class MaxPool2D:
    def __init__(self, kernel_size=2):
        self.kernel_size = kernel_size

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        K = self.kernel_size

        out_h = H // K
        out_w = W // K

        self.out = np.zeros((N, C, out_h, out_w))

        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        region = X[n, c,
                                   i*K:(i+1)*K,
                                   j*K:(j+1)*K]
                        self.out[n, c, i, j] = np.max(region)

        return self.out

    def backward(self, d_out):
        N, C, H, W = self.X.shape
        K = self.kernel_size

        dX = np.zeros_like(self.X)

        for n in range(N):
            for c in range(C):
                for i in range(H // K):
                    for j in range(W // K):
                        region = self.X[n, c,
                                        i*K:(i+1)*K,
                                        j*K:(j+1)*K]

                        max_val = np.max(region)
                        mask = (region == max_val)

                        dX[n, c,
                           i*K:(i+1)*K,
                           j*K:(j+1)*K] += (
                            d_out[n, c, i, j] * mask
                        )

        return dX
    
class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros((1, out_features))

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward(self, d_out, lr):
        dW = np.dot(self.X.T, d_out)
        db = np.sum(d_out, axis=0, keepdims=True)
        dX = np.dot(d_out, self.W.T)

        self.W -= lr * dW
        self.b -= lr * db

        return dX

class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        self.y = y
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)

        N = logits.shape[0]
        loss = -np.mean(
            np.log(self.probs[range(N), y])
        )
        return loss

    def backward(self):
        N = self.probs.shape[0]
        d_logits = self.probs.copy()
        d_logits[range(N), self.y] -= 1
        return d_logits / N