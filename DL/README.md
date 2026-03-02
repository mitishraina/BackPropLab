# Deep Learning
1958 - Perceptron by Frank RosenBlatt, it was first Neural Network Model

1969 - Marvin Minsky and Seymour Papert, they proved perceptrons couldn't solve certain problems

1970s-1980s - The AI Winter

1986 - BackPropogation Revival by Geofrrey Hinton

1997 - DeepBlue vs Garry Kasparov
and so on..


## Perceptron
- A single layer neural network having only input & output layers
- Limited to linearly separably problems
- [Perceptron](perceptron.py) learns a linear decision boundary: w.x + b = 0
- The perceptron model: **function x is calculated by some weight muliplied by input and add some bias to it**

$$
f(x) = w^T x + b
$$

where:

$$
w \in \mathbb{R}^d, \quad x \in \mathbb{R}^d, \quad b \in \mathbb{R}
$$
- Initialize hyperparameters:
    - lr: learning rate
    - n_iters: Epochs
    - weights: vector w
    - bias: vector b
    - error_: stores numbers of misclassifications
- Working of a ***Perceptron***: 
    1. initializes weights as w=0 and b=0
    2. For each sample $x_i$:

    $$z_i = w^T x_i + b$$

    3. Apply activation function
    - Prediction rule:

    $$\hat{y}_i =
    \begin{cases}
    1 & \text{if } z_i \ge 0 \\
    -1 & \text{if } z_i < 0
    \end{cases}$$

    4. Weight update rule
    - Weight update rule (if misclassified):

        $$w := w + \eta y_i x_i$$

        $$b := b + \eta y_i$$

    5. Prediction:

    $$z = w^T X + b$$

    $$\hat{y} = \text{sign}(z)$$

    ### Perceptron condition for correct classification:

    $$y_i (w^T x_i + b) > 0$$


## MLP
- Introduces Backpropogation which makes training computationally feasible
- In MLP, the output depends on many layers of weights
- Backpropogation solved one important question:
```
If the output is wrong, how do we know which hidden weight caused the error?
```
This is the credit assignment problem.
- [MLP](MLP.py), for this we use a two layer MLP, an input layer, one hidden layer, and one output neuron. For ```_initialize_hyperparameters```:
```bash
self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
```
This initializes the weights between input layer and hidden layer. This will create a matrix of:
``` number of input_features x number of hidden neurons ```
- ***Why multiply with 0.01?***
To scale random numbers down to small values
- ***Why small values?***
Large initial weights can:
    1. Cause exploding activations
    2. Make training unstable
So small random values help training start smoothly. Then
```bash
self.b1 = np.zeros((1, self.hidden_size))
```
This initializes the bias for hidden layer which creates a raw vectors of zeros. Bias is added to the weighted sum before applying activation. Typically, biases are initialized to zero.

```bash
self.W2 = np.random.randn(self.hidden_size, 1) * 0.01
self.b2 = np.zeros((1, 1))
```

This initializes the weight between hidden layer and output layer.

### Activation Functions

1. **ReLU (Rectified Linear Unit)**:
```
ReLU(z) = max(0,z)
```
What ReLU mathematically do is:
- If **z > 0** then return z
- if **z <= 0** then return 0

Compares each element in z with 0 (zero) and keeps the larger value.
- ***Why ReLU?***
    - very simple and fast
    - helps reducing vanishing gradient problem
    - commonly used in hidden layers

And then there is **ReLU derivative**, this computes the derivative of ReLU during backpropogation.

$$
\mathrm{ReLU}'(z)=
\begin{cases}
1 & \text{if } z > 0 \\
0 & \text{if } z \le 0
\end{cases}
$$

2. **Sigmoid Activation Function**: outputs values between 0 & 1. Mathematically
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Note that, ```Z = np.clip(Z, -500, 500)``` is written, this prevents numerical overflow. Clipping keeps values in safe range.**

1. ***Preparing the forward pass:***
```python
    def _forward(self, X):
        # computes the weighted sum for the first(hidden) layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        # apply relu activation function(sets -ve values to zero)
        self.A1 = self._relu(self.Z1)
        
        # computes the weighted sum for the output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        #apply sigmoid to squash values between 0 and 1
        self.A2 = self._sigmoid(self.Z2)
        
        return self.A2
```
where, *Z* = Linear combination of inputs or raw linear calculation, *A* = activated output, *X* = input data, *W* = weight, *B* = bias

So, Mathematically,

1. For first layer between input and hidden layer: $Z_1 = XW_1 + b_1$
2. For hidden layer and output layer: $Z_2 = A_1W_2 + b_2$. Here, $A_1$ becomes the input to this layer

To summarise this forward functions, it performs:
1. Input -> Hidden Layer
    - Linear Transformation
    - Applies ReLU activation
2. Hidden Layer -> Output Layer
    - Linear Transformation
    - Sigmoid Function

Hence, the full mathematical flow is:
$$
A_2 = \sigma\left(\text{ReLU}(XW_1 + b_1)W_2 + b_2\right)
$$

2. ***Compute Loss:***

This function computes the binary cross entropy loss (also called log loss)
```python
    def _compute_loss(self, y, y_hat):
        # Prevents numerical instability
        epsilon = 1e-15
        # this prevents math errors and exploding loss values
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        
        # compute binary cross entropy loss
        loss = -np.mean(
            y * np.log(y_hat) + 
            (1 - y) * np.log(1 - y_hat)
        )
        
        return loss
```
where, *y* = true labels (0 & 1), *y_hat* = predicted probabalities from the model(output of sigmoid)

Compute loss returns a single scalar value: the average loss

This below in ```_compute_loss``` function is actually,
```python 
    loss = -np.mean(
            y * np.log(y_hat) + 
            (1 - y) * np.log(1 - y_hat)
        )
```
This below mathematical notation.
$$
\text{Loss} = -\frac{1}{n} \sum_{i=1}^{n}
\left[
y_i \log(\hat{y}_i) +
(1 - y_i)\log(1 - \hat{y}_i)
\right]
$$

- ***What this does?*** 
For each example:
    - If the true label is 1: $-log(y)$ 

        -> This will punish small predicted probabilities.
    
    - If the true label is 0: $−log(1−\hat{y})$
        
        -> Punishes large predicted probabilites

To summarise this, *this function calculates how far your predicted probabilities are from the true labels using binary cross-entropy, while preventing numerical errors with clipping.*

#### NOTE: Loss is computed before back propogating to improve previous errors.

3. ***The Backward pass:***

Backpropogation takes place to update weights so that Neural Netwoks can learn from errors. Without this, Neural Networks will never know how to improve its predictions.

```python
    def _backward(self, X, y):
        # m is number of samples which will be used to compute gradient
        m = X.shape[0]
        
        dZ2 = self.A2 - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
```
*What this function does overall?*
1. Compute output error
2. Propogates it backward
3. Computes gradients for all weights and biases
4. Updates parameters using gradient descent
```
                FORWARD PASS →
X ──→ Z1 ──→ A1 ──→ Z2 ──→ A2 ──→ Loss
        ↑       ↑       ↑
        │       │       │
        │       │       │
        └── dZ1 ← dA1 ← dZ2 ←───────
                        ↑
                    A2 - y
```
Step by step gradient flow:
1. Output error: $dZ_2 = A_2 - y$, this is the starting point of backprop.

2. Output layer gradients error:

$$
dZ_2 = A_2 - y
$$

3. Gradients for output layer:

$$
dW_2 = \frac{1}{m} A_1^T dZ_2
$$

$$
db_2 = \frac{1}{m} \sum dZ_2
$$

4. Backpropagate to hidden layer:

$$
dA_1 = dZ_2 W_2^T
$$

5. Apply ReLU derivative:

$$
dZ_1 = dA_1 \odot \text{ReLU}'(Z_1)
$$

($\odot$ = element wise multiplication)

6. Gradients for hidden layer:

$$
dW_1 = \frac{1}{m} X^T dZ_1
$$

$$
db_1 = \frac{1}{m} \sum dZ_1
$$

7. Parameter Update (Gradient Descent)

$$
W := W - \eta dW
$$

$$
b := b - \eta db
$$

#### NOTE: For Backpropogation, Chain Rule is applied. This allows Neural Networks to learn by efficiently computing how changes in each weight affect final loss.

*Why it matters?* (This requires applying chain rule repeatedly backward through all layers)

- In Neural Networks, output depends on many layers of function. So to update weights, we need gradient of loss function L with respect to each weight $w_i$:

$$dL / dw_i$$

TLDR; 

    Chain rule allows us to propogate errors backwards through network. So each layer reuses derivatives from next layer to efficiently compute its own gradient. This actually makes training computationally feasible.


Voila, *This is the simplest neural network or multi layer perceptron and this lays the foundation for every single AI model that exists today such as chatGPT, self driving cars etc.*