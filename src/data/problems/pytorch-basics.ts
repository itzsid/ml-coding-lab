import { Problem } from '../../types';

export const pytorchBasicsProblems: Problem[] = [
  {
    id: 'tensor-creation',
    title: 'Tensor Creation (NumPy Style)',
    section: 'pytorch-basics',
    difficulty: 'easy',
    description: `
## Tensor Creation

Learn tensor creation patterns used in PyTorch, implemented with NumPy.

### PyTorch Equivalents
| PyTorch | NumPy |
|---------|-------|
| \`torch.tensor([1,2,3])\` | \`np.array([1,2,3])\` |
| \`torch.zeros(3, 4)\` | \`np.zeros((3, 4))\` |
| \`torch.randn(2, 3)\` | \`np.random.randn(2, 3)\` |
| \`torch.arange(10)\` | \`np.arange(10)\` |
| \`torch.linspace(0, 1, 5)\` | \`np.linspace(0, 1, 5)\` |

### Data Types
| PyTorch | NumPy |
|---------|-------|
| \`torch.float32\` | \`np.float32\` |
| \`torch.int64\` | \`np.int64\` |

### Task
Create tensors matching PyTorch patterns.

### Expected Return Format
Return a dictionary with these keys:
- \`'from_list'\`: Array from [1,2,3,4] with float32 dtype
- \`'zeros'\`: Zero array of shape (3, 4)
- \`'ones'\`: Ones array of shape (2, 3)
- \`'randn'\`: Random normal array of shape (2, 3)
- \`'arange'\`: Array from 0 to 9
- \`'linspace'\`: 5 evenly spaced values from 0 to 1
- \`'eye'\`: 4x4 identity matrix
- \`'full'\`: (2, 3) array filled with 7.0
    `,
    examples: [
      {
        input: 'create_tensors()',
        output: "{'zeros': shape (3,4), 'ones': shape (2,3), 'randn': shape (2,3), 'arange': [0..9], ...}",
        explanation: 'Dictionary containing tensors created with various methods',
      },
    ],
    starterCode: `import numpy as np

def create_tensors() -> dict:
    """
    Create tensors using various methods.

    Returns:
        Dictionary with named tensors
    """
    np.random.seed(42)

    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Zeros shape',
        input: 'create_tensors()["zeros"].shape',
        expected: '(3, 4)',
        hidden: false,
      },
      {
        id: '2',
        description: 'Random tensor shape',
        input: 'create_tensors()["randn"].shape',
        expected: '(2, 3)',
        hidden: false,
      },
      {
        id: '3',
        description: 'Values of eye, arange, linspace, full',
        input: '(lambda r: (r["eye"].tolist(), r["arange"].tolist(), r["linspace"].tolist(), r["full"].tolist()))(create_tensors())',
        expected: '([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 0.25, 0.5, 0.75, 1.0], [[7.0, 7.0, 7.0], [7.0, 7.0, 7.0]])',
        hidden: true,
      },
    ],
    hints: [
      'np.zeros((3, 4)) for 3x4 zeros',
      'np.random.randn(2, 3) for random normal',
      'Specify dtype with astype() or dtype parameter',
    ],
    solution: `import numpy as np

def create_tensors() -> dict:
    np.random.seed(42)

    return {
        'from_list': np.array([1, 2, 3, 4], dtype=np.float32),
        'zeros': np.zeros((3, 4)),
        'ones': np.ones((2, 3)),
        'randn': np.random.randn(2, 3),
        'arange': np.arange(10),
        'linspace': np.linspace(0, 1, 5),
        'eye': np.eye(4),
        'full': np.full((2, 3), 7.0),
    }
`,
  },
  {
    id: 'tensor-operations',
    title: 'Tensor Operations',
    section: 'pytorch-basics',
    difficulty: 'easy',
    description: `
## Basic Tensor Operations

Common tensor operations used in neural networks.

### Arithmetic Operations
- Element-wise: +, -, *, /
- Matrix multiplication: @ or np.matmul
- Power: ** or np.power

### Reduction Operations
- sum, mean, std, var
- min, max, argmin, argmax

### Shape Operations
- reshape, view (reshape in NumPy)
- squeeze, unsqueeze (np.squeeze, np.expand_dims)
- permute, transpose

### Task
Implement common tensor operations.

### Expected Return Format
Return a dictionary with these keys:
- **Arithmetic**: \`'add'\`, \`'sub'\`, \`'mul'\`, \`'div'\`, \`'pow'\`
- **Reductions**: \`'sum_all'\`, \`'sum_axis0'\`, \`'sum_axis1'\`, \`'mean'\`, \`'std'\`, \`'max'\`, \`'argmax'\`
- **Shape ops**: \`'reshape'\` (to 3,2), \`'flatten'\`, \`'unsqueeze'\` (add dim 0), \`'squeeze'\`, \`'transpose'\`
    `,
    examples: [
      {
        input: 'x=[[1,2,3],[4,5,6]], y=[[1,1,1],[2,2,2]]',
        output: "{'add': [[2,3,4],[6,7,8]], 'mean': 3.5, 'reshape': [[1,2],[3,4],[5,6]], ...}",
        explanation: 'Arithmetic, reduction, and shape operations on tensors',
      },
    ],
    starterCode: `import numpy as np

def tensor_ops(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Perform common tensor operations.

    Args:
        x: First tensor (2, 3)
        y: Second tensor (2, 3)

    Returns:
        Dictionary with operation results
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Addition',
        input: 'bool(np.array_equal(tensor_ops(np.array([[1,2,3],[4,5,6]]), np.array([[1,1,1],[2,2,2]]))["add"], np.array([[2,3,4],[6,7,8]])))',
        expected: 'True',
        hidden: false,
      },
      {
        id: '2',
        description: 'Mean computation',
        input: 'tensor_ops(np.array([[1,2,3],[4,5,6]]), np.array([[1,1,1],[2,2,2]]))["mean"]',
        expected: '3.5',
        hidden: false,
      },
      {
        id: '3',
        description: 'Reductions, argmax and reshapes',
        input: '(lambda r: (r["sum_axis0"].tolist(), r["sum_axis1"].tolist(), int(r["argmax"]), r["transpose"].shape, r["unsqueeze"].shape))(tensor_ops(np.array([[1, 5, 3], [4, 2, 6]]), np.array([[1, 1, 1], [2, 2, 2]])))',
        expected: '([5, 7, 9], [9, 12], 5, (3, 2), (1, 2, 3))',
        hidden: true,
      },
    ],
    hints: [
      'Element-wise ops work directly: x + y, x * y',
      'Use axis parameter for reductions along specific dims',
      'np.expand_dims(x, axis=0) adds dimension',
    ],
    solution: `import numpy as np

def tensor_ops(x: np.ndarray, y: np.ndarray) -> dict:
    return {
        # Arithmetic
        'add': x + y,
        'sub': x - y,
        'mul': x * y,
        'div': x / (y + 1e-8),
        'pow': x ** 2,

        # Reductions
        'sum_all': x.sum(),
        'sum_axis0': x.sum(axis=0),
        'sum_axis1': x.sum(axis=1),
        'mean': x.mean(),
        'std': x.std(),
        'max': x.max(),
        'argmax': x.argmax(),

        # Shape ops
        'reshape': x.reshape(3, 2),
        'flatten': x.flatten(),
        'unsqueeze': np.expand_dims(x, axis=0),  # (1, 2, 3)
        'squeeze': np.squeeze(np.expand_dims(x, 0)),
        'transpose': x.T,
    }
`,
  },
  {
    id: 'autograd-concepts',
    title: 'Autograd Concepts (Manual)',
    section: 'pytorch-basics',
    difficulty: 'medium',
    description: `
## Autograd Concepts

Understand automatic differentiation by implementing it manually.

### PyTorch Autograd
\`\`\`python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7
\`\`\`

### Manual Gradient Computation
For y = x² + 3x + 1:
- dy/dx = 2x + 3

### Task
Implement two functions:
1. \`compute_gradients(x)\`: forward pass and derivative of y = x² + 3x + 1. Return \`{'y': ..., 'grad': ...}\`
2. \`linear_layer_gradients(X, W, b, grad_output)\`: backward pass of Y = X @ W + b. Return \`{'grad_W': X.T @ grad_output, 'grad_b': grad_output summed over the batch, 'grad_X': grad_output @ W.T}\`
    `,
    examples: [
      {
        input: 'compute_gradients(x=2.0)',
        output: "{'y': 11.0, 'grad': 7.0}",
        explanation: 'Forward and backward pass',
      },
    ],
    starterCode: `import numpy as np

def compute_gradients(x: float) -> dict:
    """
    Compute forward and backward pass for y = x^2 + 3x + 1.

    Args:
        x: Input value

    Returns:
        Dictionary with 'y' (forward result) and 'grad' (dy/dx)
    """
    # Your code here
    pass


def linear_layer_gradients(X: np.ndarray, W: np.ndarray, b: np.ndarray,
                           grad_output: np.ndarray) -> dict:
    """
    Compute gradients for a linear layer: Y = X @ W + b

    Args:
        X: Input (batch, in_features)
        W: Weights (in_features, out_features)
        b: Bias (out_features,)
        grad_output: Gradient from next layer (batch, out_features)

    Returns:
        Dictionary with gradients for X, W, and b
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Simple gradient',
        input: 'compute_gradients(2.0)["grad"]',
        expected: '7.0',
        hidden: false,
      },
      {
        id: '2',
        description: 'Forward pass result',
        input: 'compute_gradients(2.0)["y"]',
        expected: '11.0',
        hidden: false,
      },
      {
        id: '3',
        description: 'Linear layer gradients match finite differences',
        input: '(lambda X, W, b, G: (lambda r: bool(np.allclose(r["grad_W"], X.T @ G) and np.allclose(r["grad_b"], G.sum(0)) and np.allclose(r["grad_X"], G @ W.T)))(linear_layer_gradients(X, W, b, G)))(np.random.randn(4, 3), np.random.randn(3, 2), np.random.randn(2), np.random.randn(4, 2))',
        expected: 'True',
        hidden: true,
      },
    ],
    hints: [
      'For y = x² + 3x + 1, dy/dx = 2x + 3',
      'For Y = X @ W + b: dL/dW = X.T @ grad_output',
      'dL/dX = grad_output @ W.T',
      'dL/db = sum(grad_output, axis=0)',
    ],
    solution: `import numpy as np

def compute_gradients(x: float) -> dict:
    # Forward pass
    y = x ** 2 + 3 * x + 1

    # Backward pass (analytical gradient)
    grad = 2 * x + 3

    return {'y': y, 'grad': grad}


def linear_layer_gradients(X: np.ndarray, W: np.ndarray, b: np.ndarray,
                           grad_output: np.ndarray) -> dict:
    # Forward: Y = X @ W + b

    # Backward:
    # dL/dW = X.T @ grad_output
    grad_W = X.T @ grad_output

    # dL/db = sum of grad_output over batch
    grad_b = grad_output.sum(axis=0)

    # dL/dX = grad_output @ W.T
    grad_X = grad_output @ W.T

    return {
        'grad_W': grad_W,
        'grad_b': grad_b,
        'grad_X': grad_X
    }
`,
  },
  {
    id: 'nn-modules',
    title: 'Neural Network Modules',
    section: 'pytorch-basics',
    difficulty: 'medium',
    description: `
## Neural Network Modules

Implement PyTorch-style nn.Module patterns in NumPy.

### Module Pattern
\`\`\`python
class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]
\`\`\`

### Task
Implement Linear and ReLU modules with forward method.
    `,
    examples: [
      {
        input: 'Linear(10, 5).forward(x)',
        output: 'Output of shape (batch, 5)',
        explanation: 'Linear transformation',
      },
    ],
    starterCode: `import numpy as np

class Linear:
    """Linear layer: Y = X @ W + b"""

    def __init__(self, in_features: int, out_features: int):
        # Initialize weights with small random values
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Your code here
        pass

    def parameters(self):
        # Return list of parameters
        pass


class ReLU:
    """ReLU activation: max(0, x)"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Your code here
        pass


class Sequential:
    """Container for sequential layers"""

    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Your code here
        pass


def create_mlp(input_dim: int, hidden_dim: int, output_dim: int):
    """Create a 2-layer MLP: Linear -> ReLU -> Linear"""
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Linear output shape',
        input: 'Linear(10, 5).forward(np.random.randn(4, 10)).shape',
        expected: '(4, 5)',
        hidden: false,
      },
      {
        id: '2',
        description: 'Sequential forward',
        input: 'create_mlp(10, 20, 5).forward(np.random.randn(4, 10)).shape',
        expected: '(4, 5)',
        hidden: false,
      },
      {
        id: '3',
        description: 'Linear computes x @ W + b with its own parameters',
        input: '(lambda l, x: bool(np.allclose(l.forward(x), x @ l.parameters()[0] + l.parameters()[1]) or np.allclose(l.forward(x), x @ l.parameters()[0].T + l.parameters()[1])))(Linear(3, 2), np.random.randn(5, 3))',
        expected: 'True',
        hidden: true,
      },
    ],
    hints: [
      'Initialize W with np.random.randn() * 0.01',
      'ReLU: np.maximum(0, x)',
      'Sequential applies layers in order',
    ],
    solution: `import numpy as np

class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x


def create_mlp(input_dim: int, hidden_dim: int, output_dim: int):
    return Sequential(
        Linear(input_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, output_dim)
    )
`,
  },
  {
    id: 'loss-functions',
    title: 'Loss Functions',
    section: 'pytorch-basics',
    difficulty: 'medium',
    description: `
## Common Loss Functions

Implement loss functions used in deep learning.

### Cross-Entropy Loss
\`\`\`
L = -sum(y * log(softmax(logits)))
\`\`\`

### MSE Loss
\`\`\`
L = mean((y_pred - y_true)²)
\`\`\`

### Binary Cross-Entropy
\`\`\`
L = -mean(y * log(p) + (1-y) * log(1-p))
\`\`\`

### Task
Implement these loss functions with numerical stability.
    `,
    examples: [
      {
        input: 'cross_entropy(logits, targets)',
        output: 'Scalar loss value',
        explanation: 'Cross-entropy for classification',
      },
    ],
    starterCode: `import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    # Your code here
    pass


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Cross-entropy loss for classification.

    Args:
        logits: Raw scores (batch, num_classes)
        targets: Integer class labels (batch,)

    Returns:
        Scalar loss
    """
    # Your code here
    pass


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error loss."""
    # Your code here
    pass


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Binary cross-entropy with numerical stability."""
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Cross-entropy computation',
        input: 'round(cross_entropy_loss(np.array([[2.0, 1.0, 0.1]]), np.array([0])), 4)',
        expected: '0.417',
        hidden: false,
      },
      {
        id: '2',
        description: 'MSE computation',
        input: 'mse_loss(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))',
        expected: '0.333333',
        hidden: false,
      },
      {
        id: '3',
        description: 'Cross-entropy on a batch and BCE',
        input: '(lambda: (round(cross_entropy_loss(np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]), np.array([2, 0])), 4), round(binary_cross_entropy(np.array([0.9, 0.2]), np.array([1.0, 0.0])), 4)))()',
        expected: '(0.7531, 0.1643)',
        hidden: true,
      },
    ],
    hints: [
      'Softmax: exp(x - max(x)) / sum(exp(x - max(x)))',
      'Use np.clip for numerical stability in log',
      'Index logits with targets for cross-entropy',
    ],
    solution: `import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    batch_size = logits.shape[0]
    probs = softmax(logits)

    # Clip for numerical stability
    probs = np.clip(probs, 1e-15, 1 - 1e-15)

    # Select probability of correct class for each sample
    correct_probs = probs[np.arange(batch_size), targets]

    # Negative log likelihood
    loss = -np.mean(np.log(correct_probs))
    return loss


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean((y_pred - y_true) ** 2)


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
`,
  },
  {
    id: 'training-loop',
    title: 'The Training Loop',
    section: 'pytorch-basics',
    difficulty: 'easy',
    description: `
## The Training Loop

Every PyTorch model, from linear regression to an LLM, is trained by the same five-line loop:

\`\`\`python
for epoch in range(epochs):
    optimizer.zero_grad()          # 1. clear the gradients from the last step
    pred = model(X)                # 2. forward pass
    loss = loss_fn(pred, y)        # 3. compute the loss
    loss.backward()                # 4. backprop: fill p.grad for every parameter
    optimizer.step()               # 5. update the parameters
\`\`\`

Forgetting \`zero_grad()\` is the classic bug: \`.grad\` **accumulates**, so every step would use the sum of all past gradients.

### Task
Fit a linear model \`y_hat = X @ w + b\` with **full-batch gradient descent** on the mean squared error, and record the loss at every epoch.

\`\`\`
L     = mean((X @ w + b - y) ** 2)
dL/dw = (2 / N) * X.T @ (X @ w + b - y)
dL/db = (2 / N) * sum(X @ w + b - y)
w     = w - lr * dL/dw
b     = b - lr * dL/db
\`\`\`

- Start from **zeros**: \`w = 0\` (shape \`(d,)\`), \`b = 0\`.
- Each epoch, compute the loss with the **current** parameters, append it to the history, then take one step. So \`losses[0]\` is the loss of the all-zero model, \`mean(y ** 2)\`.
- In the NumPy track you write the gradients by hand (what \`loss.backward()\` does for you in PyTorch).

### Function Signature
\`\`\`python
def train_linear_model(X, y, lr=0.1, epochs=100) -> dict:
\`\`\`

### Expected Return Format
Return a dictionary with these keys:
- \`'losses'\`: list of \`epochs\` Python floats, the loss **before** each update
- \`'w'\`: learned weights, shape \`(d,)\`
- \`'b'\`: learned bias, a Python float
    `,
    examples: [
      {
        input: 'np.random.seed(0); X = np.random.randn(64, 2); y = X @ [2, -3] + 0.5; lr=0.1, epochs=200',
        output: "{'losses': [14.942, 8.856, ..., ~0.0], 'w': array([2., -3.]), 'b': 0.5}",
        explanation: 'The loss decreases every epoch and the parameters converge to the true ones',
      },
    ],
    starterCode: `import numpy as np

def train_linear_model(X, y, lr=0.1, epochs=100) -> dict:
    """
    Fit y ~ X @ w + b with full-batch gradient descent on the MSE.

    Args:
        X: Inputs of shape (N, d)
        y: Targets of shape (N,)
        lr: Learning rate
        epochs: Number of gradient steps

    Returns:
        dict with 'losses' (list of floats, loss before each step),
        'w' (array of shape (d,)) and 'b' (float)
    """
    w = np.zeros(X.shape[1])
    b = 0.0
    losses = []
    # Your code here: forward, loss, gradients, update
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'History length, first loss, strictly decreasing',
        input: "(lambda: (X := np.random.randn(32, 2), y := X @ np.array([2.0, -3.0]) + 0.5, r := train_linear_model(X, y, lr=0.1, epochs=50), len(r['losses']), bool(np.isclose(r['losses'][0], np.mean(y ** 2))), all(a > b for a, b in zip(r['losses'], r['losses'][1:])))[-3:])()",
        expected: '(50, True, True)',
        hidden: false,
      },
      {
        id: '2',
        description: 'Converges to the true parameters',
        input: "(lambda: (X := np.random.randn(64, 2), y := X @ np.array([2.0, -3.0]) + 0.5, r := train_linear_model(X, y, lr=0.1, epochs=200), np.round(r['w'], 3).tolist(), round(float(r['b']), 3), bool(r['losses'][-1] < 1e-6))[-3:])()",
        expected: '([2.0, -3.0], 0.5, True)',
        hidden: false,
      },
      {
        id: '3',
        description: 'Exact loss history and parameters on a tiny dataset',
        input: "(lambda: (r := train_linear_model(np.array([[1.0], [2.0], [3.0]]), np.array([2.0, 4.0, 6.5]), lr=0.05, epochs=4), [round(l, 4) for l in r['losses']], np.round(r['w'], 4).tolist(), round(float(r['b']), 4))[1:])()",
        expected: '([20.75, 4.2638, 0.9907, 0.3383], [1.716], 0.6924)',
        hidden: true,
      },
      {
        id: '4',
        description: 'Noisy data: converges to the least-squares solution',
        input: `(lambda: (
    X := np.random.randn(100, 3), y := X @ np.array([1.0, -2.0, 0.5]) - 1.0 + 0.3 * np.random.randn(100),
    r := train_linear_model(X, y, lr=0.1, epochs=300),
    A := np.hstack([X, np.ones((100, 1))]),
    sol := np.linalg.lstsq(A, y, rcond=None)[0],
    bool(np.allclose(r['w'], sol[:3], atol=1e-4)),
    bool(abs(r['b'] - sol[3]) < 1e-4),
    bool(abs(r['losses'][-1] - np.mean((A @ sol - y) ** 2)) < 1e-6)
)[-3:])()`,
        expected: '(True, True, True)',
        hidden: true,
      },
    ],
    hints: [
      'Inside the loop: pred = X @ w + b, residual = pred - y, loss = np.mean(residual ** 2).',
      'Append float(loss) to the history before updating the parameters.',
      'Gradients: grad_w = 2 / N * X.T @ residual and grad_b = 2 / N * residual.sum(); then w -= lr * grad_w, b -= lr * grad_b.',
    ],
    solution: `import numpy as np

def train_linear_model(X, y, lr=0.1, epochs=100) -> dict:
    """
    Fit y ~ X @ w + b with full-batch gradient descent on the MSE.
    """
    n = X.shape[0]
    w = np.zeros(X.shape[1])
    b = 0.0
    losses = []
    for _ in range(epochs):
        # forward + loss
        residual = X @ w + b - y
        losses.append(float(np.mean(residual ** 2)))
        # backward (by hand)
        grad_w = 2 / n * X.T @ residual
        grad_b = 2 / n * residual.sum()
        # step
        w = w - lr * grad_w
        b = b - lr * grad_b
    return {'losses': losses, 'w': w, 'b': float(b)}
`,
  },
  {
    id: 'dataloader-batching',
    title: 'Minibatches: Dataset & DataLoader',
    section: 'pytorch-basics',
    difficulty: 'easy',
    description: `
## Minibatching with Shuffling

Training on the whole dataset at every step is slow, and training on it in a fixed order biases each epoch. A **data loader** shuffles the sample indices once per epoch and cuts them into minibatches. In PyTorch this is

\`\`\`python
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True, drop_last=False,
                    generator=torch.Generator().manual_seed(0))
for xb, yb in loader:
    ...
\`\`\`

### Task
Implement the iterator yourself and return all minibatches of one epoch as a list:

1. Build an index order: \`np.arange(N)\` if \`shuffle=False\`, otherwise **exactly** \`np.random.default_rng(seed).permutation(N)\`.
2. Cut the order into consecutive chunks of \`batch_size\`. The last chunk can be smaller.
3. If \`drop_last=True\`, drop the last chunk when it has fewer than \`batch_size\` samples.
4. For each chunk \`idx\`, emit the pair \`(X[idx], y[idx])\`, so rows of \`X\` stay paired with their labels.

Shuffling with a seeded generator keeps the run reproducible: the same seed gives the same batches.

| N | batch_size | drop_last | number of batches | last batch size |
|---|---|---|---|---|
| 10 | 4 | False | 3 | 2 |
| 10 | 4 | True | 2 | 4 |
| 12 | 4 | either | 3 | 4 |

### Function Signature
\`\`\`python
def make_batches(X, y, batch_size, shuffle=True, drop_last=False, seed=0) -> list:
\`\`\`

### Return Format
A **list** of tuples \`(X_batch, y_batch)\` with shapes \`(B, ...)\` and \`(B,)\`, where \`B = batch_size\` except possibly for the last batch.
    `,
    examples: [
      {
        input: 'X = np.arange(20).reshape(10, 2), y = np.arange(10), batch_size=4, shuffle=False',
        output: '[(X[0:4], [0, 1, 2, 3]), (X[4:8], [4, 5, 6, 7]), (X[8:10], [8, 9])]',
        explanation: 'Three batches; the last one holds the 2 leftover samples',
      },
      {
        input: 'same, drop_last=True',
        output: '[(X[0:4], [0, 1, 2, 3]), (X[4:8], [4, 5, 6, 7])]',
        explanation: 'The incomplete last batch is dropped',
      },
    ],
    starterCode: `import numpy as np

def make_batches(X, y, batch_size, shuffle=True, drop_last=False, seed=0) -> list:
    """
    Split (X, y) into minibatches for one epoch.

    Args:
        X: Inputs of shape (N, ...)
        y: Labels of shape (N,)
        batch_size: Samples per batch
        shuffle: Shuffle with np.random.default_rng(seed).permutation(N)
        drop_last: Drop the final batch if it is smaller than batch_size
        seed: Seed for the shuffle

    Returns:
        List of (X_batch, y_batch) tuples
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'No shuffle: consecutive batches, smaller last batch',
        input: '(lambda: [yb.tolist() for _, yb in make_batches(np.arange(20).reshape(10, 2), np.arange(10), 4, shuffle=False)])()',
        expected: '[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]',
        hidden: false,
      },
      {
        id: '2',
        description: 'drop_last=True: batch shapes',
        input: '(lambda: [xb.shape for xb, _ in make_batches(np.zeros((10, 3)), np.arange(10), 4, shuffle=True, drop_last=True)])()',
        expected: '[[4, 3], [4, 3]]',
        hidden: false,
      },
      {
        id: '3',
        description: 'Shuffled order is exactly default_rng(seed).permutation(N)',
        input: '(lambda: (bs := make_batches(np.arange(22).reshape(11, 2), np.arange(11), 3, seed=7), np.concatenate([yb for _, yb in bs]).tolist() == np.random.default_rng(7).permutation(11).tolist(), [len(yb) for _, yb in bs])[1:])()',
        expected: '(True, [3, 3, 3, 2])',
        hidden: true,
      },
      {
        id: '4',
        description: 'Each sample exactly once, rows stay paired with labels',
        input: '(lambda: (X := np.arange(26).reshape(13, 2), bs := make_batches(X, np.arange(13), 5, seed=3), sorted(np.concatenate([yb for _, yb in bs]).tolist()) == list(range(13)), all(bool((xb[:, 0] == 2 * yb).all()) for xb, yb in bs))[-2:])()',
        expected: '(True, True)',
        hidden: true,
      },
      {
        id: '5',
        description: 'Same seed gives the same batches, a different seed does not',
        input: '(lambda: (X := np.random.randn(16, 2), y := np.arange(16), a := [yb.tolist() for _, yb in make_batches(X, y, 4, seed=1)], b := [yb.tolist() for _, yb in make_batches(X, y, 4, seed=1)], c := [yb.tolist() for _, yb in make_batches(X, y, 4, seed=2)], a == b, a != c)[-2:])()',
        expected: '(True, True)',
        hidden: true,
      },
    ],
    hints: [
      'order = np.random.default_rng(seed).permutation(N) if shuffle else np.arange(N).',
      'Loop start = 0, batch_size, 2*batch_size, ... and take idx = order[start:start + batch_size].',
      'With drop_last, skip idx when len(idx) < batch_size. Index both arrays with the same idx: (X[idx], y[idx]).',
    ],
    solution: `import numpy as np

def make_batches(X, y, batch_size, shuffle=True, drop_last=False, seed=0) -> list:
    """
    Split (X, y) into minibatches for one epoch.
    """
    n = len(X)
    order = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    batches = []
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        if drop_last and len(idx) < batch_size:
            break
        batches.append((X[idx], y[idx]))
    return batches
`,
  },
];
