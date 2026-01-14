import { Problem } from '../../types';

export const transformerProblems: Problem[] = [
  {
    id: 'scaled-dot-product-attention',
    title: 'Scaled Dot-Product Attention',
    section: 'transformers',
    difficulty: 'medium',
    description: `
## Scaled Dot-Product Attention

Implement the core attention mechanism from "Attention Is All You Need".

### Formula
\`\`\`
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
\`\`\`

Where:
- Q: Queries (seq_len, d_k)
- K: Keys (seq_len, d_k)
- V: Values (seq_len, d_v)
- d_k: Dimension of keys

### Steps
1. Compute attention scores: Q @ K.T
2. Scale by sqrt(d_k)
3. Apply softmax
4. Multiply by V
    `,
    examples: [
      {
        input: 'Q, K, V all (4, 8) - 4 tokens, 8 dims',
        output: 'output (4, 8)',
        explanation: 'Each token attends to all tokens',
      },
    ],
    starterCode: `import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Queries (seq_len, d_k)
        K: Keys (seq_len, d_k)
        V: Values (seq_len, d_v)

    Returns:
        output: Attention output (seq_len, d_v)
        attention_weights: Attention weights (seq_len, seq_len)
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Output shape correct',
        input: '(np.random.randn(4, 8), np.random.randn(4, 8), np.random.randn(4, 8))',
        expected: '(4, 8)',
        hidden: false,
      },
      {
        id: '2',
        description: 'Attention weights sum to 1',
        input: 'attention_sum_test',
        expected: 'True',
        hidden: false,
      },
    ],
    hints: [
      'Compute scores = Q @ K.T',
      'Scale by sqrt(d_k) where d_k = Q.shape[-1]',
      'Apply softmax along the last axis',
      'Return softmax(scores) @ V',
    ],
    solution: `import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]

    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)

    # Apply softmax
    attention_weights = softmax(scores)

    # Compute output
    output = attention_weights @ V

    return output, attention_weights
`,
  },
  {
    id: 'multi-head-attention',
    title: 'Multi-Head Attention',
    section: 'transformers',
    difficulty: 'hard',
    description: `
## Multi-Head Attention

Implement multi-head attention from the Transformer architecture.

### Concept
\`\`\`
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
\`\`\`

### Steps
1. Project Q, K, V with different learned projections for each head
2. Apply scaled dot-product attention in parallel
3. Concatenate heads
4. Project back to original dimension
    `,
    examples: [
      {
        input: 'X (4, 64), num_heads=8, d_model=64',
        output: 'output (4, 64)',
        explanation: '8 heads with d_k=8 each, concatenated to 64',
      },
    ],
    starterCode: `import numpy as np

def multi_head_attention(Q, K, V, num_heads):
    """
    Multi-head attention mechanism.

    Args:
        Q: Queries (seq_len, d_model)
        K: Keys (seq_len, d_model)
        V: Values (seq_len, d_model)
        num_heads: Number of attention heads

    Returns:
        output: Multi-head attention output (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    assert d_model % num_heads == 0
    d_k = d_model // num_heads

    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Output shape preserved',
        input: '(np.random.randn(4, 64), np.random.randn(4, 64), np.random.randn(4, 64), 8)',
        expected: '(4, 64)',
        hidden: false,
      },
    ],
    hints: [
      'Reshape Q, K, V to (seq_len, num_heads, d_k)',
      'Apply attention for each head',
      'Reshape back to (seq_len, d_model)',
    ],
    solution: `import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def multi_head_attention(Q, K, V, num_heads):
    seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # Reshape to (seq_len, num_heads, d_k)
    Q = Q.reshape(seq_len, num_heads, d_k)
    K = K.reshape(seq_len, num_heads, d_k)
    V = V.reshape(seq_len, num_heads, d_k)

    # Transpose to (num_heads, seq_len, d_k)
    Q = Q.transpose(1, 0, 2)
    K = K.transpose(1, 0, 2)
    V = V.transpose(1, 0, 2)

    # Scaled dot-product attention for each head
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    attention = softmax(scores)
    heads = attention @ V

    # Transpose and reshape back
    heads = heads.transpose(1, 0, 2)  # (seq_len, num_heads, d_k)
    output = heads.reshape(seq_len, d_model)

    return output
`,
  },
  {
    id: 'positional-encoding',
    title: 'Sinusoidal Positional Encoding',
    section: 'transformers',
    difficulty: 'medium',
    description: `
## Sinusoidal Positional Encoding

Implement the positional encoding from the original Transformer paper.

### Formulas
\`\`\`
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
\`\`\`

Where:
- pos: Position in sequence
- i: Dimension index
- d_model: Model dimension

### Purpose
- Injects position information into embeddings
- Allows model to learn relative positions
- Fixed (not learned) encodings
    `,
    examples: [
      {
        input: 'max_len=100, d_model=512',
        output: 'PE matrix (100, 512)',
        explanation: 'Position encoding for 100 positions',
      },
    ],
    starterCode: `import numpy as np

def positional_encoding(max_len, d_model):
    """
    Generate sinusoidal positional encodings.

    Args:
        max_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        PE: Positional encoding matrix (max_len, d_model)
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Correct shape',
        input: '(50, 64)',
        expected: '(50, 64)',
        hidden: false,
      },
      {
        id: '2',
        description: 'First position values',
        input: '(10, 4)',
        expected: 'True',
        hidden: true,
      },
    ],
    hints: [
      'Create position indices: np.arange(max_len)',
      'Create dimension indices for the divisor',
      'Use div_term = 10000^(2i/d_model)',
      'Apply sin to even indices, cos to odd indices',
    ],
    solution: `import numpy as np

def positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))

    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE
`,
  },
  {
    id: 'layer-norm',
    title: 'Layer Normalization',
    section: 'transformers',
    difficulty: 'easy',
    description: `
## Layer Normalization

Implement layer normalization, used in Transformers instead of batch normalization.

### Formula
\`\`\`
LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β
\`\`\`

### Difference from BatchNorm
- **BatchNorm**: Normalizes across batch dimension
- **LayerNorm**: Normalizes across feature dimension
- LayerNorm works better for sequence models
    `,
    examples: [
      {
        input: 'X (batch, seq_len, features)',
        output: 'Normalized X, each position normalized independently',
        explanation: 'Normalize across last dimension',
      },
    ],
    starterCode: `import numpy as np

def layer_norm(X, gamma, beta, eps=1e-5):
    """
    Apply layer normalization.

    Args:
        X: Input (batch, seq_len, features) or (seq_len, features)
        gamma: Scale parameter (features,)
        beta: Shift parameter (features,)
        eps: Small constant

    Returns:
        output: Normalized tensor
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Output mean near zero',
        input: '(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), np.ones(4), np.zeros(4))',
        expected: 'True',
        hidden: false,
      },
      {
        id: '2',
        description: 'Gamma and beta work',
        input: '(np.array([[0, 0, 2, 2]]), np.array([1, 2, 1, 2]), np.zeros(4))',
        expected: '[[-1.0, -2.0, 1.0, 2.0]]',
        hidden: true,
      },
    ],
    hints: [
      'Compute mean and variance along the last axis',
      'Use keepdims=True for proper broadcasting',
      'Apply normalization: (x - mean) / sqrt(var + eps)',
    ],
    solution: `import numpy as np

def layer_norm(X, gamma, beta, eps=1e-5):
    # Compute statistics along last dimension
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)

    # Normalize
    X_norm = (X - mean) / np.sqrt(var + eps)

    # Scale and shift
    return gamma * X_norm + beta
`,
  },
  {
    id: 'causal-mask',
    title: 'Causal Attention Mask',
    section: 'transformers',
    difficulty: 'easy',
    description: `
## Causal (Autoregressive) Mask

Create a causal mask for decoder self-attention to prevent attending to future tokens.

### Mask Structure
\`\`\`
[[0, -inf, -inf, -inf],
 [0,    0, -inf, -inf],
 [0,    0,    0, -inf],
 [0,    0,    0,    0]]
\`\`\`

### Usage
- Add mask to attention scores before softmax
- -inf becomes 0 after softmax
- Ensures each position only attends to previous positions
    `,
    examples: [
      {
        input: 'seq_len=4',
        output: 'Lower triangular mask (4, 4)',
        explanation: 'Position i can only attend to positions <= i',
      },
    ],
    starterCode: `import numpy as np

def create_causal_mask(seq_len):
    """
    Create a causal attention mask.

    Args:
        seq_len: Sequence length

    Returns:
        mask: Causal mask (seq_len, seq_len)
               0 for allowed positions, -inf for masked
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Correct shape and pattern',
        input: '4',
        expected: 'True',
        hidden: false,
      },
      {
        id: '2',
        description: 'Works with attention',
        input: '3',
        expected: '[[0.0, -inf, -inf], [0.0, 0.0, -inf], [0.0, 0.0, 0.0]]',
        hidden: true,
      },
    ],
    hints: [
      'Create a matrix of ones using np.ones((seq_len, seq_len))',
      'Use np.triu to get upper triangular matrix (excluding diagonal)',
      'Replace 1s with -inf',
    ],
    solution: `import numpy as np

def create_causal_mask(seq_len):
    # Create upper triangular matrix (above diagonal)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    # Replace 1s with -inf
    mask = np.where(mask == 1, float('-inf'), 0.0)
    return mask
`,
  },
];
