import { Problem } from '../../types';

export const einsumProblems: Problem[] = [
  {
    id: 'einsum-basics',
    title: 'Einsum Fundamentals',
    section: 'einsum',
    difficulty: 'easy',
    description: `
## Einsum Fundamentals

Einstein summation notation provides a powerful way to express array operations.

### Syntax
\`\`\`python
np.einsum('subscripts', operands)
\`\`\`

### Basic Operations
- \`'i->'\`: Sum all elements (reduce)
- \`'ij->i'\`: Sum along columns (row sums)
- \`'ij->j'\`: Sum along rows (column sums)
- \`'ii->i'\`: Extract diagonal
- \`'ij->ji'\`: Transpose

### Task
Implement basic einsum operations on a 2D array.
    `,
    examples: [
      {
        input: 'np.array([[1,2],[3,4]])',
        output: "{'sum': 10, 'row_sum': [3,7], 'transpose': [[1,3],[2,4]]}",
        explanation: 'Basic einsum operations',
      },
    ],
    starterCode: `import numpy as np

def einsum_basics(arr: np.ndarray) -> dict:
    """
    Perform basic einsum operations.

    Args:
        arr: 2D input array

    Returns:
        Dictionary with einsum results
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Sum all elements',
        input: '(np.array([[1, 2], [3, 4]]),)',
        expected: 'sum_check',
        hidden: false,
      },
      {
        id: '2',
        description: 'Transpose',
        input: 'transpose_check',
        expected: 'True',
        hidden: false,
      },
    ],
    hints: [
      "'ij->' sums all elements",
      "'ij->i' sums each row",
      "'ij->ji' transposes",
      "'ii->i' extracts diagonal (for square matrices)",
    ],
    solution: `import numpy as np

def einsum_basics(arr: np.ndarray) -> dict:
    return {
        'sum_all': np.einsum('ij->', arr),
        'row_sum': np.einsum('ij->i', arr),
        'col_sum': np.einsum('ij->j', arr),
        'transpose': np.einsum('ij->ji', arr),
        'diagonal': np.einsum('ii->i', arr) if arr.shape[0] == arr.shape[1] else None
    }
`,
  },
  {
    id: 'einsum-matrix-ops',
    title: 'Matrix Operations with Einsum',
    section: 'einsum',
    difficulty: 'medium',
    description: `
## Matrix Operations with Einsum

Einsum can express matrix multiplication and more complex operations.

### Key Operations
- \`'ik,kj->ij'\`: Matrix multiplication (A @ B)
- \`'ij,ij->ij'\`: Element-wise multiplication (Hadamard)
- \`'ij,ij->'\`: Frobenius inner product (sum of element-wise product)
- \`'ij,j->i'\`: Matrix-vector multiplication
- \`'i,j->ij'\`: Outer product

### Task
Implement matrix operations using einsum.

### Function Signature
\`\`\`python
def einsum_matrix_ops(A: np.ndarray, B: np.ndarray, v: np.ndarray) -> dict:
\`\`\`
    `,
    examples: [
      {
        input: 'A (2,3), B (3,2), v (3,)',
        output: "{'matmul': A@B, 'matvec': A@v, ...}",
        explanation: 'Matrix operations via einsum',
      },
    ],
    starterCode: `import numpy as np

def einsum_matrix_ops(A: np.ndarray, B: np.ndarray, v: np.ndarray) -> dict:
    """
    Perform matrix operations using einsum.

    Args:
        A: Matrix of shape (m, k)
        B: Matrix of shape (k, n)
        v: Vector of shape (k,)

    Returns:
        Dictionary with operation results
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Matrix multiplication',
        input: '(np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]]), np.array([1,2]))',
        expected: 'matmul_check',
        hidden: false,
      },
      {
        id: '2',
        description: 'Outer product',
        input: 'outer_check',
        expected: 'True',
        hidden: false,
      },
    ],
    hints: [
      "'ik,kj->ij' contracts over k (matrix multiply)",
      "'ij,j->i' contracts vector with columns",
      "'i,j->ij' creates outer product (no contraction)",
    ],
    solution: `import numpy as np

def einsum_matrix_ops(A: np.ndarray, B: np.ndarray, v: np.ndarray) -> dict:
    return {
        'matmul': np.einsum('ik,kj->ij', A, B),
        'matvec': np.einsum('ij,j->i', A, v),
        'outer_product': np.einsum('i,j->ij', v, v),
        'hadamard': np.einsum('ij,ij->ij', A, A),  # A * A
        'frobenius': np.einsum('ij,ij->', A, A),   # sum(A * A)
        'trace': np.einsum('ii->', A) if A.shape[0] == A.shape[1] else None
    }
`,
  },
  {
    id: 'einsum-batch-ops',
    title: 'Batch Operations with Einsum',
    section: 'einsum',
    difficulty: 'medium',
    description: `
## Batch Operations with Einsum

Einsum shines for batch operations - multiple matrices at once.

### Batch Matrix Operations
- \`'bij,bjk->bik'\`: Batch matrix multiplication
- \`'bij,bj->bi'\`: Batch matrix-vector multiplication
- \`'bij->bji'\`: Batch transpose

### Attention-style Operations
- \`'bqd,bkd->bqk'\`: Query-Key dot products (attention scores)
- \`'bqk,bkv->bqv'\`: Weighted sum of values

### Task
Implement batch operations common in deep learning.
    `,
    examples: [
      {
        input: 'batch of 4 matrices (4, 3, 3)',
        output: 'Batch-wise operations',
        explanation: 'Operations applied independently to each batch',
      },
    ],
    starterCode: `import numpy as np

def einsum_batch_ops(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> dict:
    """
    Perform batch operations using einsum.

    Args:
        Q: Queries of shape (batch, seq_q, dim)
        K: Keys of shape (batch, seq_k, dim)
        V: Values of shape (batch, seq_k, dim_v)

    Returns:
        Dictionary with batch operation results
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Attention scores shape',
        input: '(np.random.randn(2, 4, 8), np.random.randn(2, 6, 8), np.random.randn(2, 6, 16))',
        expected: 'scores_shape_check',
        hidden: false,
      },
      {
        id: '2',
        description: 'Output shape',
        input: 'output_shape_check',
        expected: 'True',
        hidden: false,
      },
    ],
    hints: [
      "'bqd,bkd->bqk' computes Q @ K.T for each batch",
      "'bqk,bkv->bqv' applies attention weights to values",
      "b is the batch dimension, preserved in output",
    ],
    solution: `import numpy as np

def einsum_batch_ops(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> dict:
    # Q: (batch, seq_q, dim)
    # K: (batch, seq_k, dim)
    # V: (batch, seq_k, dim_v)

    # Attention scores: Q @ K.T for each batch
    # Result: (batch, seq_q, seq_k)
    scores = np.einsum('bqd,bkd->bqk', Q, K)

    # Scale scores
    d_k = Q.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)

    # Softmax (simplified, along last axis)
    exp_scores = np.exp(scaled_scores - scaled_scores.max(axis=-1, keepdims=True))
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Weighted sum of values
    # Result: (batch, seq_q, dim_v)
    output = np.einsum('bqk,bkv->bqv', attention_weights, V)

    return {
        'scores': scores,
        'attention_weights': attention_weights,
        'output': output
    }
`,
  },
  {
    id: 'einsum-advanced',
    title: 'Advanced Einsum Patterns',
    section: 'einsum',
    difficulty: 'hard',
    description: `
## Advanced Einsum Patterns

Complex tensor operations for deep learning.

### Multi-Head Attention Pattern
\`\`\`python
# Split heads: (batch, seq, heads, dim)
# Attention per head: 'bhqd,bhkd->bhqk'
# Combine heads: 'bhqv->bqhv' then reshape
\`\`\`

### Bilinear Operations
\`\`\`python
# x.T @ W @ y: 'i,ijk,k->j'
# Batch bilinear: 'bi,ijk,bk->bj'
\`\`\`

### Task
Implement multi-head attention using einsum.
    `,
    examples: [
      {
        input: 'Q, K, V with multiple heads',
        output: 'Multi-head attention output',
        explanation: 'Parallel attention across heads',
      },
    ],
    starterCode: `import numpy as np

def multi_head_attention_einsum(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 num_heads: int) -> np.ndarray:
    """
    Multi-head attention using einsum.

    Args:
        Q: Queries (batch, seq_q, d_model)
        K: Keys (batch, seq_k, d_model)
        V: Values (batch, seq_k, d_model)
        num_heads: Number of attention heads

    Returns:
        output: (batch, seq_q, d_model)
    """
    batch, seq_q, d_model = Q.shape
    seq_k = K.shape[1]
    d_k = d_model // num_heads

    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Output shape preserved',
        input: '(np.random.randn(2, 4, 64), np.random.randn(2, 4, 64), np.random.randn(2, 4, 64), 8)',
        expected: '(2, 4, 64)',
        hidden: false,
      },
      {
        id: '2',
        description: 'Attention computed correctly',
        input: 'attention_check',
        expected: 'True',
        hidden: true,
      },
    ],
    hints: [
      'Reshape Q, K, V to (batch, seq, num_heads, d_k)',
      "Use 'bqhd,bkhd->bhqk' for attention scores per head",
      "Use 'bhqk,bkhd->bqhd' for weighted values",
      'Reshape back to (batch, seq, d_model)',
    ],
    solution: `import numpy as np

def multi_head_attention_einsum(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 num_heads: int) -> np.ndarray:
    batch, seq_q, d_model = Q.shape
    seq_k = K.shape[1]
    d_k = d_model // num_heads

    # Reshape to (batch, seq, num_heads, d_k)
    Q = Q.reshape(batch, seq_q, num_heads, d_k)
    K = K.reshape(batch, seq_k, num_heads, d_k)
    V = V.reshape(batch, seq_k, num_heads, d_k)

    # Compute attention scores for all heads
    # (batch, seq_q, heads, d_k) x (batch, seq_k, heads, d_k) -> (batch, heads, seq_q, seq_k)
    scores = np.einsum('bqhd,bkhd->bhqk', Q, K) / np.sqrt(d_k)

    # Softmax
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attention = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Weighted sum of values
    # (batch, heads, seq_q, seq_k) x (batch, seq_k, heads, d_k) -> (batch, seq_q, heads, d_k)
    output = np.einsum('bhqk,bkhd->bqhd', attention, V)

    # Reshape back to (batch, seq_q, d_model)
    output = output.reshape(batch, seq_q, d_model)

    return output
`,
  },
  {
    id: 'einsum-vs-matmul',
    title: 'Einsum vs Traditional Operations',
    section: 'einsum',
    difficulty: 'easy',
    description: `
## Einsum vs Traditional Operations

Compare einsum with equivalent NumPy operations.

### Equivalences
| Einsum | NumPy |
|--------|-------|
| \`'ij->'\` | \`np.sum(A)\` |
| \`'ij->i'\` | \`np.sum(A, axis=1)\` |
| \`'ik,kj->ij'\` | \`A @ B\` |
| \`'ij,ij->'\` | \`np.sum(A * B)\` |
| \`'ij->ji'\` | \`A.T\` |

### Task
Verify einsum produces same results as traditional operations.
    `,
    examples: [
      {
        input: 'Two matrices A, B',
        output: 'Comparison showing equivalence',
        explanation: 'All operations produce identical results',
      },
    ],
    starterCode: `import numpy as np

def einsum_equivalence(A: np.ndarray, B: np.ndarray) -> dict:
    """
    Compare einsum with traditional NumPy operations.

    Args:
        A: First matrix (m, k)
        B: Second matrix (k, n)

    Returns:
        Dictionary with 'einsum' and 'numpy' results for comparison
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'All results match',
        input: '(np.random.randn(3, 4), np.random.randn(4, 5))',
        expected: 'True',
        hidden: false,
      },
    ],
    hints: [
      'Use np.allclose() to compare floating point arrays',
      "Remember 'ik,kj->ij' is matrix multiplication",
    ],
    solution: `import numpy as np

def einsum_equivalence(A: np.ndarray, B: np.ndarray) -> dict:
    results = {
        'sum': {
            'einsum': np.einsum('ij->', A),
            'numpy': np.sum(A)
        },
        'row_sum': {
            'einsum': np.einsum('ij->i', A),
            'numpy': np.sum(A, axis=1)
        },
        'matmul': {
            'einsum': np.einsum('ik,kj->ij', A, B),
            'numpy': A @ B
        },
        'transpose': {
            'einsum': np.einsum('ij->ji', A),
            'numpy': A.T
        },
        'hadamard_sum': {
            'einsum': np.einsum('ij,ij->', A, A),
            'numpy': np.sum(A * A)
        }
    }

    # Verify all match
    all_match = all(
        np.allclose(v['einsum'], v['numpy'])
        for v in results.values()
    )

    return {'results': results, 'all_match': all_match}
`,
  },
];
