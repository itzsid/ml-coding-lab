import { Problem } from '../../types';

export const cnnProblems: Problem[] = [
  {
    id: 'conv2d-forward',
    title: '2D Convolution',
    section: 'cnn',
    difficulty: 'hard',
    description: `
## 2D Convolution Operation

Implement the forward pass of a 2D convolution (no padding, stride=1).

### Operation
For each position (i, j) in the output:
\`\`\`
out[i, j] = sum(input[i:i+kH, j:j+kW] * kernel)
\`\`\`

### Output Size
\`\`\`
out_height = input_height - kernel_height + 1
out_width = input_width - kernel_width + 1
\`\`\`

### Function Signature
\`\`\`python
def conv2d(image, kernel):
    # image: (H, W)
    # kernel: (kH, kW)
    # output: (H-kH+1, W-kW+1)
\`\`\`
    `,
    examples: [
      {
        input: 'image 4x4, kernel 3x3',
        output: 'output 2x2',
        explanation: '4-3+1 = 2 in each dimension',
      },
    ],
    starterCode: `import numpy as np

def conv2d(image, kernel):
    """
    Apply 2D convolution to an image.

    Args:
        image: Input image (H, W)
        kernel: Convolution kernel (kH, kW)

    Returns:
        output: Convolved image (H-kH+1, W-kW+1)
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Identity kernel',
        input: '([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]])',
        expected: '[[5]]',
        hidden: false,
      },
      {
        id: '2',
        description: 'Edge detection',
        input: '([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])',
        expected: '[[4, 4], [4, 4]]',
        hidden: false,
      },
    ],
    hints: [
      'Use nested loops to slide the kernel over the image',
      'At each position, compute element-wise product and sum',
      'Output size is (H-kH+1, W-kW+1)',
    ],
    solution: `import numpy as np

def conv2d(image, kernel):
    image = np.array(image)
    kernel = np.array(kernel)

    H, W = image.shape
    kH, kW = kernel.shape

    out_H = H - kH + 1
    out_W = W - kW + 1

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = image[i:i+kH, j:j+kW]
            output[i, j] = np.sum(region * kernel)

    return output.astype(int).tolist()
`,
  },
  {
    id: 'max-pool',
    title: 'Max Pooling',
    section: 'cnn',
    difficulty: 'medium',
    description: `
## Max Pooling

Implement 2x2 max pooling with stride 2.

### Operation
Divide input into non-overlapping 2x2 regions and take maximum of each.

### Output Size
\`\`\`
out_height = input_height // 2
out_width = input_width // 2
\`\`\`

### Purpose
- Reduces spatial dimensions
- Provides translation invariance
- Reduces computation
    `,
    examples: [
      {
        input: '[[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]',
        output: '[[6, 8], [14, 16]]',
        explanation: 'Max of each 2x2 region',
      },
    ],
    starterCode: `import numpy as np

def max_pool2d(image, pool_size=2):
    """
    Apply 2D max pooling.

    Args:
        image: Input image (H, W)
        pool_size: Size of pooling window

    Returns:
        output: Pooled image (H//pool_size, W//pool_size)
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: '4x4 to 2x2',
        input: '[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]',
        expected: '[[6, 8], [14, 16]]',
        hidden: false,
      },
      {
        id: '2',
        description: 'With negative values',
        input: '[[-1, -2, -3, -4], [-5, -6, -7, -8], [-9, -10, -11, -12], [-13, -14, -15, -16]]',
        expected: '[[-1, -3], [-9, -11]]',
        hidden: true,
      },
    ],
    hints: [
      'Iterate with step size = pool_size',
      'For each 2x2 region, use np.max()',
      'Output dimensions are input dimensions // pool_size',
    ],
    solution: `import numpy as np

def max_pool2d(image, pool_size=2):
    image = np.array(image)
    H, W = image.shape

    out_H = H // pool_size
    out_W = W // pool_size

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = image[i*pool_size:(i+1)*pool_size,
                          j*pool_size:(j+1)*pool_size]
            output[i, j] = np.max(region)

    return output.astype(int).tolist()
`,
  },
  {
    id: 'flatten-layer',
    title: 'Flatten Layer',
    section: 'cnn',
    difficulty: 'easy',
    description: `
## Flatten Layer

Implement the flatten operation that converts a 3D feature map to a 1D vector for the fully connected layer.

### Operation
\`\`\`
(batch, height, width, channels) → (batch, height * width * channels)
\`\`\`

### Usage
- Connects convolutional layers to fully connected layers
- Preserves batch dimension
    `,
    examples: [
      {
        input: 'shape (2, 4, 4, 3)',
        output: 'shape (2, 48)',
        explanation: '4 * 4 * 3 = 48 features per sample',
      },
    ],
    starterCode: `import numpy as np

def flatten(X):
    """
    Flatten feature maps to vectors.

    Args:
        X: Input tensor (batch, height, width, channels)

    Returns:
        output: Flattened tensor (batch, height*width*channels)
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Correct output shape',
        input: 'np.random.randn(2, 4, 4, 3)',
        expected: '(2, 48)',
        hidden: false,
      },
      {
        id: '2',
        description: 'Values preserved',
        input: 'np.arange(24).reshape(1, 2, 3, 4)',
        expected: 'True',
        hidden: true,
      },
    ],
    hints: [
      'Get batch size as X.shape[0]',
      'Use reshape to flatten remaining dimensions',
      'np.reshape(X, (batch_size, -1)) uses -1 to infer size',
    ],
    solution: `import numpy as np

def flatten(X):
    batch_size = X.shape[0]
    return X.reshape(batch_size, -1)
`,
  },
  {
    id: 'conv-output-size',
    title: 'Convolution Output Size',
    section: 'cnn',
    difficulty: 'easy',
    description: `
## Calculate Convolution Output Size

Implement a function to calculate the output dimensions of a convolution layer.

### Formula
\`\`\`
output_size = (input_size - kernel_size + 2 * padding) / stride + 1
\`\`\`

### Parameters
- **input_size**: Height or width of input
- **kernel_size**: Height or width of kernel
- **padding**: Zero-padding added to input
- **stride**: Step size of kernel
    `,
    examples: [
      {
        input: 'input=32, kernel=3, padding=1, stride=1',
        output: '32',
        explanation: '(32 - 3 + 2*1) / 1 + 1 = 32',
      },
    ],
    starterCode: `def conv_output_size(input_size, kernel_size, padding=0, stride=1):
    """
    Calculate output size of a convolution layer.

    Args:
        input_size: Input dimension (height or width)
        kernel_size: Kernel dimension
        padding: Zero-padding
        stride: Stride

    Returns:
        output_size: Output dimension
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Same padding',
        input: '(32, 3, 1, 1)',
        expected: '32',
        hidden: false,
      },
      {
        id: '2',
        description: 'No padding, stride 2',
        input: '(32, 3, 0, 2)',
        expected: '15',
        hidden: false,
      },
      {
        id: '3',
        description: 'Large kernel',
        input: '(224, 7, 3, 2)',
        expected: '112',
        hidden: true,
      },
    ],
    hints: [
      'Apply the formula: (input - kernel + 2*padding) / stride + 1',
      'Use integer division (//) for the result',
    ],
    solution: `def conv_output_size(input_size, kernel_size, padding=0, stride=1):
    return (input_size - kernel_size + 2 * padding) // stride + 1
`,
  },
];
