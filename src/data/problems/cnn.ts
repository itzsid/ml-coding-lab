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
        expected: '[[5, 5], [5, 5]]',
        hidden: false,
      },
      {
        id: '3',
        description: 'Asymmetric kernel on a 4x4 image',
        input: '([[1, 2, 3, 0], [4, 1, 0, 2], [0, 5, 1, 3], [2, 0, 6, 1]], [[1, 0], [2, -1]])',
        expected: '[[8, 4, 1], [-1, 10, -1], [4, -1, 12]]',
        hidden: true,
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
        input: 'flatten(np.random.randn(2, 4, 4, 3)).shape',
        expected: '(2, 48)',
        hidden: false,
      },
      {
        id: '2',
        description: 'Values preserved',
        input: 'bool(np.allclose(flatten(np.arange(24).reshape(1, 2, 3, 4)), np.arange(24).reshape(1, 24)))',
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
  {
    id: 'conv2d-advanced',
    title: 'Advanced 2D Convolution',
    section: 'cnn',
    difficulty: 'hard',
    description: `
## Advanced 2D Convolution

Implement a full-featured 2D convolution with **padding**, **stride**, and **groups** support.

### Parameters
- **padding**: Zero-padding added to input borders
- **stride**: Step size when sliding the kernel
- **groups**: Split input/output channels into groups (used in depthwise separable convolutions)

### Group Convolution
When \`groups > 1\`:
- Input channels are split into \`groups\` chunks
- Output channels are split into \`groups\` chunks
- Each group's input is convolved only with its corresponding kernel group
- \`in_channels\` and \`out_channels\` must be divisible by \`groups\`

### Special Cases
- \`groups=1\`: Standard convolution
- \`groups=in_channels=out_channels\`: Depthwise convolution (MobileNet)

### Output Size Formula
\`\`\`
H_out = (H_in + 2*padding - kernel_height) // stride + 1
W_out = (W_in + 2*padding - kernel_width) // stride + 1
\`\`\`

### Function Signature
\`\`\`python
def conv2d_advanced(input, kernel, padding=0, stride=1, groups=1):
    # input: (batch, in_channels, H, W)
    # kernel: (out_channels, in_channels//groups, kH, kW)
    # output: (batch, out_channels, H_out, W_out)
\`\`\`
    `,
    examples: [
      {
        input: 'input (1, 4, 5, 5), kernel (8, 2, 3, 3), groups=2',
        output: 'output (1, 8, 3, 3)',
        explanation: 'Groups=2: channels 0-1 use kernels 0-3, channels 2-3 use kernels 4-7',
      },
      {
        input: 'input (1, 1, 5, 5), kernel (1, 1, 3, 3), padding=1, stride=2',
        output: 'output (1, 1, 3, 3)',
        explanation: '(5+2*1-3)//2+1 = 3',
      },
    ],
    starterCode: `import numpy as np

def conv2d_advanced(input, kernel, padding=0, stride=1, groups=1):
    """
    Advanced 2D convolution with padding, stride, and groups.

    Args:
        input: Input tensor (batch, in_channels, H, W)
        kernel: Convolution kernels (out_channels, in_channels//groups, kH, kW)
        padding: Zero-padding added to input (default: 0)
        stride: Stride of the convolution (default: 1)
        groups: Number of groups for grouped convolution (default: 1)

    Returns:
        output: Convolved tensor (batch, out_channels, H_out, W_out)
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Basic convolution with padding and stride',
        input: 'conv2d_advanced(np.ones((1, 1, 5, 5)), np.ones((1, 1, 3, 3)), 1, 2, 1).shape',
        expected: '(1, 1, 3, 3)',
        hidden: false,
      },
      {
        id: '2',
        description: 'Grouped convolution (groups=2)',
        input: 'conv2d_advanced(np.arange(32).reshape(1, 2, 4, 4).astype(float), np.ones((4, 1, 2, 2)), 0, 1, 2).shape',
        expected: '(1, 4, 3, 3)',
        hidden: false,
      },
      {
        id: '3',
        description: 'Depthwise convolution (groups=in_channels)',
        input: 'conv2d_advanced(np.ones((2, 3, 4, 4)), np.ones((3, 1, 2, 2)), 0, 1, 3).shape',
        expected: '(2, 3, 3, 3)',
        hidden: true,
      },
      {
        id: '4',
        description: 'Verify correct output values with groups',
        input: 'bool(np.allclose(conv2d_advanced(np.ones((1, 2, 3, 3)), np.ones((2, 1, 2, 2)), 0, 1, 2), np.array([[[[4., 4.], [4., 4.]], [[4., 4.], [4., 4.]]]])) )',
        expected: 'True',
        hidden: true,
      },
    ],
    hints: [
      'First, pad the input using np.pad() along the H and W dimensions',
      'Calculate output dimensions: H_out = (H_padded - kH) // stride + 1',
      'For groups: split input channels into chunks of size in_channels//groups',
      'For groups: split kernels into chunks of size out_channels//groups',
      'Convolve each input group with its corresponding kernel group',
      'Concatenate results along the channel dimension',
    ],
    solution: `import numpy as np

def conv2d_advanced(input, kernel, padding=0, stride=1, groups=1):
    input = np.array(input, dtype=float)
    kernel = np.array(kernel, dtype=float)

    batch, in_channels, H, W = input.shape
    out_channels, kernel_in_channels, kH, kW = kernel.shape

    # Pad input
    if padding > 0:
        input = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    _, _, H_padded, W_padded = input.shape

    # Calculate output dimensions
    H_out = (H_padded - kH) // stride + 1
    W_out = (W_padded - kW) // stride + 1

    # Initialize output
    output = np.zeros((batch, out_channels, H_out, W_out))

    # Channels per group
    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups

    for g in range(groups):
        # Input channels for this group
        in_start = g * in_channels_per_group
        in_end = in_start + in_channels_per_group

        # Output channels for this group
        out_start = g * out_channels_per_group
        out_end = out_start + out_channels_per_group

        # Get the input slice for this group
        input_group = input[:, in_start:in_end, :, :]

        # Get the kernels for this group
        kernel_group = kernel[out_start:out_end, :, :, :]

        # Perform convolution for this group
        for b in range(batch):
            for oc in range(out_channels_per_group):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        region = input_group[b, :, h_start:h_start+kH, w_start:w_start+kW]
                        output[b, out_start + oc, i, j] = np.sum(region * kernel_group[oc])

    return output
`,
  },
  {
    id: 'resnet-block',
    title: 'ResNet Basic Block',
    section: 'cnn',
    difficulty: 'medium',
    description: `
## ResNet Basic Block

Very deep plain CNNs train *worse* than shallower ones: optimization, not overfitting, is the problem. ResNet's fix is the **skip connection**: each block learns a residual \`F(x)\` and outputs \`ReLU(F(x) + x)\`. If the identity is the best a block can do, it only has to push \`F\` to zero, and gradients flow straight through the additions.

Implement \`residual_block(x, params, stride=1)\`, the "basic block" of ResNet-18/34:
\`\`\`
x ──► conv3x3(stride) ─► BN ─► ReLU ─► conv3x3 ─► BN ──► (+) ─► ReLU ─► out
│                                                         ▲
└──────────────────────── shortcut ───────────────────────┘
\`\`\`

### Main path
1. \`conv1\`: 3x3, stride \`stride\`, padding 1, \`C_in → C_out\`, then \`BN1\`, then ReLU
2. \`conv2\`: 3x3, stride 1, padding 1, \`C_out → C_out\`, then \`BN2\` (**no** ReLU yet)

### Shortcut
- **Identity** when \`stride == 1\` and \`C_in == C_out\`: the shapes already match.
- **Projection** otherwise: a 1x1 conv with stride \`stride\` (\`C_in → C_out\`) followed by its own BatchNorm.

### Output
\`\`\`
out = ReLU(main + shortcut)          # the ReLU comes AFTER the addition
\`\`\`
The convs have no bias (BatchNorm's \`beta\` makes it redundant). The spatial size is \`H_out = (H - 1) // stride + 1\`.

### Provided helpers (already in the starter code)
- \`conv2d(x, w, stride, padding)\`: \`x (N, C_in, H, W)\`, \`w (C_out, C_in, k, k)\`
- \`batch_norm2d(x, gamma, beta)\`: **training-mode** BatchNorm (per-channel batch statistics over \`N, H, W\`, biased variance, \`eps = 1e-5\`), like \`nn.BatchNorm2d\` in \`.train()\` mode
- \`init_block_params(c_in, c_out, stride)\`: deterministic parameters

### Expected \`params\` keys
- \`'conv1_w'\` \`(C_out, C_in, 3, 3)\`, \`'bn1_gamma'\`, \`'bn1_beta'\` \`(C_out,)\`
- \`'conv2_w'\` \`(C_out, C_out, 3, 3)\`, \`'bn2_gamma'\`, \`'bn2_beta'\` \`(C_out,)\`
- Only for a projection: \`'proj_w'\` \`(C_out, C_in, 1, 1)\`, \`'proj_gamma'\`, \`'proj_beta'\` \`(C_out,)\`

Return the output array of shape \`(N, C_out, H_out, W_out)\`.
    `,
    examples: [
      {
        input: 'x: (2, 4, 6, 6), params = init_block_params(4, 4, 1), stride = 1',
        output: 'shape (2, 4, 6, 6)',
        explanation: 'Identity shortcut: channels and spatial size are unchanged.',
      },
      {
        input: 'x: (2, 3, 6, 6), params = init_block_params(3, 8, 2), stride = 2',
        output: 'shape (2, 8, 3, 3)',
        explanation: 'Channels change and the resolution halves, so the shortcut is a strided 1x1 conv + BN.',
      },
    ],
    starterCode: `import numpy as np

def conv2d(x, w, stride=1, padding=0):
    """Convolution (cross-correlation), no bias. x: (N, C_in, H, W), w: (C_out, C_in, k, k)."""
    N, C, H, W = x.shape
    k = w.shape[2]
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    H_out = (H + 2 * padding - k) // stride + 1
    W_out = (W + 2 * padding - k) // stride + 1
    out = np.zeros((N, w.shape[0], H_out, W_out))
    for a in range(k):
        for b in range(k):
            patch = xp[:, :, a:a + stride * H_out:stride, b:b + stride * W_out:stride]
            out += np.einsum('nchw,oc->nohw', patch, w[:, :, a, b])
    return out


def batch_norm2d(x, gamma, beta, eps=1e-5):
    """Training-mode BatchNorm2d: normalize each channel with batch statistics over (N, H, W)."""
    mu = x.mean(axis=(0, 2, 3), keepdims=True)
    var = x.var(axis=(0, 2, 3), keepdims=True)
    return gamma[None, :, None, None] * (x - mu) / np.sqrt(var + eps) + beta[None, :, None, None]


def init_block_params(c_in, c_out, stride=1):
    """Deterministic parameters for a basic block (proj_* only if a projection is needed)."""
    def weight(shape, phase):
        fan_in = shape[1] * shape[2] * shape[3]
        return np.sin(np.arange(np.prod(shape)) * 1.3 + phase).reshape(shape) / np.sqrt(fan_in)

    def affine(c, phase):
        return 1 + 0.1 * np.cos(np.arange(c) + phase), 0.05 * np.sin(np.arange(c) + phase)

    p = {'conv1_w': weight((c_out, c_in, 3, 3), 0.0), 'conv2_w': weight((c_out, c_out, 3, 3), 1.0)}
    p['bn1_gamma'], p['bn1_beta'] = affine(c_out, 0.0)
    p['bn2_gamma'], p['bn2_beta'] = affine(c_out, 2.0)
    if stride != 1 or c_in != c_out:
        p['proj_w'] = weight((c_out, c_in, 1, 1), 2.0)
        p['proj_gamma'], p['proj_beta'] = affine(c_out, 4.0)
    return p


def residual_block(x, params, stride=1):
    """
    Basic residual block (training-mode BatchNorm).

    main     = BN2(conv2(ReLU(BN1(conv1(x)))))      conv1: 3x3, stride, pad 1; conv2: 3x3, pad 1
    shortcut = x                                   if stride == 1 and C_in == C_out
             = BN_proj(conv1x1(x, stride))         otherwise
    out      = ReLU(main + shortcut)

    Args:
        x: Input, shape (N, C_in, H, W)
        params: dict from init_block_params(C_in, C_out, stride)
        stride: Stride of conv1 (and of the projection)

    Returns:
        Output, shape (N, C_out, H_out, W_out) with H_out = (H - 1) // stride + 1
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Identity block keeps the shape',
        input: 'residual_block(np.random.randn(2, 4, 6, 6), init_block_params(4, 4, 1), 1).shape',
        expected: '(2, 4, 6, 6)',
        hidden: false,
      },
      {
        id: '2',
        description: 'Projection block with stride 2',
        input: 'residual_block(np.random.randn(2, 3, 6, 6), init_block_params(3, 8, 2), 2).shape',
        expected: '(2, 8, 3, 3)',
        hidden: false,
      },
      {
        id: '3',
        description: 'Output is non-negative (ReLU after the addition)',
        input: '(lambda x: (lambda out: bool(out.min() >= 0 and (out == 0).any() and not np.allclose(out, np.maximum(x, 0))))(residual_block(x, init_block_params(4, 4, 1), 1)))(np.random.randn(2, 4, 5, 5))',
        expected: 'True',
        hidden: false,
      },
      {
        id: '4',
        description: 'Zeroed convs: identity block reduces to ReLU(x + bn2_beta)',
        input: `(lambda: (
    x := np.random.randn(2, 4, 5, 5),
    p := {**init_block_params(4, 4, 1), 'conv1_w': np.zeros((4, 4, 3, 3)), 'conv2_w': np.zeros((4, 4, 3, 3))},
    bool(np.allclose(residual_block(x, p, 1), np.maximum(x + p['bn2_beta'][None, :, None, None], 0)))
)[-1])()`,
        expected: 'True',
        hidden: true,
      },
      {
        id: '5',
        description: 'Zeroed convs: projection shortcut gives ReLU(bn2_beta + proj_beta)',
        input: `(lambda: (
    p := {**init_block_params(3, 6, 2), 'conv1_w': np.zeros((6, 3, 3, 3)), 'conv2_w': np.zeros((6, 6, 3, 3)), 'proj_w': np.zeros((6, 3, 1, 1))},
    out := residual_block(np.random.randn(2, 3, 4, 4), p, 2),
    bool(out.shape == (2, 6, 2, 2) and np.allclose(out, np.maximum(p['bn2_beta'] + p['proj_beta'], 0)[None, :, None, None]))
)[-1])()`,
        expected: 'True',
        hidden: true,
      },
      {
        id: '6',
        description: 'Identity block values',
        input: '(lambda out: (round(float(out.sum()), 4), np.round(out[1, 2, 1], 4).tolist()))(residual_block(np.cos(np.arange(2 * 4 * 5 * 5) * 0.37).reshape(2, 4, 5, 5), init_block_params(4, 4, 1), 1))',
        expected: '(81.5228, [0.5662, 0.949, 1.2229, 0.0, 0.0])',
        hidden: true,
      },
      {
        id: '7',
        description: 'Projection block values',
        input: '(lambda out: (round(float(out.sum()), 4), np.round(out[0, :, 1, 2], 4).tolist()))(residual_block(np.cos(np.arange(2 * 3 * 6 * 6) * 0.37).reshape(2, 3, 6, 6), init_block_params(3, 8, 2), 2))',
        expected: '(84.7368, [0.6048, 1.2406, 0.0, 1.5251, 0.0, 0.0, 0.5989, 0.0])',
        hidden: true,
      },
    ],
    hints: [
      'Main path: conv2d(x, conv1_w, stride, padding=1) -> batch_norm2d -> ReLU -> conv2d(..., conv2_w, 1, 1) -> batch_norm2d.',
      'Use a projection shortcut exactly when stride != 1 or C_in != C_out: batch_norm2d(conv2d(x, proj_w, stride=stride), proj_gamma, proj_beta).',
      'The final ReLU is applied after adding the shortcut, not before.',
    ],
    solution: `import numpy as np

def conv2d(x, w, stride=1, padding=0):
    """Convolution (cross-correlation), no bias. x: (N, C_in, H, W), w: (C_out, C_in, k, k)."""
    N, C, H, W = x.shape
    k = w.shape[2]
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    H_out = (H + 2 * padding - k) // stride + 1
    W_out = (W + 2 * padding - k) // stride + 1
    out = np.zeros((N, w.shape[0], H_out, W_out))
    for a in range(k):
        for b in range(k):
            patch = xp[:, :, a:a + stride * H_out:stride, b:b + stride * W_out:stride]
            out += np.einsum('nchw,oc->nohw', patch, w[:, :, a, b])
    return out


def batch_norm2d(x, gamma, beta, eps=1e-5):
    """Training-mode BatchNorm2d: normalize each channel with batch statistics over (N, H, W)."""
    mu = x.mean(axis=(0, 2, 3), keepdims=True)
    var = x.var(axis=(0, 2, 3), keepdims=True)
    return gamma[None, :, None, None] * (x - mu) / np.sqrt(var + eps) + beta[None, :, None, None]


def init_block_params(c_in, c_out, stride=1):
    """Deterministic parameters for a basic block (proj_* only if a projection is needed)."""
    def weight(shape, phase):
        fan_in = shape[1] * shape[2] * shape[3]
        return np.sin(np.arange(np.prod(shape)) * 1.3 + phase).reshape(shape) / np.sqrt(fan_in)

    def affine(c, phase):
        return 1 + 0.1 * np.cos(np.arange(c) + phase), 0.05 * np.sin(np.arange(c) + phase)

    p = {'conv1_w': weight((c_out, c_in, 3, 3), 0.0), 'conv2_w': weight((c_out, c_out, 3, 3), 1.0)}
    p['bn1_gamma'], p['bn1_beta'] = affine(c_out, 0.0)
    p['bn2_gamma'], p['bn2_beta'] = affine(c_out, 2.0)
    if stride != 1 or c_in != c_out:
        p['proj_w'] = weight((c_out, c_in, 1, 1), 2.0)
        p['proj_gamma'], p['proj_beta'] = affine(c_out, 4.0)
    return p


def residual_block(x, params, stride=1):
    """
    Basic residual block (training-mode BatchNorm).

    main     = BN2(conv2(ReLU(BN1(conv1(x)))))      conv1: 3x3, stride, pad 1; conv2: 3x3, pad 1
    shortcut = x                                   if stride == 1 and C_in == C_out
             = BN_proj(conv1x1(x, stride))         otherwise
    out      = ReLU(main + shortcut)

    Args:
        x: Input, shape (N, C_in, H, W)
        params: dict from init_block_params(C_in, C_out, stride)
        stride: Stride of conv1 (and of the projection)

    Returns:
        Output, shape (N, C_out, H_out, W_out) with H_out = (H - 1) // stride + 1
    """
    p = params
    c_in, c_out = x.shape[1], p['conv1_w'].shape[0]

    # Main path: conv-BN-ReLU-conv-BN
    out = conv2d(x, p['conv1_w'], stride=stride, padding=1)
    out = np.maximum(batch_norm2d(out, p['bn1_gamma'], p['bn1_beta']), 0)
    out = conv2d(out, p['conv2_w'], stride=1, padding=1)
    out = batch_norm2d(out, p['bn2_gamma'], p['bn2_beta'])

    # Shortcut: identity, or 1x1 projection when the shape changes
    if stride != 1 or c_in != c_out:
        shortcut = batch_norm2d(conv2d(x, p['proj_w'], stride=stride), p['proj_gamma'], p['proj_beta'])
    else:
        shortcut = x

    # Add, THEN ReLU
    return np.maximum(out + shortcut, 0)
`,
  },
];
