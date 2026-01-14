import { Problem } from '../../types';

export const generativeModelProblems: Problem[] = [
  {
    id: 'vae-reparameterization',
    title: 'VAE Reparameterization Trick',
    section: 'generative-models',
    difficulty: 'medium',
    description: `
## VAE Reparameterization Trick

Implement the reparameterization trick that allows backpropagation through stochastic sampling in VAEs.

### The Problem
We want to sample z ~ N(μ, σ²), but sampling is not differentiable.

### The Solution
\`\`\`
ε ~ N(0, 1)
z = μ + σ * ε
\`\`\`

### Why It Works
- ε is sampled independently of parameters
- z is now a deterministic function of μ, σ, and ε
- Gradients can flow through μ and σ
    `,
    examples: [
      {
        input: 'mu = [0, 1], log_var = [0, 0]',
        output: 'z = mu + exp(0.5 * log_var) * epsilon',
        explanation: 'log_var=0 means σ=1',
      },
    ],
    starterCode: `import numpy as np

def reparameterize(mu, log_var):
    """
    Sample from latent distribution using reparameterization trick.

    Args:
        mu: Mean of latent distribution (batch, latent_dim)
        log_var: Log variance of latent distribution (batch, latent_dim)

    Returns:
        z: Sampled latent vectors (batch, latent_dim)
    """
    np.random.seed(42)  # For reproducibility
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Output shape matches input',
        input: '(np.zeros((2, 4)), np.zeros((2, 4)))',
        expected: '(2, 4)',
        hidden: false,
      },
      {
        id: '2',
        description: 'Zero variance returns mu',
        input: 'zero_var_test',
        expected: 'True',
        hidden: true,
      },
    ],
    hints: [
      'std = exp(0.5 * log_var)',
      'Sample epsilon from standard normal',
      'z = mu + std * epsilon',
    ],
    solution: `import numpy as np

def reparameterize(mu, log_var):
    np.random.seed(42)
    # Compute standard deviation
    std = np.exp(0.5 * log_var)
    # Sample epsilon from standard normal
    eps = np.random.randn(*mu.shape)
    # Reparameterization trick
    z = mu + std * eps
    return z
`,
  },
  {
    id: 'vae-loss',
    title: 'VAE Loss Function',
    section: 'generative-models',
    difficulty: 'medium',
    description: `
## VAE Loss (ELBO)

Implement the VAE loss function: reconstruction loss + KL divergence.

### Loss Components
\`\`\`
L = L_reconstruction + L_KL

L_reconstruction = MSE(x, x_reconstructed)
L_KL = -0.5 * sum(1 + log_var - mu² - exp(log_var))
\`\`\`

### Intuition
- **Reconstruction loss**: Output should match input
- **KL divergence**: Latent distribution should be close to N(0,1)
    `,
    examples: [
      {
        input: 'Perfect reconstruction, mu=0, var=1',
        output: 'Loss ≈ 0',
        explanation: 'Both terms are minimized',
      },
    ],
    starterCode: `import numpy as np

def vae_loss(x, x_reconstructed, mu, log_var):
    """
    Compute VAE loss (negative ELBO).

    Args:
        x: Original input (batch, features)
        x_reconstructed: Reconstructed input (batch, features)
        mu: Latent mean (batch, latent_dim)
        log_var: Latent log variance (batch, latent_dim)

    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Perfect case',
        input: '(np.zeros((2, 4)), np.zeros((2, 4)), np.zeros((2, 2)), np.zeros((2, 2)))',
        expected: '(0.0, 0.0, 0.0)',
        hidden: false,
      },
      {
        id: '2',
        description: 'Non-zero KL',
        input: '(np.zeros((1, 4)), np.zeros((1, 4)), np.ones((1, 2)), np.zeros((1, 2)))',
        expected: 'True',
        hidden: true,
      },
    ],
    hints: [
      'Reconstruction: np.mean((x - x_reconstructed)²)',
      'KL: -0.5 * sum(1 + log_var - mu² - exp(log_var))',
      'Average over batch',
    ],
    solution: `import numpy as np

def vae_loss(x, x_reconstructed, mu, log_var):
    # Reconstruction loss (MSE)
    recon_loss = np.mean((x - x_reconstructed) ** 2)

    # KL divergence
    kl_loss = -0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var))

    total_loss = recon_loss + kl_loss

    return round(total_loss, 4), round(recon_loss, 4), round(kl_loss, 4)
`,
  },
  {
    id: 'diffusion-noise-schedule',
    title: 'Diffusion Noise Schedule',
    section: 'generative-models',
    difficulty: 'easy',
    description: `
## Diffusion Noise Schedule

Implement a linear noise schedule for diffusion models.

### Linear Schedule
\`\`\`
β_t = β_start + t * (β_end - β_start) / T

where t = 0, 1, ..., T-1
\`\`\`

### Derived Quantities
\`\`\`
α_t = 1 - β_t
ᾱ_t = prod(α_1, ..., α_t)  # cumulative product
\`\`\`

These control how much noise is added at each step.
    `,
    examples: [
      {
        input: 'T=1000, beta_start=0.0001, beta_end=0.02',
        output: 'betas, alphas, alpha_bars arrays',
        explanation: 'Standard DDPM schedule',
      },
    ],
    starterCode: `import numpy as np

def linear_noise_schedule(T, beta_start=0.0001, beta_end=0.02):
    """
    Create linear noise schedule for diffusion.

    Args:
        T: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value

    Returns:
        betas: Beta values (T,)
        alphas: Alpha values (T,)
        alpha_bars: Cumulative alpha product (T,)
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Beta range correct',
        input: '(100, 0.0001, 0.02)',
        expected: 'True',
        hidden: false,
      },
      {
        id: '2',
        description: 'Alpha bar decreases',
        input: '(50, 0.001, 0.01)',
        expected: 'True',
        hidden: true,
      },
    ],
    hints: [
      'Use np.linspace for linear interpolation',
      'alphas = 1 - betas',
      'alpha_bars = np.cumprod(alphas)',
    ],
    solution: `import numpy as np

def linear_noise_schedule(T, beta_start=0.0001, beta_end=0.02):
    # Linear schedule
    betas = np.linspace(beta_start, beta_end, T)

    # Compute alphas
    alphas = 1 - betas

    # Cumulative product
    alpha_bars = np.cumprod(alphas)

    return betas, alphas, alpha_bars
`,
  },
  {
    id: 'diffusion-forward',
    title: 'Diffusion Forward Process',
    section: 'generative-models',
    difficulty: 'medium',
    description: `
## Diffusion Forward Process (Adding Noise)

Implement the forward diffusion process that adds noise to data.

### Formula
\`\`\`
x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

where ε ~ N(0, I)
\`\`\`

### Intuition
- As t increases, ᾱ_t decreases
- More noise is added, signal is reduced
- At t=T, x_T ≈ pure noise
    `,
    examples: [
      {
        input: 'x_0 (image), t=500, T=1000',
        output: 'x_t (noisy image)',
        explanation: 'Halfway through diffusion process',
      },
    ],
    starterCode: `import numpy as np

def diffusion_forward(x_0, t, alpha_bars):
    """
    Add noise to data using forward diffusion.

    Args:
        x_0: Original data (batch, ...)
        t: Timestep (int)
        alpha_bars: Cumulative alpha products (T,)

    Returns:
        x_t: Noisy data at timestep t
        noise: The noise that was added
    """
    np.random.seed(42)
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Output shape matches input',
        input: '(np.random.randn(2, 4), 50, np.linspace(0.99, 0.01, 100))',
        expected: '(2, 4)',
        hidden: false,
      },
      {
        id: '2',
        description: 't=0 returns near original',
        input: 't_zero_test',
        expected: 'True',
        hidden: true,
      },
    ],
    hints: [
      'Get alpha_bar_t = alpha_bars[t]',
      'Sample noise from standard normal',
      'Apply formula: sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise',
    ],
    solution: `import numpy as np

def diffusion_forward(x_0, t, alpha_bars):
    np.random.seed(42)

    alpha_bar_t = alpha_bars[t]

    # Sample noise
    noise = np.random.randn(*x_0.shape)

    # Forward process
    x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1 - alpha_bar_t) * noise

    return x_t, noise
`,
  },
  {
    id: 'kl-divergence',
    title: 'KL Divergence (Gaussians)',
    section: 'generative-models',
    difficulty: 'easy',
    description: `
## KL Divergence Between Gaussians

Compute the KL divergence between two univariate Gaussian distributions.

### Formula
\`\`\`
KL(P || Q) = log(σ_q/σ_p) + (σ_p² + (μ_p - μ_q)²) / (2σ_q²) - 0.5
\`\`\`

Where P = N(μ_p, σ_p²) and Q = N(μ_q, σ_q²)

### Properties
- KL ≥ 0
- KL = 0 iff P = Q
- Not symmetric: KL(P||Q) ≠ KL(Q||P)
    `,
    examples: [
      {
        input: 'P = N(0, 1), Q = N(0, 1)',
        output: '0',
        explanation: 'Same distribution',
      },
    ],
    starterCode: `import numpy as np

def kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    """
    Compute KL divergence between two Gaussians.

    Args:
        mu_p, sigma_p: Mean and std of P
        mu_q, sigma_q: Mean and std of Q

    Returns:
        kl: KL(P || Q)
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Same distribution',
        input: '(0, 1, 0, 1)',
        expected: '0.0',
        hidden: false,
      },
      {
        id: '2',
        description: 'Different means',
        input: '(1, 1, 0, 1)',
        expected: '0.5',
        hidden: false,
      },
      {
        id: '3',
        description: 'Different variances',
        input: '(0, 2, 0, 1)',
        expected: '0.4431',
        hidden: true,
      },
    ],
    hints: [
      'Apply the formula directly',
      'Use np.log for natural logarithm',
      'Remember: σ² is variance, σ is std',
    ],
    solution: `import numpy as np

def kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    term1 = np.log(sigma_q / sigma_p)
    term2 = (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2)
    term3 = -0.5

    kl = term1 + term2 + term3
    return round(kl, 4)
`,
  },
];
