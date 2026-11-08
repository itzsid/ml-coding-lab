import { LanguageVariant } from '../../../types';

/**
 * PyTorch tracks for the problems in ../post-training.ts, keyed by problem id.
 * Code runs against the NumPy-backed torch shim (public/py/torch_shim.py).
 */
export const postTrainingPyTorch: Record<string, LanguageVariant> = {
  'rl-reward-modeling': {
    note: "Use torch.sigmoid and torch.log. Don't use F.binary_cross_entropy_with_logits (it computes exactly this loss when the target is 1); a hidden test uses it as the reference.",
    starterCode: `import torch

def reward_model_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> float:
    """
    Compute RLHF reward model loss (Bradley-Terry).

    Args:
        r_chosen: Reward scores for preferred/chosen responses
        r_rejected: Reward scores for rejected responses

    Returns:
        Average negative log-likelihood loss (Python float rounded to 3 decimals)
    """
    r_chosen = r_chosen.double()
    r_rejected = r_rejected.double()

    # Your code here
    # Loss = -E[log σ(r_chosen - r_rejected)]
    pass
`,
    solution: `import torch

def reward_model_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> float:
    r_chosen = r_chosen.double()
    r_rejected = r_rejected.double()

    # P(chosen > rejected) = σ(r_chosen - r_rejected)
    logits = r_chosen - r_rejected
    loss = -torch.log(torch.sigmoid(logits)).mean()
    return round(loss.item(), 3)
`,
    testCases: [
      {
        id: '1',
        description: 'Clear preference',
        input: 'reward_model_loss(torch.tensor([2.0, 1.5]), torch.tensor([1.0, 0.5]))',
        expected: '0.313',
        hidden: false,
      },
      {
        id: '2',
        description: 'Equal scores',
        input: 'reward_model_loss(torch.tensor([0.0]), torch.tensor([0.0]))',
        expected: '0.693',
        hidden: false,
      },
      {
        id: '3',
        description: 'Large margin',
        input: 'reward_model_loss(torch.tensor([5.0, 4.0, 3.0]), torch.tensor([0.0, 0.0, 0.0]))',
        expected: '0.024',
        hidden: true,
      },
      {
        id: '4',
        description: 'Matches BCE-with-logits against target 1',
        input: `(lambda: (
            c := torch.randn(8), r := torch.randn(8),
            bool(abs(reward_model_loss(c, r) - torch.nn.functional.binary_cross_entropy_with_logits(c - r, torch.ones(8)).item()) < 1e-3)
        )[-1])()`,
        expected: 'True',
        hidden: true,
      },
    ],
    hints: [
      'Compute the difference: r_chosen - r_rejected',
      'torch.sigmoid gives the preference probability',
      'Loss: -torch.log(probs).mean()',
    ],
  },
  'rl-dpo-loss': {
    note: 'Pure tensor arithmetic with torch.sigmoid and torch.log. Return a Python float.',
    starterCode: `import torch

def dpo_loss(logp_chosen: torch.Tensor, logp_rejected: torch.Tensor,
             ref_logp_chosen: torch.Tensor, ref_logp_rejected: torch.Tensor,
             beta: float) -> float:
    """
    Compute Direct Preference Optimization (DPO) loss.

    Args:
        logp_chosen: Log prob of chosen response under policy π_θ
        logp_rejected: Log prob of rejected response under policy π_θ
        ref_logp_chosen: Log prob of chosen under reference π_ref
        ref_logp_rejected: Log prob of rejected under reference π_ref
        beta: Temperature parameter

    Returns:
        DPO loss (Python float rounded to 3 decimals)
    """
    logp_chosen = logp_chosen.double()
    logp_rejected = logp_rejected.double()
    ref_logp_chosen = ref_logp_chosen.double()
    ref_logp_rejected = ref_logp_rejected.double()

    # Your code here
    pass
`,
    solution: `import torch

def dpo_loss(logp_chosen: torch.Tensor, logp_rejected: torch.Tensor,
             ref_logp_chosen: torch.Tensor, ref_logp_rejected: torch.Tensor,
             beta: float) -> float:
    logp_chosen = logp_chosen.double()
    logp_rejected = logp_rejected.double()
    ref_logp_chosen = ref_logp_chosen.double()
    ref_logp_rejected = ref_logp_rejected.double()

    # Log-ratios of policy vs reference
    chosen_logratios = logp_chosen - ref_logp_chosen
    rejected_logratios = logp_rejected - ref_logp_rejected

    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -torch.log(torch.sigmoid(logits)).mean()
    return round(loss.item(), 3)
`,
    testCases: [
      {
        id: '1',
        description: 'Policy prefers chosen',
        input: 'dpo_loss(torch.tensor([-1.0]), torch.tensor([-2.0]), torch.tensor([-1.5]), torch.tensor([-1.5]), 0.1)',
        expected: '0.644',
        hidden: false,
      },
      {
        id: '2',
        description: 'Equal preferences',
        input: 'dpo_loss(torch.tensor([-1.0, -1.0]), torch.tensor([-1.0, -1.0]), torch.tensor([-1.0, -1.0]), torch.tensor([-1.0, -1.0]), 0.5)',
        expected: '0.693',
        hidden: false,
      },
      {
        id: '3',
        description: 'Strong preference signal',
        input: 'dpo_loss(torch.tensor([0.0]), torch.tensor([-5.0]), torch.tensor([-2.0]), torch.tensor([-3.0]), 0.1)',
        expected: '0.513',
        hidden: true,
      },
      {
        id: '4',
        description: 'Policy prefers the rejected response: loss above log 2',
        input: 'dpo_loss(torch.tensor([-3.0, -2.0]), torch.tensor([-1.0, -1.0]), torch.tensor([-2.0, -2.0]), torch.tensor([-2.0, -2.0]), 0.5)',
        expected: '1.144',
        hidden: true,
      },
    ],
    hints: [
      'Δ_w = logp_chosen - ref_logp_chosen, Δ_l = logp_rejected - ref_logp_rejected',
      'logits = beta * (Δ_w - Δ_l)',
      'Loss: -torch.log(torch.sigmoid(logits)).mean()',
    ],
  },
  'rl-kl-penalty': {
    note: 'One line of tensor arithmetic. Return a Python float.',
    starterCode: `import torch

def kl_penalty(logp_policy: torch.Tensor, logp_reference: torch.Tensor,
               beta: float) -> float:
    """
    Compute KL divergence penalty for RLHF.

    Args:
        logp_policy: Log probs from current policy π_θ
        logp_reference: Log probs from reference policy π_ref
        beta: KL penalty coefficient

    Returns:
        KL penalty = β * mean(KL divergence) (Python float rounded to 2 decimals)
    """
    logp_policy = logp_policy.double()
    logp_reference = logp_reference.double()

    # Your code here
    pass
`,
    solution: `import torch

def kl_penalty(logp_policy: torch.Tensor, logp_reference: torch.Tensor,
               beta: float) -> float:
    logp_policy = logp_policy.double()
    logp_reference = logp_reference.double()

    # Per-token KL estimate: log π_θ - log π_ref
    mean_kl = (logp_policy - logp_reference).mean()
    return round((beta * mean_kl).item(), 2)
`,
    testCases: [
      {
        id: '1',
        description: 'Positive divergence',
        input: 'kl_penalty(torch.tensor([-1.0, -2.0]), torch.tensor([-1.5, -2.5]), 0.1)',
        expected: '0.05',
        hidden: false,
      },
      {
        id: '2',
        description: 'No divergence',
        input: 'kl_penalty(torch.tensor([-1.0, -1.0]), torch.tensor([-1.0, -1.0]), 0.1)',
        expected: '0.0',
        hidden: false,
      },
      {
        id: '3',
        description: 'Higher beta',
        input: 'kl_penalty(torch.tensor([-0.5, -1.0, -1.5]), torch.tensor([-1.0, -1.5, -2.0]), 0.5)',
        expected: '0.25',
        hidden: true,
      },
      {
        id: '4',
        description: 'Mean over tokens of mixed-sign differences',
        input: 'kl_penalty(torch.tensor([-1.0, -3.0, -0.5, -2.0]), torch.tensor([-2.0, -2.0, -1.5, -2.6]), 1.0)',
        expected: '0.4',
        hidden: true,
      },
    ],
    hints: [
      'Per-token KL: logp_policy - logp_reference',
      'Take the mean over tokens and multiply by beta',
      'Return round(penalty.item(), 2)',
    ],
  },
  'rl-ppo-llm-objective': {
    note: 'Use torch.exp, torch.clamp and torch.minimum. Return a dict of Python floats.',
    starterCode: `import torch

def ppo_rlhf_objective(old_logp: torch.Tensor, new_logp: torch.Tensor,
                       ref_logp: torch.Tensor, rewards: torch.Tensor,
                       epsilon: float, beta: float) -> dict:
    """
    Compute PPO-RLHF objective for LLM fine-tuning.

    Args:
        old_logp: Log probs from old policy (before update)
        new_logp: Log probs from current policy
        ref_logp: Log probs from reference (pretrained) model
        rewards: Rewards from reward model
        epsilon: PPO clipping parameter
        beta: KL penalty coefficient

    Returns:
        Dictionary with 'objective', 'policy_loss', 'kl_penalty' (Python floats rounded to 2 decimals)
    """
    old_logp = old_logp.double()
    new_logp = new_logp.double()
    ref_logp = ref_logp.double()
    rewards = rewards.double()

    # Your code here
    pass
`,
    solution: `import torch

def ppo_rlhf_objective(old_logp: torch.Tensor, new_logp: torch.Tensor,
                       ref_logp: torch.Tensor, rewards: torch.Tensor,
                       epsilon: float, beta: float) -> dict:
    old_logp = old_logp.double()
    new_logp = new_logp.double()
    ref_logp = ref_logp.double()
    rewards = rewards.double()

    # Probability ratio π_θ / π_old from log probs
    ratio = torch.exp(new_logp - old_logp)

    # KL from the reference model, and the penalty
    kl = new_logp - ref_logp
    kl_penalty = beta * kl.mean()

    # RLHF advantage: reward minus per-token KL penalty
    advantage = rewards - beta * kl

    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_loss = torch.minimum(ratio * advantage, clipped_ratio * advantage).mean()

    objective = policy_loss

    return {
        'objective': round(objective.item(), 2),
        'policy_loss': round(policy_loss.item(), 2),
        'kl_penalty': round(kl_penalty.item(), 2),
    }
`,
    testCases: [
      {
        id: '1',
        description: 'Positive reward, small update',
        input: `(lambda r: bool(abs(r['objective'] - 1.18) < 0.1 and abs(r['kl_penalty'] - 0.02) < 0.01))(ppo_rlhf_objective(torch.tensor([-2.0]), torch.tensor([-1.8]), torch.tensor([-2.0]), torch.tensor([1.0]), 0.2, 0.1))`,
        expected: 'True',
        hidden: false,
      },
      {
        id: '2',
        description: 'No policy change',
        input: `(lambda r: bool(abs(r['objective'] - 1.0) < 0.01 and abs(r['kl_penalty']) < 0.01))(ppo_rlhf_objective(torch.tensor([-1.0]), torch.tensor([-1.0]), torch.tensor([-1.0]), torch.tensor([1.0]), 0.2, 0.1))`,
        expected: 'True',
        hidden: false,
      },
      {
        id: '3',
        description: 'Clipped ratio and KL penalty values',
        input: `(lambda r: (r['policy_loss'], r['kl_penalty']))(ppo_rlhf_objective(torch.tensor([-2.0, -1.0]), torch.tensor([-1.0, -1.0]), torch.tensor([-1.5, -2.0]), torch.tensor([2.0, 1.0]), 0.2, 0.2))`,
        expected: '(1.54, 0.15)',
        hidden: true,
      },
    ],
    hints: [
      'ratio = torch.exp(new_logp - old_logp)',
      'kl = new_logp - ref_logp; kl_penalty = beta * kl.mean()',
      'advantage = rewards - beta * kl',
      'policy_loss = torch.minimum(ratio * A, torch.clamp(ratio, 1 - eps, 1 + eps) * A).mean()',
    ],
  },

  'grpo-advantages': {
    note: "Pure tensor ops. Use the population std (r.std(dim=1, unbiased=False, keepdim=True)); torch's default std is unbiased and gives different advantages. Return Python floats for the scalars.",
    starterCode: `import torch

def group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Group-normalized advantages.

    Args:
        rewards: (B, G) rewards, G sampled responses per prompt
        eps: added to the std

    Returns:
        (B, G) tensor: (r - group mean) / (group std + eps), population std
    """
    # Your code here
    pass


def grpo_objective(logp, old_logp, ref_logp, rewards, mask,
                   epsilon=0.2, beta=0.04, eps=1e-8):
    """
    GRPO clipped objective with a k3 KL penalty.

    Args:
        logp, old_logp, ref_logp: (B, G, T) token log-probs (current, sampling, reference)
        rewards: (B, G) one scalar reward per response
        mask: (B, G, T) 1 for real tokens, 0 for padding
        epsilon: clipping range
        beta: KL coefficient
        eps: std epsilon for the advantages

    Returns:
        dict with 'advantages' (B, G) tensor, 'objective', 'loss', 'mean_kl' (Python floats)
    """
    # Your code here
    pass
`,
    solution: `import torch

def group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r = rewards.double()
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, unbiased=False, keepdim=True)  # population std
    return (r - mean) / (std + eps)


def grpo_objective(logp, old_logp, ref_logp, rewards, mask,
                   epsilon=0.2, beta=0.04, eps=1e-8):
    logp, old_logp, ref_logp = logp.double(), old_logp.double(), ref_logp.double()
    mask = mask.double()

    adv = group_advantages(rewards, eps)       # (B, G)
    A = adv.unsqueeze(-1)                      # broadcast over tokens

    ratio = torch.exp(logp - old_logp)
    surr = torch.minimum(ratio * A, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * A)

    # k3 KL estimator against the reference model
    d = ref_logp - logp
    kl = torch.exp(d) - d - 1

    per_token = surr - beta * kl

    lengths = mask.sum(dim=-1)
    objective = ((per_token * mask).sum(dim=-1) / lengths).mean()
    mean_kl = ((kl * mask).sum(dim=-1) / lengths).mean()

    return {
        'advantages': adv,
        'objective': objective.item(),
        'loss': -objective.item(),
        'mean_kl': mean_kl.item(),
    }
`,
    testCases: [
      {
        id: '1',
        description: 'Advantages for a 0/1 verifier reward',
        input: 'torch.round(group_advantages(torch.tensor([[1.0, 0.0, 0.0, 1.0]])), decimals=4).tolist()',
        expected: '[[1.0, -1.0, -1.0, 1.0]]',
        hidden: false,
      },
      {
        id: '2',
        description: 'On-policy, no drift: objective 0 and KL 0',
        input: `(lambda: (
    lp := torch.log(torch.full((2, 3, 4), 0.25)),
    r := grpo_objective(lp, lp, lp, torch.tensor([[1.0, 0.0, 2.0], [0.5, 0.5, 1.0]]), torch.ones(2, 3, 4)),
    (round(r['objective'], 6), round(r['mean_kl'], 6))
)[-1])()`,
        expected: '(0.0, 0.0)',
        hidden: false,
      },
      {
        id: '3',
        description: 'Identical rewards give zero advantages; population std is used',
        input: `(lambda a: [[round(v, 6) for v in row] for row in a.tolist()])(group_advantages(torch.tensor([[0.5, 0.5, 0.5], [1.0, 2.0, 3.0]])))`,
        expected: '[[0.0, 0.0, 0.0], [-1.224745, 0.0, 1.224745]]',
        hidden: true,
      },
      {
        id: '4',
        description: 'Objective, loss and KL with padding (values)',
        input: `(lambda: (
    r := grpo_objective(
        torch.tensor([[[-1.0, -0.5, -2.0], [-0.3, -1.2, -0.7], [-0.9, -0.4, -1.5]]]),
        torch.tensor([[[-1.1, -0.5, -1.8], [-0.4, -1.0, -0.7], [-0.9, -0.6, -1.5]]]),
        torch.tensor([[[-1.2, -0.7, -1.9], [-0.3, -1.1, -0.9], [-1.0, -0.4, -1.4]]]),
        torch.tensor([[1.0, 0.0, 0.5]]),
        torch.tensor([[[1, 1, 1], [1, 1, 0], [1, 1, 1]]]),
        epsilon=0.2, beta=0.1),
    (round(r['objective'], 4), round(r['loss'], 4), round(r['mean_kl'], 4))
)[-1])()`,
        expected: '(0.0045, -0.0045, 0.0067)',
        hidden: true,
      },
      {
        id: '5',
        description: 'Large ratio with positive advantage is clipped at 1 + epsilon',
        input: `(lambda: (
    r := grpo_objective(torch.tensor([[[0.0], [0.0]]]), torch.tensor([[[-1.0], [0.0]]]), torch.tensor([[[0.0], [0.0]]]),
                        torch.tensor([[1.0, 0.0]]), torch.tensor([[[1], [1]]]), epsilon=0.2, beta=0.0),
    round(r['objective'], 4)
)[-1])()`,
        expected: '0.1',
        hidden: true,
      },
    ],
    hints: [
      'Advantages: (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, unbiased=False, keepdim=True) + eps)',
      'adv.unsqueeze(-1) broadcasts the per-response advantage over tokens',
      'k3 KL: d = ref_logp - logp; kl = torch.exp(d) - d - 1',
      'Per response: (x * mask).sum(-1) / mask.sum(-1), then .mean(); return .item() for the scalars',
    ],
  },

  'int8-quantization': {
    note: 'Use torch.round, torch.clamp and .to(torch.int8). Accumulate the int8 matmul in an integer dtype (.long()) and rescale afterwards.',
    starterCode: `import torch

def quantize_per_channel(W: torch.Tensor):
    """Returns (W_q int8 (d_out, d_in), scales (d_out,)) with one scale per row."""
    # Your code here
    pass


def quantize_per_tensor(W: torch.Tensor):
    """Returns (W_q int8, scale float) with one scale for the whole matrix."""
    # Your code here
    pass


def dequantize(W_q: torch.Tensor, scales):
    """W_q * scale; scales is a (d_out,) tensor (per row) or a Python float."""
    # Your code here
    pass


def int8_matmul(x: torch.Tensor, W_q: torch.Tensor, w_scales: torch.Tensor) -> torch.Tensor:
    """
    Approximate x @ W.T with int8 operands.

    Args:
        x: (n, d_in) float activations (quantize per row)
        W_q: (d_out, d_in) int8 weights
        w_scales: (d_out,) per-row weight scales

    Returns:
        (n, d_out) float tensor
    """
    # Your code here
    pass


def quantization_report(W: torch.Tensor) -> dict:
    """Returns {'per_channel_error': float, 'per_tensor_error': float}."""
    # Your code here
    pass
`,
    solution: `import torch

def _absmax_quantize_rows(M):
    absmax = M.abs().max(dim=1).values
    scales = torch.clamp(absmax, min=1e-8) / 127.0
    M_q = torch.clamp(torch.round(M / scales.unsqueeze(1)), -127, 127).to(torch.int8)
    return M_q, scales


def quantize_per_channel(W):
    return _absmax_quantize_rows(W.double())


def quantize_per_tensor(W):
    W = W.double()
    scale = max(W.abs().max().item(), 1e-8) / 127.0
    W_q = torch.clamp(torch.round(W / scale), -127, 127).to(torch.int8)
    return W_q, scale


def dequantize(W_q, scales):
    s = torch.as_tensor(scales, dtype=torch.float64).reshape(-1, 1)
    return W_q.double() * s


def int8_matmul(x, W_q, w_scales):
    x_q, x_scales = _absmax_quantize_rows(x.double())   # per-token activation scales
    acc = x_q.long() @ W_q.long().T                      # integer accumulation
    return acc.double() * x_scales.unsqueeze(1) * w_scales.double().unsqueeze(0)


def quantization_report(W):
    W = W.double()
    per_channel = dequantize(*quantize_per_channel(W))
    per_tensor = dequantize(*quantize_per_tensor(W))
    return {
        'per_channel_error': (W - per_channel).abs().max().item(),
        'per_tensor_error': (W - per_tensor).abs().max().item(),
    }
`,
    testCases: [
      {
        id: '1',
        description: 'Per-channel int8 values',
        input: 'quantize_per_channel(torch.tensor([[0.7, -1.2, 0.3], [2.0, 0.1, -0.5]]))[0].tolist()',
        expected: '[[74, -127, 32], [127, 6, -32]]',
        hidden: false,
      },
      {
        id: '2',
        description: 'Per-channel scales and int8 dtype',
        input: '(lambda q: ([round(s, 6) for s in q[1].tolist()], q[0].dtype == torch.int8))(quantize_per_channel(torch.tensor([[0.7, -1.2, 0.3], [2.0, 0.1, -0.5]])))',
        expected: '([0.009449, 0.015748], True)',
        hidden: false,
      },
      {
        id: '3',
        description: 'Per-tensor error is larger when rows have very different ranges',
        input: `(lambda r: (round(r['per_channel_error'], 6), round(r['per_tensor_error'], 6)))(quantization_report(torch.tensor([[0.01, -0.02, 0.013], [5.0, -3.0, 1.7]], dtype=torch.float64)))`,
        expected: '(0.007874, 0.01937)',
        hidden: true,
      },
      {
        id: '4',
        description: 'Rounding error is at most scale / 2 for every weight',
        input: `(lambda: (
    W := torch.randn(8, 16, dtype=torch.float64) * torch.rand(8, 1, dtype=torch.float64) * 10,
    q := quantize_per_channel(W),
    bool(torch.all((dequantize(*q) - W).abs() <= q[1].unsqueeze(1) / 2 + 1e-12)) and q[0].abs().max().item() == 127
)[-1])()`,
        expected: 'True',
        hidden: true,
      },
      {
        id: '5',
        description: 'Int8 matmul values (rescaled with activation and weight scales)',
        input: `(lambda: (
    W := torch.tensor([[0.7, -1.2, 0.3], [2.0, 0.1, -0.5]], dtype=torch.float64),
    x := torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.25, 4.0]], dtype=torch.float64),
    [[round(v, 4) for v in row] for row in int8_matmul(x, *quantize_per_channel(W)).tolist()]
)[-1])()`,
        expected: '[[-2.0, 2.7126], [1.8642, -1.0317]]',
        hidden: true,
      },
      {
        id: '6',
        description: 'Int8 matmul stays close to the float matmul',
        input: `(lambda: (
    W := torch.randn(16, 32), x := torch.randn(4, 32),
    y := int8_matmul(x, *quantize_per_channel(W)),
    ref := x.double() @ W.double().T,
    bool((y - ref).abs().max() < 0.05 * ref.abs().max())
)[-1])()`,
        expected: 'True',
        hidden: true,
      },
    ],
    hints: [
      'Per-row absmax: W.abs().max(dim=1).values; scales = torch.clamp(absmax, min=1e-8) / 127',
      'torch.clamp(torch.round(W / scales.unsqueeze(1)), -127, 127).to(torch.int8)',
      'int8_matmul: quantize x per row, then x_q.long() @ W_q.long().T, and multiply by both scales',
    ],
  },
};
