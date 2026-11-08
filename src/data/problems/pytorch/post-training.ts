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
};
