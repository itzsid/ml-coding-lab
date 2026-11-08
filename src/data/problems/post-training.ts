import { Problem } from '../../types';

export const postTrainingProblems: Problem[] = [
  // ==========================================
  // RL + LLM PROBLEMS (RLHF, DPO, etc.)
  // ==========================================
  {
    id: 'rl-reward-modeling',
    title: 'RLHF Reward Model Loss',
    section: 'post-training',
    difficulty: 'hard',
    description: `
## RLHF Reward Model Loss

Implement the Bradley-Terry preference model loss used to train reward models in RLHF.

### Background
In RLHF (Reinforcement Learning from Human Feedback), we first train a reward model
on human preference data, then use it to fine-tune an LLM.

### Bradley-Terry Model
Given a pair of responses (y_w, y_l) where y_w is preferred (winner):
\`\`\`
P(y_w > y_l) = σ(r(y_w) - r(y_l))
\`\`\`

Where σ is the sigmoid function and r(y) is the reward model output.

### Loss Function
\`\`\`
L = -E[log σ(r(y_w) - r(y_l))]
\`\`\`

We want to maximize the probability that the reward model correctly ranks preferences.

### Function Signature
\`\`\`python
def reward_model_loss(r_chosen: np.ndarray, r_rejected: np.ndarray) -> float:
\`\`\`

Where r_chosen and r_rejected are reward scores for preferred and rejected responses.
    `,
    examples: [
      {
        input: 'r_chosen = [2.0, 1.5], r_rejected = [1.0, 0.5]',
        output: '0.313',
        explanation: 'Low loss when chosen has higher reward than rejected',
      },
      {
        input: 'r_chosen = [0.0], r_rejected = [0.0]',
        output: '0.693',
        explanation: 'Equal rewards = 50% probability = -log(0.5) = 0.693',
      },
    ],
    starterCode: `import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def reward_model_loss(r_chosen: np.ndarray, r_rejected: np.ndarray) -> float:
    """
    Compute RLHF reward model loss (Bradley-Terry).

    Args:
        r_chosen: Reward scores for preferred/chosen responses
        r_rejected: Reward scores for rejected responses

    Returns:
        Average negative log-likelihood loss
    """
    r_chosen = np.array(r_chosen, dtype=float)
    r_rejected = np.array(r_rejected, dtype=float)

    # Your code here
    # Loss = -E[log σ(r_chosen - r_rejected)]
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Clear preference',
        input: '([2.0, 1.5], [1.0, 0.5])',
        expected: '0.313',
        hidden: false,
      },
      {
        id: '2',
        description: 'Equal scores',
        input: '([0.0], [0.0])',
        expected: '0.693',
        hidden: false,
      },
      {
        id: '3',
        description: 'Large margin',
        input: '([5.0, 4.0, 3.0], [0.0, 0.0, 0.0])',
        expected: '0.024',
        hidden: true,
      },
    ],
    hints: [
      'Compute the difference: r_chosen - r_rejected',
      'Apply sigmoid to get probability',
      'Take log of probability',
      'Return negative mean (loss to minimize)',
    ],
    solution: `import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def reward_model_loss(r_chosen: np.ndarray, r_rejected: np.ndarray) -> float:
    """
    Compute RLHF reward model loss (Bradley-Terry).
    """
    r_chosen = np.array(r_chosen, dtype=float)
    r_rejected = np.array(r_rejected, dtype=float)

    # Probability of correctly ranking: σ(r_chosen - r_rejected)
    logits = r_chosen - r_rejected
    probs = sigmoid(logits)

    # Negative log-likelihood
    loss = -np.mean(np.log(probs + 1e-10))

    return round(loss, 3)
`,
  },
  {
    id: 'rl-dpo-loss',
    title: 'Direct Preference Optimization (DPO)',
    section: 'post-training',
    difficulty: 'hard',
    description: `
## Direct Preference Optimization (DPO) Loss

Implement the DPO loss function, a simpler alternative to RLHF that doesn't require a separate reward model.

### Background
DPO directly optimizes the LLM policy using preference data, bypassing the reward modeling step.

### DPO Loss
\`\`\`
L_DPO = -E[log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x)
                    - log π_θ(y_l|x) + log π_ref(y_l|x)))]
\`\`\`

Simplified:
\`\`\`
L_DPO = -E[log σ(β * (Δ_w - Δ_l))]
\`\`\`
Where Δ = log π_θ(y|x) - log π_ref(y|x) (log-ratio of policy to reference)

### Parameters
- β: Temperature parameter (typically 0.1 to 0.5)
- π_θ: Policy being trained
- π_ref: Reference policy (frozen)
- y_w: Preferred (winner) response
- y_l: Rejected (loser) response

### Function Signature
\`\`\`python
def dpo_loss(logp_chosen: np.ndarray, logp_rejected: np.ndarray,
             ref_logp_chosen: np.ndarray, ref_logp_rejected: np.ndarray,
             beta: float) -> float:
\`\`\`
    `,
    examples: [
      {
        input: 'logp_chosen=[-1], logp_rejected=[-2], ref_logp_chosen=[-1.5], ref_logp_rejected=[-1.5], beta=0.1',
        output: '0.644',
        explanation: 'Policy prefers chosen over reference',
      },
    ],
    starterCode: `import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dpo_loss(logp_chosen: np.ndarray, logp_rejected: np.ndarray,
             ref_logp_chosen: np.ndarray, ref_logp_rejected: np.ndarray,
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
        DPO loss
    """
    logp_chosen = np.array(logp_chosen, dtype=float)
    logp_rejected = np.array(logp_rejected, dtype=float)
    ref_logp_chosen = np.array(ref_logp_chosen, dtype=float)
    ref_logp_rejected = np.array(ref_logp_rejected, dtype=float)

    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Policy prefers chosen',
        input: '([-1.0], [-2.0], [-1.5], [-1.5], 0.1)',
        expected: '0.644',
        hidden: false,
      },
      {
        id: '2',
        description: 'Equal preferences',
        input: '([-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], 0.5)',
        expected: '0.693',
        hidden: false,
      },
      {
        id: '3',
        description: 'Strong preference signal',
        input: '([0.0], [-5.0], [-2.0], [-3.0], 0.1)',
        expected: '0.513',
        hidden: true,
      },
    ],
    hints: [
      'Compute log-ratios: Δ_w = logp_chosen - ref_logp_chosen',
      'Compute log-ratios: Δ_l = logp_rejected - ref_logp_rejected',
      'DPO logits: β * (Δ_w - Δ_l)',
      'Loss: -mean(log(sigmoid(logits)))',
    ],
    solution: `import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dpo_loss(logp_chosen: np.ndarray, logp_rejected: np.ndarray,
             ref_logp_chosen: np.ndarray, ref_logp_rejected: np.ndarray,
             beta: float) -> float:
    """
    Compute Direct Preference Optimization (DPO) loss.
    """
    logp_chosen = np.array(logp_chosen, dtype=float)
    logp_rejected = np.array(logp_rejected, dtype=float)
    ref_logp_chosen = np.array(ref_logp_chosen, dtype=float)
    ref_logp_rejected = np.array(ref_logp_rejected, dtype=float)

    # Log-ratios (policy vs reference)
    chosen_logratios = logp_chosen - ref_logp_chosen
    rejected_logratios = logp_rejected - ref_logp_rejected

    # DPO logits
    logits = beta * (chosen_logratios - rejected_logratios)

    # Negative log-sigmoid loss
    loss = -np.mean(np.log(sigmoid(logits) + 1e-10))

    return round(loss, 3)
`,
  },
  {
    id: 'rl-kl-penalty',
    title: 'KL Divergence Penalty (RLHF)',
    section: 'post-training',
    difficulty: 'medium',
    description: `
## KL Divergence Penalty for RLHF

Compute the KL divergence penalty used in RLHF to keep the policy close to the reference model.

### Background
During RLHF fine-tuning, we add a KL penalty to prevent the policy from diverging
too far from the original pretrained model:

\`\`\`
reward_total = reward - β * KL(π_θ || π_ref)
\`\`\`

### KL Divergence (per token)
\`\`\`
KL = E[log π_θ(a|s) - log π_ref(a|s)]
   = E[log(π_θ/π_ref)]
\`\`\`

For a sequence, we typically compute the average KL per token.

### Why KL Penalty?
- Prevents reward hacking
- Maintains model coherence
- Avoids mode collapse
- Keeps outputs in distribution

### Function Signature
\`\`\`python
def kl_penalty(logp_policy: np.ndarray, logp_reference: np.ndarray,
               beta: float) -> float:
\`\`\`

Returns: KL divergence multiplied by β
    `,
    examples: [
      {
        input: 'logp_policy = [-1, -2], logp_reference = [-1.5, -2.5], beta = 0.1',
        output: '0.05',
        explanation: 'KL = mean([0.5, 0.5]) = 0.5, penalty = 0.1 * 0.5 = 0.05',
      },
    ],
    starterCode: `import numpy as np

def kl_penalty(logp_policy: np.ndarray, logp_reference: np.ndarray,
               beta: float) -> float:
    """
    Compute KL divergence penalty for RLHF.

    Args:
        logp_policy: Log probs from current policy π_θ
        logp_reference: Log probs from reference policy π_ref
        beta: KL penalty coefficient

    Returns:
        KL penalty = β * mean(KL divergence)
    """
    logp_policy = np.array(logp_policy, dtype=float)
    logp_reference = np.array(logp_reference, dtype=float)

    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Positive divergence',
        input: '([-1.0, -2.0], [-1.5, -2.5], 0.1)',
        expected: '0.05',
        hidden: false,
      },
      {
        id: '2',
        description: 'No divergence',
        input: '([-1.0, -1.0], [-1.0, -1.0], 0.1)',
        expected: '0.0',
        hidden: false,
      },
      {
        id: '3',
        description: 'Higher beta',
        input: '([-0.5, -1.0, -1.5], [-1.0, -1.5, -2.0], 0.5)',
        expected: '0.25',
        hidden: true,
      },
    ],
    hints: [
      'KL divergence per token: log(π_θ) - log(π_ref)',
      'Take the mean over all tokens',
      'Multiply by β to get the penalty',
      'KL should be non-negative (though approximation may give small negatives)',
    ],
    solution: `import numpy as np

def kl_penalty(logp_policy: np.ndarray, logp_reference: np.ndarray,
               beta: float) -> float:
    """
    Compute KL divergence penalty for RLHF.
    """
    logp_policy = np.array(logp_policy, dtype=float)
    logp_reference = np.array(logp_reference, dtype=float)

    # KL divergence: log(π_θ/π_ref) = log(π_θ) - log(π_ref)
    kl_div = logp_policy - logp_reference

    # Mean KL (per token)
    mean_kl = np.mean(kl_div)

    # KL penalty
    penalty = beta * mean_kl

    return round(penalty, 2)
`,
  },
  {
    id: 'rl-ppo-llm-objective',
    title: 'PPO-RLHF Objective',
    section: 'post-training',
    difficulty: 'hard',
    description: `
## PPO Objective for LLM Fine-tuning (RLHF)

Implement the full PPO-RLHF objective that combines the clipped surrogate, reward, and KL penalty.\n\nThe KL penalty is applied through the reward (InstructGPT-style): \`advantage = reward - beta * KL\`, where \`KL = new_logp - ref_logp\`. The \`'objective'\` is the clipped surrogate computed with that advantage; \`'kl_penalty'\` reports \`beta * mean(KL)\` for monitoring.

### RLHF PPO Objective
\`\`\`
L = E[min(r_t * Â_t, clip(r_t, 1-ε, 1+ε) * Â_t) - β * KL]
\`\`\`

Where:
- r_t = π_θ(a|s) / π_old(a|s) is the probability ratio
- Â_t is the advantage (reward - baseline, typically from reward model)
- ε is the PPO clipping parameter (0.2)
- β is the KL penalty coefficient
- KL = log π_θ - log π_ref

### RLHF Advantage
In RLHF, the advantage is often simplified to:
\`\`\`
Â = R(x, y) - β * KL(π_θ || π_ref)
\`\`\`

### Function Signature
\`\`\`python
def ppo_rlhf_objective(old_logp: np.ndarray, new_logp: np.ndarray,
                       ref_logp: np.ndarray, rewards: np.ndarray,
                       epsilon: float, beta: float) -> dict:
\`\`\`

### Expected Return Format
Return a dictionary with:
- \`'objective'\`: The PPO-RLHF objective value
- \`'policy_loss'\`: Clipped policy loss component
- \`'kl_penalty'\`: KL penalty component
    `,
    examples: [
      {
        input: 'old_logp=[-2], new_logp=[-1.8], ref_logp=[-2], rewards=[1.0], ε=0.2, β=0.1',
        output: "{'objective': ..., 'policy_loss': ..., 'kl_penalty': ...}",
        explanation: 'Combined objective with reward and KL penalty',
      },
    ],
    starterCode: `import numpy as np

def ppo_rlhf_objective(old_logp: np.ndarray, new_logp: np.ndarray,
                       ref_logp: np.ndarray, rewards: np.ndarray,
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
        Dictionary with 'objective', 'policy_loss', 'kl_penalty'
    """
    old_logp = np.array(old_logp, dtype=float)
    new_logp = np.array(new_logp, dtype=float)
    ref_logp = np.array(ref_logp, dtype=float)
    rewards = np.array(rewards, dtype=float)

    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Positive reward, small update',
        input: `(lambda r: bool(abs(r['objective'] - 1.18) < 0.1 and abs(r['kl_penalty'] - 0.02) < 0.01))(ppo_rlhf_objective([-2.0], [-1.8], [-2.0], [1.0], 0.2, 0.1))`,
        expected: 'True',
        hidden: false,
      },
      {
        id: '2',
        description: 'No policy change',
        input: `(lambda r: bool(abs(r['objective'] - 1.0) < 0.01 and abs(r['kl_penalty']) < 0.01))(ppo_rlhf_objective([-1.0], [-1.0], [-1.0], [1.0], 0.2, 0.1))`,
        expected: 'True',
        hidden: false,
      },
      {
        id: '3',
        description: 'Clipping kicks in for a large ratio',
        input: '(lambda r: (round(float(r["objective"]), 3), round(float(r["kl_penalty"]), 3)))(ppo_rlhf_objective([-1.0, -2.0], [-0.5, -2.0], [-1.0, -2.0], [1.0, -1.0], 0.2, 0.1))',
        expected: '(0.07, 0.03)',
        hidden: true,
      },
    ],
    hints: [
      'Compute ratio: exp(new_logp - old_logp)',
      'Compute KL: new_logp - ref_logp',
      'Compute advantage: rewards - beta * KL',
      'Apply PPO clipping to ratio * advantage',
      'The KL penalty is already inside the advantage, so the objective is the clipped surrogate (report beta * mean KL separately)',
    ],
    solution: `import numpy as np

def ppo_rlhf_objective(old_logp: np.ndarray, new_logp: np.ndarray,
                       ref_logp: np.ndarray, rewards: np.ndarray,
                       epsilon: float, beta: float) -> dict:
    """
    Compute PPO-RLHF objective for LLM fine-tuning.
    """
    old_logp = np.array(old_logp, dtype=float)
    new_logp = np.array(new_logp, dtype=float)
    ref_logp = np.array(ref_logp, dtype=float)
    rewards = np.array(rewards, dtype=float)

    # Probability ratio
    ratio = np.exp(new_logp - old_logp)

    # KL divergence from reference
    kl = new_logp - ref_logp

    # KL penalty
    kl_penalty = beta * np.mean(kl)

    # Advantage: reward minus KL penalty
    advantage = rewards - beta * kl

    # Clipped ratio
    clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)

    # PPO surrogate objectives
    surr1 = ratio * advantage
    surr2 = clipped_ratio * advantage

    # Policy loss (take minimum for conservative update)
    policy_loss = np.mean(np.minimum(surr1, surr2))

    # Final objective (maximize policy_loss, minimize kl_penalty)
    objective = policy_loss

    return {
        'objective': round(float(objective), 2),
        'policy_loss': round(float(policy_loss), 2),
        'kl_penalty': round(float(kl_penalty), 2),
    }
`,
  },
];
