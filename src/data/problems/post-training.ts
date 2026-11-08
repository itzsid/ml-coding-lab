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
  // ==========================================
  // GRPO AND QUANTIZATION
  // ==========================================
  {
    id: 'grpo-advantages',
    title: 'GRPO: Group-Relative Advantages & Objective',
    section: 'post-training',
    difficulty: 'hard',
    description: `
## GRPO: Group Relative Policy Optimization

GRPO (DeepSeekMath, Shao et al. 2024) is PPO without a value network. For each prompt, sample a **group** of G responses, score each with a reward model (or a verifier), and use the reward **relative to the group** as the advantage. The group mean plays the role of the critic's baseline.

### Step 1: group-normalized advantages
For prompt b with rewards r_{b,1..G}:
\`\`\`
A_{b,i} = (r_{b,i} - mean_i(r_{b,:})) / (std_i(r_{b,:}) + eps)
\`\`\`
- \`std\` is the **population** standard deviation (ddof = 0, i.e. \`np.std\` default; in PyTorch \`unbiased=False\`)
- \`eps = 1e-8\`, so a group where every response gets the same reward has advantage 0 everywhere (no learning signal, no division by zero)
- This is outcome supervision: every token of response i gets the same advantage A_{b,i}

### Step 2: per-token clipped surrogate with a KL term
For token t of response i (log-probs of the sampled token):
\`\`\`
ratio = exp(logp - old_logp)
surr  = min(ratio * A, clip(ratio, 1 - ε, 1 + ε) * A)

# k3 estimator of KL(π_θ || π_ref), always >= 0
kl    = exp(ref_logp - logp) - (ref_logp - logp) - 1

per_token = surr - β * kl
\`\`\`

### Step 3: averaging (as in the DeepSeekMath objective)
\`\`\`
J = mean over (b, i) of [ (1 / |o_i|) * Σ_t mask_t * per_token_t ]
\`\`\`
Average over each response's **valid** tokens first (\`mask\` is 1 for real tokens and 0 for padding, |o_i| = mask.sum()), then average over all B·G responses. \`'mean_kl'\` uses the same averaging applied to \`kl\`. The training loss is \`-J\`.

### Shapes
- \`rewards\`: (B, G)
- \`logp\`, \`old_logp\`, \`ref_logp\`, \`mask\`: (B, G, T)

### Expected Return Format
\`group_advantages(rewards, eps=1e-8)\` returns the (B, G) array of advantages.

\`grpo_objective(...)\` returns a dictionary with these keys:
- \`'advantages'\`: (B, G) array from step 1
- \`'objective'\`: J as a Python float (to maximize)
- \`'loss'\`: \`-J\` as a Python float (to minimize)
- \`'mean_kl'\`: averaged k3 KL as a Python float
    `,
    examples: [
      {
        input: 'group_advantages([[1.0, 0.0, 0.0, 1.0]])',
        output: '[[1.0, -1.0, -1.0, 1.0]]',
        explanation: 'mean 0.5, std 0.5: correct answers get +1, wrong ones -1',
      },
      {
        input: 'logp == old_logp == ref_logp',
        output: "{'objective': 0.0, 'mean_kl': 0.0, ...}",
        explanation: 'ratio = 1 and KL = 0, so J is the mean advantage, which is 0 within every group',
      },
    ],
    starterCode: `import numpy as np

def group_advantages(rewards, eps=1e-8):
    """
    Group-normalized advantages.

    Args:
        rewards: (B, G) rewards, G sampled responses per prompt
        eps: added to the std

    Returns:
        (B, G) array: (r - group mean) / (group std + eps), population std
    """
    # Your code here
    pass


def grpo_objective(logp, old_logp, ref_logp, rewards, mask,
                   epsilon=0.2, beta=0.04, eps=1e-8):
    """
    GRPO clipped objective with a k3 KL penalty.

    Args:
        logp: (B, G, T) token log-probs under the current policy
        old_logp: (B, G, T) token log-probs under the policy that sampled
        ref_logp: (B, G, T) token log-probs under the frozen reference model
        rewards: (B, G) one scalar reward per response
        mask: (B, G, T) 1 for real tokens, 0 for padding
        epsilon: clipping range
        beta: KL coefficient
        eps: std epsilon for the advantages

    Returns:
        dict with 'advantages' (B, G), 'objective', 'loss', 'mean_kl' (floats)
    """
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Advantages for a 0/1 verifier reward',
        input: 'np.round(group_advantages([[1.0, 0.0, 0.0, 1.0]]), 4).tolist()',
        expected: '[[1.0, -1.0, -1.0, 1.0]]',
        hidden: false,
      },
      {
        id: '2',
        description: 'On-policy, no drift: objective 0 and KL 0',
        input: `(lambda: (
    lp := np.log(np.full((2, 3, 4), 0.25)),
    r := grpo_objective(lp, lp, lp, [[1.0, 0.0, 2.0], [0.5, 0.5, 1.0]], np.ones((2, 3, 4))),
    (round(r['objective'], 6), round(r['mean_kl'], 6))
)[-1])()`,
        expected: '(0.0, 0.0)',
        hidden: false,
      },
      {
        id: '3',
        description: 'Identical rewards give zero advantages (no NaN)',
        input: 'np.round(group_advantages([[0.5, 0.5, 0.5], [1.0, 2.0, 3.0]]), 6).tolist()',
        expected: '[[0.0, 0.0, 0.0], [-1.224745, 0.0, 1.224745]]',
        hidden: true,
      },
      {
        id: '4',
        description: 'Objective, loss and KL with padding (values)',
        input: `(lambda: (
    r := grpo_objective(
        [[[-1.0, -0.5, -2.0], [-0.3, -1.2, -0.7], [-0.9, -0.4, -1.5]]],
        [[[-1.1, -0.5, -1.8], [-0.4, -1.0, -0.7], [-0.9, -0.6, -1.5]]],
        [[[-1.2, -0.7, -1.9], [-0.3, -1.1, -0.9], [-1.0, -0.4, -1.4]]],
        [[1.0, 0.0, 0.5]],
        [[[1, 1, 1], [1, 1, 0], [1, 1, 1]]],
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
    r := grpo_objective([[[0.0], [0.0]]], [[[-1.0], [0.0]]], [[[0.0], [0.0]]], [[1.0, 0.0]], [[[1], [1]]], epsilon=0.2, beta=0.0),
    round(r['objective'], 4)
)[-1])()`,
        expected: '0.1',
        hidden: true,
      },
    ],
    hints: [
      'Advantages: subtract rewards.mean(axis=1, keepdims=True) and divide by rewards.std(axis=1, keepdims=True) + eps',
      'Broadcast the (B, G) advantages over tokens with adv[..., None]',
      'k3 KL: with d = ref_logp - logp, kl = np.exp(d) - d - 1 (it is >= 0 for every token)',
      'Per response: (per_token * mask).sum(-1) / mask.sum(-1); then .mean() over all responses',
    ],
    solution: `import numpy as np

def group_advantages(rewards, eps=1e-8):
    r = np.asarray(rewards, dtype=float)
    mean = r.mean(axis=1, keepdims=True)
    std = r.std(axis=1, keepdims=True)  # population std (ddof=0)
    return (r - mean) / (std + eps)


def grpo_objective(logp, old_logp, ref_logp, rewards, mask,
                   epsilon=0.2, beta=0.04, eps=1e-8):
    logp = np.asarray(logp, dtype=float)
    old_logp = np.asarray(old_logp, dtype=float)
    ref_logp = np.asarray(ref_logp, dtype=float)
    mask = np.asarray(mask, dtype=float)

    adv = group_advantages(rewards, eps)        # (B, G)
    A = adv[..., None]                          # same advantage for every token

    # PPO-style clipped surrogate per token
    ratio = np.exp(logp - old_logp)
    surr = np.minimum(ratio * A, np.clip(ratio, 1 - epsilon, 1 + epsilon) * A)

    # k3 KL estimator against the reference model
    d = ref_logp - logp
    kl = np.exp(d) - d - 1

    per_token = surr - beta * kl

    # Mean over each response's valid tokens, then over all responses
    lengths = mask.sum(axis=-1)
    objective = ((per_token * mask).sum(axis=-1) / lengths).mean()
    mean_kl = ((kl * mask).sum(axis=-1) / lengths).mean()

    return {
        'advantages': adv,
        'objective': float(objective),
        'loss': float(-objective),
        'mean_kl': float(mean_kl),
    }
`,
  },
  {
    id: 'int8-quantization',
    title: 'Int8 Weight Quantization (absmax)',
    section: 'post-training',
    difficulty: 'medium',
    description: `
## Symmetric Int8 Quantization

Storing weights in int8 instead of float32 cuts memory by 4×. **Absmax** (symmetric) quantization maps the largest magnitude to 127:
\`\`\`
scale = max|w| / 127
w_q   = clip(round(w / scale), -127, 127)     (stored as int8)
w_hat = w_q * scale                            (dequantized)
\`\`\`
Rounding is round-half-to-even (\`np.round\` / \`torch.round\`). The rounding error per weight is at most scale / 2.

### Per-tensor vs per-channel
- **Per-tensor**: one scale for the whole matrix. One large outlier makes the scale large, and every small weight loses precision.
- **Per-channel**: one scale per **output row** of W (shape (d_out, d_in)), so each row uses its own range. This is what LLM.int8() calls vector-wise quantization.

To avoid dividing by zero, use \`scale = max(absmax, 1e-8) / 127\`.

### Int8 matmul
To compute y = x Wᵀ in int8, also quantize the activations **per row** of x (per token), accumulate the integer products in int32 and rescale with both scales:
\`\`\`
x_q, s_x = absmax-quantize each row of x          (n, d_in), (n,)
acc      = x_q @ W_qᵀ   (int32 accumulation)       (n, d_out)
y        ≈ acc * s_x[:, None] * s_w[None, :]
\`\`\`

### What to implement
- \`quantize_per_channel(W)\` returns \`(W_q, scales)\`: W_q is an **int8** array (d_out, d_in), scales is (d_out,)
- \`quantize_per_tensor(W)\` returns \`(W_q, scale)\`: int8 array and a Python float
- \`dequantize(W_q, scales)\` returns floats; \`scales\` is either a (d_out,) array (one per row) or a scalar
- \`int8_matmul(x, W_q, w_scales)\` returns the (n, d_out) float approximation of x Wᵀ
- \`quantization_report(W)\` returns a dictionary:
  - \`'per_channel_error'\`: max |W - dequantize(per-channel)| (Python float)
  - \`'per_tensor_error'\`: max |W - dequantize(per-tensor)| (Python float)
    `,
    examples: [
      {
        input: 'W = [[0.7, -1.2, 0.3], [2.0, 0.1, -0.5]]',
        output: 'W_q = [[74, -127, 32], [127, 6, -32]], scales = [1.2/127, 2.0/127] = [0.009449, 0.015748]',
        explanation: 'Each row is scaled so that its largest magnitude maps to ±127',
      },
    ],
    starterCode: `import numpy as np

def quantize_per_channel(W):
    """Returns (W_q int8 (d_out, d_in), scales (d_out,)) with one scale per row."""
    # Your code here
    pass


def quantize_per_tensor(W):
    """Returns (W_q int8, scale float) with one scale for the whole matrix."""
    # Your code here
    pass


def dequantize(W_q, scales):
    """W_q * scale; scales is (d_out,) (per row) or a scalar."""
    # Your code here
    pass


def int8_matmul(x, W_q, w_scales):
    """
    Approximate x @ W.T with int8 operands.

    Args:
        x: (n, d_in) float activations (quantize per row)
        W_q: (d_out, d_in) int8 weights
        w_scales: (d_out,) per-row weight scales

    Returns:
        (n, d_out) float array
    """
    # Your code here
    pass


def quantization_report(W):
    """Returns {'per_channel_error': float, 'per_tensor_error': float}."""
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Per-channel int8 values',
        input: 'quantize_per_channel(np.array([[0.7, -1.2, 0.3], [2.0, 0.1, -0.5]]))[0].tolist()',
        expected: '[[74, -127, 32], [127, 6, -32]]',
        hidden: false,
      },
      {
        id: '2',
        description: 'Per-channel scales and int8 dtype',
        input: '(lambda q: (np.round(q[1], 6).tolist(), q[0].dtype == np.int8))(quantize_per_channel(np.array([[0.7, -1.2, 0.3], [2.0, 0.1, -0.5]])))',
        expected: '([0.009449, 0.015748], True)',
        hidden: false,
      },
      {
        id: '3',
        description: 'Per-tensor error is larger when rows have very different ranges',
        input: `(lambda r: (round(r['per_channel_error'], 6), round(r['per_tensor_error'], 6)))(quantization_report(np.array([[0.01, -0.02, 0.013], [5.0, -3.0, 1.7]])))`,
        expected: '(0.007874, 0.01937)',
        hidden: true,
      },
      {
        id: '4',
        description: 'Rounding error is at most scale / 2 for every weight',
        input: `(lambda: (
    W := np.random.randn(8, 16) * np.random.rand(8, 1) * 10,
    q := quantize_per_channel(W),
    bool(np.all(np.abs(dequantize(*q) - W) <= q[1][:, None] / 2 + 1e-12) and np.abs(q[0]).max() == 127)
)[-1])()`,
        expected: 'True',
        hidden: true,
      },
      {
        id: '5',
        description: 'Int8 matmul values (rescaled with activation and weight scales)',
        input: `(lambda: (
    W := np.array([[0.7, -1.2, 0.3], [2.0, 0.1, -0.5]]),
    x := np.array([[1.0, 2.0, -1.0], [0.5, -0.25, 4.0]]),
    np.round(int8_matmul(x, *quantize_per_channel(W)), 4).tolist()
)[-1])()`,
        expected: '[[-2.0, 2.7126], [1.8642, -1.0317]]',
        hidden: true,
      },
      {
        id: '6',
        description: 'Int8 matmul stays close to the float matmul',
        input: `(lambda: (
    W := np.random.randn(16, 32), x := np.random.randn(4, 32),
    y := int8_matmul(x, *quantize_per_channel(W)),
    bool(np.abs(y - x @ W.T).max() < 0.05 * np.abs(x @ W.T).max())
)[-1])()`,
        expected: 'True',
        hidden: true,
      },
    ],
    hints: [
      'Per-channel absmax: np.abs(W).max(axis=1); scales = np.maximum(absmax, 1e-8) / 127',
      'Quantize: np.clip(np.round(W / scales[:, None]), -127, 127).astype(np.int8)',
      'dequantize can reshape scales with np.reshape(scales, (-1, 1)) so it works for a scalar too',
      'int8_matmul: quantize x per row the same way, then x_q.astype(np.int32) @ W_q.astype(np.int32).T, times s_x[:, None] * w_scales[None, :]',
    ],
    solution: `import numpy as np

def _absmax_quantize_rows(M):
    absmax = np.abs(M).max(axis=1)
    scales = np.maximum(absmax, 1e-8) / 127.0
    M_q = np.clip(np.round(M / scales[:, None]), -127, 127).astype(np.int8)
    return M_q, scales


def quantize_per_channel(W):
    W = np.asarray(W, dtype=float)
    return _absmax_quantize_rows(W)


def quantize_per_tensor(W):
    W = np.asarray(W, dtype=float)
    scale = max(float(np.abs(W).max()), 1e-8) / 127.0
    W_q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    return W_q, scale


def dequantize(W_q, scales):
    return W_q.astype(float) * np.reshape(scales, (-1, 1))


def int8_matmul(x, W_q, w_scales):
    x = np.asarray(x, dtype=float)
    x_q, x_scales = _absmax_quantize_rows(x)           # per-token activation scales
    acc = x_q.astype(np.int32) @ W_q.astype(np.int32).T  # int32 accumulation
    return acc * x_scales[:, None] * np.asarray(w_scales)[None, :]


def quantization_report(W):
    W = np.asarray(W, dtype=float)
    per_channel = dequantize(*quantize_per_channel(W))
    per_tensor = dequantize(*quantize_per_tensor(W))
    return {
        'per_channel_error': float(np.abs(W - per_channel).max()),
        'per_tensor_error': float(np.abs(W - per_tensor).max()),
    }
`,
  },
];
