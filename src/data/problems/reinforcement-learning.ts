import { Problem } from '../../types';

export const reinforcementLearningProblems: Problem[] = [
  // ==========================================
  // BASIC PROBLEMS
  // ==========================================
  {
    id: 'rl-discounted-return',
    title: 'Discounted Return',
    section: 'reinforcement-learning',
    difficulty: 'easy',
    description: `
## Discounted Return (Cumulative Reward)

Calculate the discounted return (cumulative reward) from a sequence of rewards.

### Formula
\`\`\`
G_t = r_t + γ * r_{t+1} + γ² * r_{t+2} + ... + γ^(T-t) * r_T
    = Σ_{k=0}^{T-t} γ^k * r_{t+k}
\`\`\`

Where:
- \`r_t\`: Reward at timestep t
- \`γ\` (gamma): Discount factor (0 ≤ γ ≤ 1)
- \`T\`: Final timestep

### Why Discount?
- Future rewards are uncertain
- Prefer immediate rewards
- γ = 0: Only care about immediate reward
- γ = 1: All rewards equally important
- Typical values: 0.9 to 0.99

### Function Signature
\`\`\`python
def discounted_return(rewards: np.ndarray, gamma: float) -> float:
\`\`\`

Returns the total discounted return from timestep 0.
    `,
    examples: [
      {
        input: 'rewards = [1, 1, 1, 1], gamma = 0.9',
        output: '3.439',
        explanation: '1 + 0.9*1 + 0.81*1 + 0.729*1 = 3.439',
      },
      {
        input: 'rewards = [0, 0, 0, 10], gamma = 0.5',
        output: '1.25',
        explanation: '0 + 0 + 0 + 0.125*10 = 1.25',
      },
    ],
    starterCode: `import numpy as np

def discounted_return(rewards: np.ndarray, gamma: float) -> float:
    """
    Calculate the discounted cumulative return.

    Args:
        rewards: Array of rewards [r_0, r_1, ..., r_T]
        gamma: Discount factor (0 <= gamma <= 1)

    Returns:
        Total discounted return G_0
    """
    rewards = np.array(rewards)
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Constant rewards',
        input: '([1, 1, 1, 1], 0.9)',
        expected: '3.439',
        hidden: false,
      },
      {
        id: '2',
        description: 'Delayed reward',
        input: '([0, 0, 0, 10], 0.5)',
        expected: '1.25',
        hidden: false,
      },
      {
        id: '3',
        description: 'No discount',
        input: '([1, 2, 3, 4], 1.0)',
        expected: '10.0',
        hidden: false,
      },
      {
        id: '4',
        description: 'Only immediate',
        input: '([5, 10, 15], 0.0)',
        expected: '5.0',
        hidden: true,
      },
    ],
    hints: [
      'Create an array of discount factors: [γ⁰, γ¹, γ², ...]',
      'Use np.arange to create exponents, then gamma ** exponents',
      'Multiply rewards by discount factors and sum',
    ],
    solution: `import numpy as np

def discounted_return(rewards: np.ndarray, gamma: float) -> float:
    """
    Calculate the discounted cumulative return.
    """
    rewards = np.array(rewards)
    T = len(rewards)

    # Create discount factors: [gamma^0, gamma^1, gamma^2, ...]
    discounts = gamma ** np.arange(T)

    # Compute discounted return
    G = np.sum(rewards * discounts)

    return round(G, 3)
`,
  },
  {
    id: 'rl-epsilon-greedy',
    title: 'Epsilon-Greedy Policy',
    section: 'reinforcement-learning',
    difficulty: 'easy',
    description: `
## Epsilon-Greedy Policy

Implement the epsilon-greedy action selection policy, which balances exploration and exploitation.

### Algorithm
\`\`\`
With probability ε: choose random action (explore)
With probability 1-ε: choose action with highest Q-value (exploit)
\`\`\`

### Why Epsilon-Greedy?
- **Exploration**: Try new actions to discover better strategies
- **Exploitation**: Use known good actions to maximize reward
- ε typically decays over time (start exploring, then exploit)

### Function Signature
\`\`\`python
def epsilon_greedy(q_values: np.ndarray, epsilon: float,
                   random_value: float) -> int:
\`\`\`

**Note**: We pass \`random_value\` (a number between 0-1) instead of generating it,
so tests are deterministic. In practice, you'd use \`np.random.random()\`.

### Expected Return
- If \`random_value < epsilon\`: return \`random_action\` (passed as floor of random_value * n_actions for testing)
- Else: return \`argmax(q_values)\`
    `,
    examples: [
      {
        input: 'q_values = [1.0, 3.0, 2.0], epsilon = 0.1, random_value = 0.5',
        output: '1',
        explanation: '0.5 >= 0.1, so exploit: argmax([1,3,2]) = 1',
      },
      {
        input: 'q_values = [1.0, 3.0, 2.0], epsilon = 0.9, random_value = 0.5',
        output: '1',
        explanation: '0.5 < 0.9, so explore: floor(0.5 * 3) = 1',
      },
    ],
    starterCode: `import numpy as np

def epsilon_greedy(q_values: np.ndarray, epsilon: float,
                   random_value: float) -> int:
    """
    Select action using epsilon-greedy policy.

    Args:
        q_values: Q-values for each action [Q(a_0), Q(a_1), ...]
        epsilon: Exploration probability (0 <= epsilon <= 1)
        random_value: Pre-generated random number for deterministic testing

    Returns:
        Selected action index
    """
    q_values = np.array(q_values)
    n_actions = len(q_values)

    # Your code here
    # If random_value < epsilon: explore (return floor(random_value * n_actions))
    # Else: exploit (return argmax of q_values)
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Exploit (greedy)',
        input: '([1.0, 3.0, 2.0], 0.1, 0.5)',
        expected: '1',
        hidden: false,
      },
      {
        id: '2',
        description: 'Explore',
        input: '([1.0, 3.0, 2.0], 0.9, 0.5)',
        expected: '1',
        hidden: false,
      },
      {
        id: '3',
        description: 'Always exploit (epsilon=0)',
        input: '([5.0, 2.0, 8.0, 1.0], 0.0, 0.99)',
        expected: '2',
        hidden: false,
      },
      {
        id: '4',
        description: 'Explore with low random',
        input: '([1.0, 2.0, 3.0, 4.0], 0.5, 0.2)',
        expected: '0',
        hidden: true,
      },
    ],
    hints: [
      'Compare random_value to epsilon to decide explore vs exploit',
      'For exploration, compute floor(random_value * n_actions)',
      'For exploitation, use np.argmax(q_values)',
    ],
    solution: `import numpy as np

def epsilon_greedy(q_values: np.ndarray, epsilon: float,
                   random_value: float) -> int:
    """
    Select action using epsilon-greedy policy.
    """
    q_values = np.array(q_values)
    n_actions = len(q_values)

    if random_value < epsilon:
        # Explore: random action
        return int(random_value * n_actions)
    else:
        # Exploit: greedy action
        return int(np.argmax(q_values))
`,
  },
  {
    id: 'rl-bellman-value',
    title: 'Bellman Expectation (State Value)',
    section: 'reinforcement-learning',
    difficulty: 'easy',
    description: `
## Bellman Expectation Equation for State Value

Compute the value of a state given transition probabilities, rewards, and next-state values.

### Bellman Expectation Equation
\`\`\`
V(s) = Σ_a π(a|s) * Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * V(s')]
\`\`\`

For a fixed policy, this simplifies to:
\`\`\`
V(s) = Σ_{s'} P(s'|s) * [R(s,s') + γ * V(s')]
\`\`\`

### What It Means
The value of state s equals the expected:
- Immediate reward R
- Plus discounted value of next state V(s')

### Function Signature
\`\`\`python
def bellman_value(transitions: list, gamma: float,
                  next_values: np.ndarray) -> float:
\`\`\`

Where each transition is \`(probability, reward, next_state_index)\`.
    `,
    examples: [
      {
        input: 'transitions=[(0.5, 1, 0), (0.5, 2, 1)], gamma=0.9, V=[5, 10]',
        output: '8.25',
        explanation: '0.5*(1 + 0.9*5) + 0.5*(2 + 0.9*10) = 0.5*5.5 + 0.5*11 = 8.25',
      },
    ],
    starterCode: `import numpy as np

def bellman_value(transitions: list, gamma: float,
                  next_values: np.ndarray) -> float:
    """
    Compute state value using Bellman expectation equation.

    Args:
        transitions: List of (probability, reward, next_state_index) tuples
        gamma: Discount factor
        next_values: Array of values V(s') for each state

    Returns:
        Value V(s) for the current state
    """
    next_values = np.array(next_values)
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Two transitions',
        input: 'bellman_value([(0.5, 1, 0), (0.5, 2, 1)], 0.9, [5, 10])',
        expected: '8.25',
        hidden: false,
      },
      {
        id: '2',
        description: 'Deterministic transition',
        input: 'bellman_value([(1.0, 5, 0)], 0.9, [10])',
        expected: '14.0',
        hidden: false,
      },
      {
        id: '3',
        description: 'Three outcomes',
        input: 'bellman_value([(0.2, 0, 0), (0.5, 1, 1), (0.3, 2, 2)], 0.9, [0, 5, 10])',
        expected: '6.05',
        hidden: true,
      },
    ],
    hints: [
      'Loop through each transition (p, r, s_next)',
      'For each: add p * (r + gamma * V[s_next]) to the total',
      'This is the expected value over all possible transitions',
    ],
    solution: `import numpy as np

def bellman_value(transitions: list, gamma: float,
                  next_values: np.ndarray) -> float:
    """
    Compute state value using Bellman expectation equation.
    """
    next_values = np.array(next_values)

    value = 0.0
    for prob, reward, next_state in transitions:
        # V(s) = Σ P(s'|s) * [R + γ * V(s')]
        value += prob * (reward + gamma * next_values[next_state])

    return round(value, 2)
`,
  },
  // ==========================================
  // MEDIUM PROBLEMS
  // ==========================================
  {
    id: 'rl-q-learning-update',
    title: 'Q-Learning Update',
    section: 'reinforcement-learning',
    difficulty: 'medium',
    description: `
## Q-Learning Update Rule

Implement the Q-learning update rule, the core of this model-free RL algorithm.

### Q-Learning Update
\`\`\`
Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
\`\`\`

Where:
- \`α\`: Learning rate
- \`r\`: Reward received
- \`γ\`: Discount factor
- \`s'\`: Next state
- \`max_a' Q(s', a')\`: Best Q-value in next state

### Key Properties
- **Off-policy**: Updates use max Q-value, not the action actually taken
- **Model-free**: Doesn't need environment dynamics
- **Converges**: To optimal Q* under certain conditions

### Function Signature
\`\`\`python
def q_learning_update(Q: np.ndarray, state: int, action: int,
                      reward: float, next_state: int,
                      alpha: float, gamma: float) -> np.ndarray:
\`\`\`

Returns the updated Q-table.
    `,
    examples: [
      {
        input: 'Q=[[0,0],[0,0]], s=0, a=1, r=1, s\'=1, α=0.1, γ=0.9',
        output: 'Q[0,1] = 0.1',
        explanation: 'Q[0,1] += 0.1 * (1 + 0.9*0 - 0) = 0.1',
      },
    ],
    starterCode: `import numpy as np

def q_learning_update(Q: np.ndarray, state: int, action: int,
                      reward: float, next_state: int,
                      alpha: float, gamma: float) -> np.ndarray:
    """
    Perform one Q-learning update.

    Args:
        Q: Q-table of shape (n_states, n_actions)
        state: Current state index
        action: Action taken
        reward: Reward received
        next_state: Next state index
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        Updated Q-table
    """
    Q = np.array(Q, dtype=float)
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Simple update',
        input: '([[0, 0], [0, 0]], 0, 1, 1.0, 1, 0.1, 0.9)',
        expected: '[[0.0, 0.1], [0.0, 0.0]]',
        hidden: false,
      },
      {
        id: '2',
        description: 'With existing values',
        input: '([[1, 2], [3, 4]], 0, 0, 5.0, 1, 0.5, 0.9)',
        expected: '[[4.8, 2.0], [3.0, 4.0]]',
        hidden: false,
      },
      {
        id: '3',
        description: 'Terminal state (no future)',
        input: '([[0, 0], [0, 0]], 0, 0, 10.0, 1, 1.0, 0.0)',
        expected: '[[10.0, 0.0], [0.0, 0.0]]',
        hidden: true,
      },
    ],
    hints: [
      'Find the maximum Q-value in the next state: np.max(Q[next_state])',
      'Compute TD target: reward + gamma * max_next_Q',
      'Compute TD error: target - Q[state, action]',
      'Update: Q[state, action] += alpha * td_error',
    ],
    solution: `import numpy as np

def q_learning_update(Q: np.ndarray, state: int, action: int,
                      reward: float, next_state: int,
                      alpha: float, gamma: float) -> np.ndarray:
    """
    Perform one Q-learning update.
    """
    Q = np.array(Q, dtype=float)

    # Best Q-value in next state
    max_next_Q = np.max(Q[next_state])

    # TD target (Bellman optimality target)
    td_target = reward + gamma * max_next_Q

    # TD error
    td_error = td_target - Q[state, action]

    # Update Q-value
    Q[state, action] += alpha * td_error

    return np.round(Q, 1).tolist()
`,
  },
  {
    id: 'rl-sarsa-update',
    title: 'SARSA Update',
    section: 'reinforcement-learning',
    difficulty: 'medium',
    description: `
## SARSA Update Rule

Implement SARSA (State-Action-Reward-State-Action), an on-policy TD control algorithm.

### SARSA Update
\`\`\`
Q(s, a) ← Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]
\`\`\`

### SARSA vs Q-Learning
| Aspect | SARSA | Q-Learning |
|--------|-------|------------|
| Policy | On-policy | Off-policy |
| Next Q | Q(s', a') actual | max_a' Q(s', a') |
| Safety | More conservative | Can be risky |

### On-Policy Meaning
SARSA uses the **actual next action** a' that will be taken (following the current policy),
while Q-learning uses the **best possible** next action.

### Function Signature
\`\`\`python
def sarsa_update(Q: np.ndarray, state: int, action: int, reward: float,
                 next_state: int, next_action: int,
                 alpha: float, gamma: float) -> np.ndarray:
\`\`\`
    `,
    examples: [
      {
        input: 'Q=[[1,2],[3,4]], s=0, a=0, r=5, s\'=1, a\'=0, α=0.5, γ=0.9',
        output: 'Q[0,0] = 4.35',
        explanation: 'Q[0,0] += 0.5*(5 + 0.9*3 - 1) = 0.5*6.7 = 3.35, so 1+3.35=4.35',
      },
    ],
    starterCode: `import numpy as np

def sarsa_update(Q: np.ndarray, state: int, action: int, reward: float,
                 next_state: int, next_action: int,
                 alpha: float, gamma: float) -> np.ndarray:
    """
    Perform one SARSA update.

    Args:
        Q: Q-table of shape (n_states, n_actions)
        state: Current state s
        action: Action taken a
        reward: Reward received r
        next_state: Next state s'
        next_action: Next action a' (will be taken)
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        Updated Q-table
    """
    Q = np.array(Q, dtype=float)
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Basic SARSA update',
        input: '([[1, 2], [3, 4]], 0, 0, 5.0, 1, 0, 0.5, 0.9)',
        expected: '[[4.35, 2.0], [3.0, 4.0]]',
        hidden: false,
      },
      {
        id: '2',
        description: 'Different next action',
        input: '([[1, 2], [3, 4]], 0, 0, 5.0, 1, 1, 0.5, 0.9)',
        expected: '[[4.8, 2.0], [3.0, 4.0]]',
        hidden: false,
      },
      {
        id: '3',
        description: 'From zero',
        input: '([[0, 0], [0, 0]], 0, 1, 1.0, 1, 0, 0.1, 0.9)',
        expected: '[[0.0, 0.1], [0.0, 0.0]]',
        hidden: true,
      },
    ],
    hints: [
      'The key difference from Q-learning: use Q[next_state, next_action] not max',
      'TD target: reward + gamma * Q[next_state, next_action]',
      'TD error: target - Q[state, action]',
      'Update: Q[state, action] += alpha * td_error',
    ],
    solution: `import numpy as np

def sarsa_update(Q: np.ndarray, state: int, action: int, reward: float,
                 next_state: int, next_action: int,
                 alpha: float, gamma: float) -> np.ndarray:
    """
    Perform one SARSA update.
    """
    Q = np.array(Q, dtype=float)

    # SARSA uses actual next action, not max
    next_Q = Q[next_state, next_action]

    # TD target
    td_target = reward + gamma * next_Q

    # TD error
    td_error = td_target - Q[state, action]

    # Update Q-value
    Q[state, action] += alpha * td_error

    return np.round(Q, 2).tolist()
`,
  },
  {
    id: 'rl-td0-prediction',
    title: 'TD(0) Value Prediction',
    section: 'reinforcement-learning',
    difficulty: 'medium',
    description: `
## TD(0) Value Prediction

Implement TD(0), the simplest temporal difference learning algorithm for value estimation.

### TD(0) Update
\`\`\`
V(s) ← V(s) + α * [r + γ * V(s') - V(s)]
\`\`\`

### Key Concepts
- **TD Error (δ)**: \`r + γ * V(s') - V(s)\`
- **Bootstrap**: Uses estimated V(s') rather than waiting for episode end
- **Online**: Updates after each step, not at episode end

### TD vs Monte Carlo
| Aspect | TD(0) | Monte Carlo |
|--------|-------|-------------|
| Update | Every step | End of episode |
| Bias | Some bias | Unbiased |
| Variance | Low variance | High variance |
| Bootstrap | Yes | No |

### Function Signature
\`\`\`python
def td0_update(V: np.ndarray, state: int, reward: float,
               next_state: int, alpha: float, gamma: float,
               done: bool = False) -> np.ndarray:
\`\`\`

If \`done=True\`, the next state is terminal (V(s')=0).
    `,
    examples: [
      {
        input: 'V=[0,0,0], s=0, r=1, s\'=1, α=0.1, γ=0.9, done=False',
        output: 'V=[0.1, 0, 0]',
        explanation: 'V[0] += 0.1 * (1 + 0.9*0 - 0) = 0.1',
      },
    ],
    starterCode: `import numpy as np

def td0_update(V: np.ndarray, state: int, reward: float,
               next_state: int, alpha: float, gamma: float,
               done: bool = False) -> np.ndarray:
    """
    Perform one TD(0) value update.

    Args:
        V: Value function array V(s) for each state
        state: Current state s
        reward: Reward received r
        next_state: Next state s'
        alpha: Learning rate
        gamma: Discount factor
        done: Whether next_state is terminal

    Returns:
        Updated value function
    """
    V = np.array(V, dtype=float)
    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Non-terminal transition',
        input: '([0, 0, 0], 0, 1.0, 1, 0.1, 0.9, False)',
        expected: '[0.1, 0.0, 0.0]',
        hidden: false,
      },
      {
        id: '2',
        description: 'Terminal state',
        input: '([0, 0, 0], 0, 10.0, 1, 0.5, 0.9, True)',
        expected: '[5.0, 0.0, 0.0]',
        hidden: false,
      },
      {
        id: '3',
        description: 'With existing values',
        input: '([5, 10, 15], 1, 2.0, 2, 0.1, 0.9, False)',
        expected: '[5.0, 10.55, 15.0]',
        hidden: true,
      },
    ],
    hints: [
      'If done=True, the next state value is 0 (terminal)',
      'TD target: reward + gamma * V[next_state] (or just reward if done)',
      'TD error: target - V[state]',
      'Update: V[state] += alpha * td_error',
    ],
    solution: `import numpy as np

def td0_update(V: np.ndarray, state: int, reward: float,
               next_state: int, alpha: float, gamma: float,
               done: bool = False) -> np.ndarray:
    """
    Perform one TD(0) value update.
    """
    V = np.array(V, dtype=float)

    # Next state value (0 if terminal)
    next_value = 0.0 if done else V[next_state]

    # TD target
    td_target = reward + gamma * next_value

    # TD error
    td_error = td_target - V[state]

    # Update value
    V[state] += alpha * td_error

    return np.round(V, 2).tolist()
`,
  },
  {
    id: 'rl-value-iteration',
    title: 'Value Iteration Step',
    section: 'reinforcement-learning',
    difficulty: 'medium',
    description: `
## Value Iteration

Implement one step of the value iteration algorithm for solving MDPs.

### Bellman Optimality Update
\`\`\`
V(s) = max_a Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * V(s')]
\`\`\`

For each state, we find the action that maximizes expected value.

### Algorithm
1. For each state s:
   - For each action a, compute Q(s,a)
   - V(s) = max over all Q(s,a)
2. Repeat until convergence

### Function Signature
\`\`\`python
def value_iteration_step(V: np.ndarray, transitions: dict,
                         gamma: float) -> np.ndarray:
\`\`\`

Where \`transitions[s][a]\` is a list of \`(prob, reward, next_state)\` tuples.
    `,
    examples: [
      {
        input: 'V=[0,0], transitions for 2-state MDP, gamma=0.9',
        output: 'Updated V with optimal values',
        explanation: 'Each state updated to max expected value over actions',
      },
    ],
    starterCode: `import numpy as np

def value_iteration_step(V: np.ndarray, transitions: dict,
                         gamma: float) -> np.ndarray:
    """
    Perform one step of value iteration.

    Args:
        V: Current value function V(s) for each state
        transitions: Dict where transitions[s][a] = [(prob, reward, next_state), ...]
        gamma: Discount factor

    Returns:
        Updated value function
    """
    V = np.array(V, dtype=float)
    n_states = len(V)
    V_new = np.zeros(n_states)

    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Simple 2-state MDP',
        input: `([0.0, 0.0], {0: {0: [(1.0, 1.0, 0)], 1: [(1.0, 5.0, 1)]}, 1: {0: [(1.0, 2.0, 0)], 1: [(1.0, 0.0, 1)]}}, 0.9)`,
        expected: '[5.0, 2.0]',
        hidden: false,
      },
      {
        id: '2',
        description: 'With existing values',
        input: `([5.0, 2.0], {0: {0: [(1.0, 1.0, 0)], 1: [(1.0, 5.0, 1)]}, 1: {0: [(1.0, 2.0, 0)], 1: [(1.0, 0.0, 1)]}}, 0.9)`,
        expected: '[6.8, 6.5]',
        hidden: false,
      },
      {
        id: '3',
        description: 'Discounting with a self-loop',
        input: '([1.0, 0.0], {0: {0: [(1.0, 0.0, 0)], 1: [(0.5, 2.0, 1), (0.5, 0.0, 0)]}, 1: {0: [(1.0, 1.0, 1)]}}, 0.5)',
        expected: '[1.2, 1.0]',
        hidden: true,
      },
    ],
    hints: [
      'For each state, compute the value of each action',
      'Action value: sum over transitions of prob * (reward + gamma * V[next])',
      'State value: max over all action values',
      'Use the new values, not the old ones being updated',
    ],
    solution: `import numpy as np

def value_iteration_step(V: np.ndarray, transitions: dict,
                         gamma: float) -> np.ndarray:
    """
    Perform one step of value iteration.
    """
    V = np.array(V, dtype=float)
    n_states = len(V)
    V_new = np.zeros(n_states)

    for s in range(n_states):
        if s not in transitions:
            continue

        # Compute Q-value for each action
        q_values = []
        for a in transitions[s]:
            q_sa = 0.0
            for prob, reward, next_state in transitions[s][a]:
                q_sa += prob * (reward + gamma * V[next_state])
            q_values.append(q_sa)

        # V(s) = max Q(s, a)
        V_new[s] = max(q_values) if q_values else 0.0

    return np.round(V_new, 1).tolist()
`,
  },
  // ==========================================
  // ADVANCED PROBLEMS
  // ==========================================
  {
    id: 'rl-policy-gradient',
    title: 'REINFORCE Policy Gradient',
    section: 'reinforcement-learning',
    difficulty: 'hard',
    description: `
## REINFORCE (Monte Carlo Policy Gradient)

Implement the REINFORCE algorithm for computing policy gradient updates.

### Policy Gradient Theorem
\`\`\`
∇J(θ) = E[Σ_t ∇log π(a_t|s_t; θ) * G_t]
\`\`\`

### REINFORCE Algorithm
1. Collect episode: (s_0, a_0, r_1, s_1, a_1, r_2, ...)
2. For each timestep t:
   - Compute return G_t (discounted sum of future rewards)
   - Compute gradient: ∇log π(a_t|s_t) * G_t
3. Update policy parameters

### Softmax Policy
\`\`\`
π(a|s) = exp(θ[s,a]) / Σ_a' exp(θ[s,a'])
\`\`\`

### Gradient of Log-Softmax
\`\`\`
∇_θ log π(a|s) = one_hot(a) - π(·|s)
\`\`\`
(1 for action taken, minus probability of each action)

### Function Signature
\`\`\`python
def reinforce_gradient(theta: np.ndarray, episode: list,
                       gamma: float) -> np.ndarray:
\`\`\`

Episode is list of (state, action, reward) tuples.

### Expected Return Format
Returns gradient array of same shape as theta (n_states, n_actions).
    `,
    examples: [
      {
        input: 'theta (3x2), episode=[(0,1,1), (1,0,2)], gamma=0.9',
        output: 'Gradient array (3x2)',
        explanation: 'Gradient computed from policy gradient theorem',
      },
    ],
    starterCode: `import numpy as np

def softmax(x):
    """Compute softmax probabilities."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def reinforce_gradient(theta: np.ndarray, episode: list,
                       gamma: float) -> np.ndarray:
    """
    Compute REINFORCE policy gradient.

    Args:
        theta: Policy parameters (n_states, n_actions)
        episode: List of (state, action, reward) tuples
        gamma: Discount factor

    Returns:
        Gradient of same shape as theta
    """
    theta = np.array(theta, dtype=float)
    n_states, n_actions = theta.shape
    gradient = np.zeros_like(theta)

    # Your code here
    # 1. Compute returns G_t for each timestep
    # 2. For each (s, a, r), compute gradient contribution
    # 3. Gradient = (one_hot(a) - softmax(theta[s])) * G_t
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Simple episode',
        input: 'bool(np.allclose(reinforce_gradient(np.zeros((2, 2)), [(0, 1, 1.0), (1, 0, 1.0)], 0.9), [[-0.95, 0.95], [0.5, -0.5]], atol=0.01))',
        expected: 'True',
        hidden: false,
      },
      {
        id: '2',
        description: 'Single step episode',
        input: 'bool(np.allclose(reinforce_gradient(np.zeros((2, 2)), [(0, 0, 5.0)], 0.9), [[2.5, -2.5], [0.0, 0.0]], atol=0.01))',
        expected: 'True',
        hidden: false,
      },
      {
        id: '3',
        description: 'Gradient shape and direction for a single step',
        input: '(lambda g: (g.shape, bool(g[1, 1] > 0 and g[1, 0] < 0)))(reinforce_gradient(np.zeros((2, 2)), [(1, 1, 2.0)], 0.9))',
        expected: '((2, 2), True)',
        hidden: true,
      },
    ],
    hints: [
      'First compute all returns: G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...',
      'For softmax policy: gradient = one_hot(action) - π(·|state)',
      'Multiply gradient by return G_t and accumulate',
      'Use backward iteration to compute returns efficiently',
    ],
    solution: `import numpy as np

def softmax(x):
    """Compute softmax probabilities."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def reinforce_gradient(theta: np.ndarray, episode: list,
                       gamma: float) -> np.ndarray:
    """
    Compute REINFORCE policy gradient.
    """
    theta = np.array(theta, dtype=float)
    n_states, n_actions = theta.shape
    gradient = np.zeros_like(theta)

    T = len(episode)
    if T == 0:
        return gradient

    # Compute returns for each timestep (backward)
    returns = np.zeros(T)
    G = 0
    for t in range(T - 1, -1, -1):
        _, _, reward = episode[t]
        G = reward + gamma * G
        returns[t] = G

    # Compute gradient
    for t in range(T):
        state, action, _ = episode[t]
        G_t = returns[t]

        # Policy probabilities for this state
        probs = softmax(theta[state])

        # Gradient of log π(a|s) = one_hot(a) - π(·|s)
        grad_log_pi = -probs.copy()
        grad_log_pi[action] += 1.0

        # Accumulate: ∇log π(a|s) * G_t
        gradient[state] += grad_log_pi * G_t

    return np.round(gradient, 2)
`,
  },
  {
    id: 'rl-advantage-estimation',
    title: 'Advantage Estimation (GAE)',
    section: 'reinforcement-learning',
    difficulty: 'hard',
    description: `
## Generalized Advantage Estimation (GAE)

Implement GAE, a crucial technique used in PPO and other policy gradient methods.

### Why Advantage?
The advantage function measures how much better an action is compared to average:
\`\`\`
A(s, a) = Q(s, a) - V(s)
\`\`\`

### GAE Formula
\`\`\`
Â_t = Σ_{l=0}^{T-t} (γλ)^l * δ_{t+l}
\`\`\`

Where δ_t is the TD error:
\`\`\`
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
\`\`\`

### Lambda (λ) Trade-off
- λ = 0: Uses only one-step TD (low variance, high bias)
- λ = 1: Uses full Monte Carlo return (high variance, low bias)
- Typical: λ = 0.95

### Function Signature
\`\`\`python
def compute_gae(rewards: np.ndarray, values: np.ndarray,
                gamma: float, lam: float) -> np.ndarray:
\`\`\`

Where values has length T+1 (includes V(s_T+1), use 0 if terminal).
    `,
    examples: [
      {
        input: 'rewards=[1,1,1], values=[0,0,0,0], gamma=0.99, lambda=0.95',
        output: 'advantages=[2.83, 1.94, 1.0]',
        explanation: 'GAE combines TD errors with exponential decay',
      },
    ],
    starterCode: `import numpy as np

def compute_gae(rewards: np.ndarray, values: np.ndarray,
                gamma: float, lam: float) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: Rewards r_0, r_1, ..., r_{T-1} (length T)
        values: Values V(s_0), V(s_1), ..., V(s_T) (length T+1)
                Last value is V(s_T) or 0 if terminal
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        Advantages Â_0, Â_1, ..., Â_{T-1} (length T)
    """
    rewards = np.array(rewards, dtype=float)
    values = np.array(values, dtype=float)
    T = len(rewards)
    advantages = np.zeros(T)

    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'Zero values baseline',
        input: 'bool(np.allclose(compute_gae([1, 1, 1], [0, 0, 0, 0], 0.99, 0.95), [2.825, 1.9405, 1.0], atol=0.01))',
        expected: 'True',
        hidden: false,
      },
      {
        id: '2',
        description: 'With value estimates',
        input: 'bool(np.allclose(compute_gae([1, 2, 3], [1, 2, 3, 0], 0.9, 0.9), [3.99, 2.7, 0.0], atol=0.1))',
        expected: 'True',
        hidden: false,
      },
      {
        id: '3',
        description: 'Single step',
        input: 'bool(np.allclose(compute_gae([5], [1, 0], 0.99, 0.95), [4.0], atol=0.01))',
        expected: 'True',
        hidden: true,
      },
    ],
    hints: [
      'Compute TD errors first: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)',
      'Work backwards from the last timestep',
      'GAE formula: Â_t = δ_t + γλ*Â_{t+1}',
      'At the last timestep: Â_{T-1} = δ_{T-1}',
    ],
    solution: `import numpy as np

def compute_gae(rewards: np.ndarray, values: np.ndarray,
                gamma: float, lam: float) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation.
    """
    rewards = np.array(rewards, dtype=float)
    values = np.array(values, dtype=float)
    T = len(rewards)
    advantages = np.zeros(T)

    # Compute advantages backwards
    gae = 0.0
    for t in range(T - 1, -1, -1):
        # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * values[t + 1] - values[t]

        # GAE: Â_t = δ_t + γλ*Â_{t+1}
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    return np.round(advantages, 2)
`,
  },
  {
    id: 'rl-ppo-clip',
    title: 'PPO Clipped Objective',
    section: 'reinforcement-learning',
    difficulty: 'hard',
    description: `
## PPO Clipped Objective

Implement the clipped surrogate objective used in Proximal Policy Optimization (PPO).

### PPO Objective
\`\`\`
L^{CLIP}(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
\`\`\`

Where:
- r_t(θ) = π_θ(a|s) / π_θ_old(a|s) is the probability ratio
- A_t is the advantage estimate
- ε is the clipping parameter (typically 0.2)

### Why Clipping?
- Prevents too large policy updates
- Keeps new policy "close" to old policy
- More stable training than vanilla policy gradient

### Key Insight
- If A > 0 (good action): clip ratio to ≤ 1+ε
- If A < 0 (bad action): clip ratio to ≥ 1-ε

### Function Signature
\`\`\`python
def ppo_clip_objective(old_probs: np.ndarray, new_probs: np.ndarray,
                       advantages: np.ndarray, epsilon: float) -> float:
\`\`\`
    `,
    examples: [
      {
        input: 'old_probs=[0.5], new_probs=[0.6], advantages=[1.0], epsilon=0.2',
        output: '1.2',
        explanation: 'ratio = 1.2 is inside [0.8, 1.2], so min(1.2 * 1, clip(1.2) * 1) = 1.2',
      },
    ],
    starterCode: `import numpy as np

def ppo_clip_objective(old_probs: np.ndarray, new_probs: np.ndarray,
                       advantages: np.ndarray, epsilon: float) -> float:
    """
    Compute PPO clipped surrogate objective.

    Args:
        old_probs: π_old(a|s) probabilities for taken actions
        new_probs: π_new(a|s) probabilities for taken actions
        advantages: Advantage estimates A_t
        epsilon: Clipping parameter (e.g., 0.2)

    Returns:
        PPO clipped objective (scalar, should be maximized)
    """
    old_probs = np.array(old_probs, dtype=float)
    new_probs = np.array(new_probs, dtype=float)
    advantages = np.array(advantages, dtype=float)

    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: 'No clipping needed',
        input: '([0.5, 0.5], [0.5, 0.5], [1.0, -1.0], 0.2)',
        expected: '0.0',
        hidden: false,
      },
      {
        id: '2',
        description: 'Positive advantage, ratio > 1',
        input: '([0.5], [0.7], [1.0], 0.2)',
        expected: '1.2',
        hidden: false,
      },
      {
        id: '3',
        description: 'Negative advantage, ratio > 1',
        input: '([0.5], [0.7], [-1.0], 0.2)',
        expected: '-1.4',
        hidden: false,
      },
      {
        id: '4',
        description: 'Clipping in action',
        input: '([0.3], [0.6], [2.0], 0.2)',
        expected: '2.4',
        hidden: true,
      },
    ],
    hints: [
      'Compute probability ratios: r = new_probs / old_probs',
      'Compute clipped ratios: clip(r, 1-epsilon, 1+epsilon)',
      'Unclipped term: r * advantages',
      'Clipped term: clipped_r * advantages',
      'Take element-wise minimum, then mean',
    ],
    solution: `import numpy as np

def ppo_clip_objective(old_probs: np.ndarray, new_probs: np.ndarray,
                       advantages: np.ndarray, epsilon: float) -> float:
    """
    Compute PPO clipped surrogate objective.
    """
    old_probs = np.array(old_probs, dtype=float)
    new_probs = np.array(new_probs, dtype=float)
    advantages = np.array(advantages, dtype=float)

    # Probability ratios
    ratios = new_probs / (old_probs + 1e-8)

    # Clipped ratios
    clipped_ratios = np.clip(ratios, 1 - epsilon, 1 + epsilon)

    # Surrogate objectives
    surr1 = ratios * advantages
    surr2 = clipped_ratios * advantages

    # PPO objective: min of clipped and unclipped
    objective = np.mean(np.minimum(surr1, surr2))

    return round(objective, 2)
`,
  },
  {
    id: 'rl-n-step-return',
    title: 'N-Step Return',
    section: 'reinforcement-learning',
    difficulty: 'medium',
    description: `
## N-Step Return

Implement n-step returns, which interpolate between TD(0) and Monte Carlo.

### N-Step Return Formula
\`\`\`
G_t^{(n)} = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n * V(s_{t+n})
\`\`\`

### Trade-offs
| n | Method | Bias | Variance |
|---|--------|------|----------|
| 1 | TD(0) | High | Low |
| n | n-step | Medium | Medium |
| ∞ | MC | None | High |

### Function Signature
\`\`\`python
def n_step_return(rewards: np.ndarray, values: np.ndarray,
                  t: int, n: int, gamma: float) -> float:
\`\`\`

Compute n-step return starting from timestep t.
    `,
    examples: [
      {
        input: 'rewards=[1,2,3,4], values=[0,0,0,0,10], t=0, n=2, gamma=0.9',
        output: '2.8',
        explanation: 'G_0^(2) = 1 + 0.9*2 + 0.81*0 = 2.8',
      },
    ],
    starterCode: `import numpy as np

def n_step_return(rewards: np.ndarray, values: np.ndarray,
                  t: int, n: int, gamma: float) -> float:
    """
    Compute n-step return from timestep t.

    Args:
        rewards: All rewards [r_0, r_1, ..., r_{T-1}]
        values: All values [V(s_0), ..., V(s_T)]
        t: Starting timestep
        n: Number of steps
        gamma: Discount factor

    Returns:
        N-step return G_t^(n)
    """
    rewards = np.array(rewards, dtype=float)
    values = np.array(values, dtype=float)
    T = len(rewards)

    # Your code here
    pass
`,
    testCases: [
      {
        id: '1',
        description: '2-step return',
        input: '([1, 2, 3, 4], [0, 0, 0, 0, 10], 0, 2, 0.9)',
        expected: '2.8',
        hidden: false,
      },
      {
        id: '2',
        description: '3-step return with bootstrap',
        input: '([1, 1, 1, 1], [0, 0, 0, 5, 0], 0, 3, 0.9)',
        expected: '6.36',
        hidden: false,
      },
      {
        id: '3',
        description: '1-step (TD) return',
        input: '([5, 2, 3], [1, 10, 5, 0], 0, 1, 0.9)',
        expected: '14.0',
        hidden: true,
      },
    ],
    hints: [
      'Sum discounted rewards from t to min(t+n-1, T-1)',
      'Bootstrap with V(s_{t+n}) if t+n <= T',
      'Handle boundary: if t+n > T, just sum remaining rewards',
      'Discounts: γ^0, γ^1, ..., γ^{n-1} for rewards, γ^n for value',
    ],
    solution: `import numpy as np

def n_step_return(rewards: np.ndarray, values: np.ndarray,
                  t: int, n: int, gamma: float) -> float:
    """
    Compute n-step return from timestep t.
    """
    rewards = np.array(rewards, dtype=float)
    values = np.array(values, dtype=float)
    T = len(rewards)

    G = 0.0

    # Sum discounted rewards
    for k in range(n):
        if t + k < T:
            G += (gamma ** k) * rewards[t + k]

    # Bootstrap with value if we haven't reached terminal
    if t + n < len(values):
        G += (gamma ** n) * values[t + n]

    return round(G, 2)
`,
  },
];
