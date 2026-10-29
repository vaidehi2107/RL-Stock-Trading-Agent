# Simplified Reinforcement Learning Stock Trading (Q-Learning & DQN)

A clean, easy-to-understand implementation of Reinforcement Learning for Stock Trading, specifically simplified for **academic projects, student presentations, and faculty reviews**.

---

## 🎯 The Core Idea in 30 Seconds

The trading agent learns by trial and error whether to **HOLD**, **BUY**, or **SELL** stock each day to maximize profit.

| Component | What it is in this project | Plain English Explanation |
|:---|:---|:---|
| **State ($S$)** | `[Market Trend, Current Position]` | *Is the market going UP, FLAT, or DOWN? Are we holding CASH or STOCK?* |
| **Actions ($A$)** | `0 = HOLD`, `1 = BUY`, `2 = SELL` | *Do nothing, buy shares with cash, or sell shares for cash.* |
| **Reward ($R$)** | Daily % change in Net Worth | *Did today's decision make us money or lose money?* |
| **Goal** | Maximize total portfolio value | *Grow the initial \$10,000 cash balance over time.* |

---

## 📊 The 6-State Tabular Q-Table

In Tabular Q-Learning, the agent learns an exact lookup table with **only 6 possible states**:

| State | Market Trend | Our Position | Intuitive Best Action |
|:---:|:---:|:---:|:---|
| **1** | Market DOWN (-1) | In CASH (0) | **HOLD** (stay safe in cash, avoid falling knife) |
| **2** | Market DOWN (-1) | In STOCK (1) | **SELL** (cut loss before it falls further) |
| **3** | Market FLAT (0) | In CASH (0) | **HOLD** (wait for clear trend) |
| **4** | Market FLAT (0) | In STOCK (1) | **HOLD** (no reason to sell yet) |
| **5** | Market UP (+1) | In CASH (0) | **BUY** (enter market to catch the upward wave) |
| **6** | Market UP (+1) | In STOCK (1) | **HOLD** (ride the trend and maximize profits) |

### The Bellman Update Formula (Memorize for Faculty):
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
* **$\alpha$ (Learning Rate = 0.1)**: How much new experience overrides old knowledge.
* **$\gamma$ (Discount Factor = 0.95)**: How much the agent values future rewards vs immediate rewards.
* **$\epsilon$ (Epsilon Greedy = 1.0 $\to$ 0.01)**: Starts exploring randomly, then gradually shifts to exploiting best known actions.

---

## 🧠 Deep Q-Network (DQN) Architecture

When explaining DQN to faculty, explain that **DQN replaces the lookup table with a small Neural Network**:

```
Input Layer (2 neurons: [Trend, Position])
       ↓
Hidden Layer 1 (32 neurons, ReLU activation)
       ↓
Hidden Layer 2 (32 neurons, ReLU activation)
       ↓
Output Layer (3 neurons: [Q_HOLD, Q_BUY, Q_SELL])
```

* **Why DQN?** Instead of maintaining a discrete table, a Neural Network can accept continuous numbers and generalize across states.
* **Experience Replay**: Stores past transitions `(state, action, reward, next_state)` in a memory buffer and samples random mini-batches (size 32) so the network doesn't overfit to consecutive days.
* **Target Network**: A duplicate network whose weights update slowly, providing a stable target for the loss function ($MSE$).

---

## 🚀 How to Run the Code

### 1. Train and View the 6-State Q-Table (Runs in 2 seconds!)
```bash
python train_qlearning.py
```

### 2. Train the Deep Q-Network (Runs in 15 seconds!)
```bash
python train_dqn.py
```

### 3. Compare All Strategies on Out-of-Sample Test Data
```bash
python compare_agents.py
```

---

## 🎓 Faculty Presentation Guide (5-Minute Script)

When presenting to your professors, follow this structured outline:

### 1. Introduction (1 minute)
> *"Good morning professors. My project implements an autonomous Reinforcement Learning Stock Trading Agent. Financial markets are dynamic, so hardcoded rules often fail. We formulated stock trading as a Markov Decision Process (MDP) where the agent learns when to Buy, Sell, or Hold purely from trial and reward feedback."*

### 2. The Formulation (1 minute)
> *"To keep the model clean and explainable, we defined:*
> * *A minimal **State space of 2 features**: Market Trend (Up, Flat, Down based on a 5-day moving average) and Position Status (holding Cash or Stock).*
> * *An **Action space of 3 choices**: Hold, Buy, or Sell.*
> * *A **Reward function** based on the percentage change in portfolio net worth from one day to the next."*

### 3. Algorithm 1: Tabular Q-Learning (1.5 minutes)
> *"First, we implemented Tabular Q-Learning. Because our state has 3 trend values and 2 position values, the agent learns an exact 6-state by 3-action Q-Table using the Bellman equation.*
> *(Show the printed 6-state Q-table from `train_qlearning.py`).*
> *Notice how the agent learns to BUY when the market is UP and in CASH, and HOLD when already invested."*

### 4. Algorithm 2: Deep Q-Network (DQN) (1 minute)
> *"Second, to scale beyond lookup tables, we implemented a Deep Q-Network with PyTorch. It takes the state vector as input and outputs the 3 Q-values through two hidden layers of 32 neurons. We used Experience Replay to break temporal correlation in daily stock prices and a Target Network for stable convergence."*

### 5. Conclusion & Backtest Evaluation (30 seconds)
> *"We benchmarked both agents against the standard Buy-and-Hold strategy on an unseen 252-day test set. Both agents successfully manage risk and capital without human intervention."*

---

## ❓ Probable Faculty Viva Questions & Answers

**Q1: Why did you discretize the market trend into 3 values instead of using continuous prices?**
* **Answer**: *"Raw stock prices change continuously (e.g., \$150.25 vs \$150.30). For Tabular Q-Learning, continuous prices create an infinite state space that a lookup table cannot store. Discretizing into Uptrend, Downtrend, and Flat creates a compact 6-state space that is mathematically tractable and easy to verify."*

**Q2: What is the difference between Q-Learning and DQN?**
* **Answer**: *"Tabular Q-Learning stores values in a discrete 2D matrix $Q(s, a)$. DQN uses a Deep Neural Network as a function approximator to estimate $Q(s, a)$. DQN is capable of handling complex continuous inputs."*

**Q3: Why do we need an Experience Replay buffer in DQN?**
* **Answer**: *"In financial time series, consecutive daily prices are strongly correlated. If we train the network sequentially, the network forgets older patterns (catastrophic forgetting). Experience Replay samples random transitions from memory, breaking this autocorrelation."*

**Q4: How do you prevent lookahead bias in your evaluation?**
* **Answer**: *"We strictly split the historical data chronologically into 80% training (first 4 years) and 20% testing (final year). The agent was evaluated exclusively on unseen test days."*
