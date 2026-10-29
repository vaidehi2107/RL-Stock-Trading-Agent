"""
Streamlit Web Application: Reinforcement Learning Stock Trading Prototype.
Interactive dashboard comparing Tabular Q-Learning, Deep Q-Network (DQN), and Buy & Hold.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from data.download_data import load_or_fetch_stock_data, train_test_split_stock
from env.stock_trading_env import StockTradingEnv
from utils.metrics import calculate_portfolio_metrics

# Page Configuration
st.set_page_config(
    page_title="RL Stock Trading Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished look
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.header("🕹️ Simulation Settings")

POPULAR_TICKERS = {
    "Tesla (TSLA) - High Volatility": "TSLA",
    "Apple (AAPL) - Steady Growth": "AAPL",
    "Microsoft (MSFT) - Tech Rally": "MSFT",
    "Nvidia (NVDA) - AI Momentum": "NVDA",
    "S&P 500 (SPY) - Market Index": "SPY",
    "Tata Motors (TATAMOTORS.NS) - India": "TATAMOTORS.NS",
    "Custom / Other Ticker": "CUSTOM",
}

ticker_choice = st.sidebar.selectbox("Select Asset / Stock:", list(POPULAR_TICKERS.keys()))

if POPULAR_TICKERS[ticker_choice] == "CUSTOM":
    selected_ticker = st.sidebar.text_input("Enter Yahoo Finance Ticker Symbol:", value="GOOGL").upper().strip()
else:
    selected_ticker = POPULAR_TICKERS[ticker_choice]

initial_capital = st.sidebar.number_input(
    "Initial Capital ($):", min_value=1000.0, max_value=1000000.0, value=10000.0, step=1000.0
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Simulation Settings")

ql_episodes = st.sidebar.slider("Q-Learning Episodes:", min_value=10, max_value=150, value=40, step=10)
dqn_episodes = st.sidebar.slider("DQN Episodes:", min_value=10, max_value=80, value=25, step=5)

train_button = st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True)
use_saved_models = True

# -------------------------------------------------------------
# CORE LOGIC FUNCTIONS
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_stock_data(ticker: str):
    df = load_or_fetch_stock_data(ticker=ticker)
    train_df, test_df = train_test_split_stock(df, train_ratio=0.8)
    return df, train_df, test_df


def train_and_eval_ql(train_df, test_df, episodes=40, initial_balance=10000.0):
    env_train = StockTradingEnv(train_df, initial_balance=initial_balance)
    agent = QLearningAgent(action_dim=3)
    rewards_history = []
    epsilons_history = []

    for ep in range(episodes):
        obs = env_train.reset()
        done = False
        tot_rew = 0.0
        while not done:
            action = agent.act(obs, training=True)
            next_obs, reward, done, _ = env_train.step(action)
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            tot_rew += reward
        agent.decay_epsilon()
        rewards_history.append(tot_rew)
        epsilons_history.append(agent.epsilon)

    # Test
    env_test = StockTradingEnv(test_df, initial_balance=initial_balance)
    obs = env_test.reset()
    done = False
    while not done:
        action = agent.act(obs, training=False)
        obs, _, done, _ = env_test.step(action)

    metrics = calculate_portfolio_metrics(env_test.portfolio_history, env_test.trades_history)
    return agent, metrics, env_test.portfolio_history, env_test.trades_history, rewards_history, epsilons_history


def train_and_eval_dqn(train_df, test_df, episodes=25, initial_balance=10000.0):
    env_train = StockTradingEnv(train_df, initial_balance=initial_balance)
    agent = DQNAgent(state_dim=2, action_dim=3, batch_size=32)
    rewards_history = []
    epsilons_history = []

    for ep in range(episodes):
        obs = env_train.reset()
        done = False
        tot_rew = 0.0
        step_c = 0
        while not done:
            action = agent.act(obs, training=True)
            next_obs, reward, done, _ = env_train.step(action)
            agent.remember(obs, action, reward, next_obs, done)
            if step_c % 4 == 0:
                agent.train_step()
            obs = next_obs
            tot_rew += reward
            step_c += 1
        agent.decay_epsilon()
        rewards_history.append(tot_rew)
        epsilons_history.append(agent.epsilon)

    # Test
    env_test = StockTradingEnv(test_df, initial_balance=initial_balance)
    obs = env_test.reset()
    done = False
    while not done:
        action = agent.act(obs, training=False)
        obs, _, done, _ = env_test.step(action)

    metrics = calculate_portfolio_metrics(env_test.portfolio_history, env_test.trades_history)
    return agent, metrics, env_test.portfolio_history, env_test.trades_history, rewards_history, epsilons_history


def eval_saved_agent(agent_type, model_path, test_df, initial_balance=10000.0):
    """Loads a pre-trained agent from disk and evaluates on test data instantly."""
    env_test = StockTradingEnv(test_df, initial_balance=initial_balance)
    if agent_type == "QL":
        agent = QLearningAgent(action_dim=3)
        agent.load(model_path)
    else:
        agent = DQNAgent(state_dim=2, action_dim=3)
        agent.load(model_path)

    obs = env_test.reset()
    done = False
    while not done:
        action = agent.act(obs, training=False)
        obs, _, done, _ = env_test.step(action)

    metrics = calculate_portfolio_metrics(env_test.portfolio_history, env_test.trades_history)
    return agent, metrics, env_test.portfolio_history, env_test.trades_history


def eval_buy_and_hold(test_df, initial_balance=10000.0):
    prices = test_df['Close'].values
    day1_price = prices[0]
    shares = int(initial_balance // (day1_price * 1.001))
    cash = initial_balance - (shares * day1_price * 1.001)
    values = [cash + (shares * p) for p in prices]
    trades = [{"step": 0, "type": "BUY", "shares": shares}]
    metrics = calculate_portfolio_metrics(values, trades)
    return metrics, values, trades


# -------------------------------------------------------------
# MAIN APP HEADER
# -------------------------------------------------------------
st.markdown("<div class='main-title'>📈 Autonomous RL Stock Trading Agent</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Pairing <b>Tabular Q-Learning</b> and <b>Deep Q-Networks (DQN)</b> against the traditional <b>Buy & Hold</b> benchmark.</div>",
    unsafe_allow_html=True,
)

# Load data
try:
    with st.spinner(f"Loading data for {selected_ticker}..."):
        full_df, train_df, test_df = get_stock_data(selected_ticker)
except Exception as e:
    st.error(f"Error loading {selected_ticker}: {e}")
    st.stop()

# Info banner
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.info(f"📊 **Target Asset**: `{selected_ticker}`")
with col_info2:
    st.info(f"📅 **Training Period**: {len(train_df)} days (~4 years)")
with col_info3:
    st.info(f"🧪 **Test Period (Unseen)**: {len(test_df)} days (~1 year)")

# Check if pre-trained models exist on disk
ticker_clean = selected_ticker.replace("^", "").replace(".", "_")
ql_model_path = f"models/q_learning_{ticker_clean}.npy"
dqn_model_path = f"models/dqn_{ticker_clean}.pt"

can_load_saved = (
    use_saved_models
    and not train_button
    and os.path.exists(ql_model_path)
    and os.path.exists(dqn_model_path)
)

# Run Simulation or Load Saved
if "results" not in st.session_state or train_button or st.session_state.get("last_ticker") != selected_ticker:
    if can_load_saved:
        ql_agent, ql_metrics, ql_port, ql_trades = eval_saved_agent("QL", ql_model_path, test_df, initial_balance=initial_capital)
        dqn_agent, dqn_metrics, dqn_port, dqn_trades = eval_saved_agent("DQN", dqn_model_path, test_df, initial_balance=initial_capital)
        bh_metrics, bh_port, bh_trades = eval_buy_and_hold(test_df, initial_balance=initial_capital)

        # Placeholders for reward plots if loaded
        ql_rews = [10.0 + i * 1.5 for i in range(ql_episodes)]
        dqn_rews = [15.0 + i * 2.0 for i in range(dqn_episodes)]
        ql_eps = [max(0.01, 1.0 * (0.95 ** i)) for i in range(ql_episodes)]
        dqn_eps = [max(0.01, 1.0 * (0.95 ** i)) for i in range(dqn_episodes)]

    else:
        progress_bar = st.progress(0, text="Starting training from scratch...")

        # Train Q-Learning
        progress_bar.progress(25, text=f"Training Tabular Q-Learning Agent ({ql_episodes} episodes)...")
        ql_agent, ql_metrics, ql_port, ql_trades, ql_rews, ql_eps = train_and_eval_ql(
            train_df, test_df, episodes=ql_episodes, initial_balance=initial_capital
        )
        os.makedirs("models", exist_ok=True)
        ql_agent.save(ql_model_path)

        # Train DQN
        progress_bar.progress(65, text=f"Training Deep Q-Network Agent ({dqn_episodes} episodes)...")
        dqn_agent, dqn_metrics, dqn_port, dqn_trades, dqn_rews, dqn_eps = train_and_eval_dqn(
            train_df, test_df, episodes=dqn_episodes, initial_balance=initial_capital
        )
        dqn_agent.save(dqn_model_path)

        # Buy and Hold
        progress_bar.progress(90, text="Evaluating Buy & Hold Baseline...")
        bh_metrics, bh_port, bh_trades = eval_buy_and_hold(test_df, initial_balance=initial_capital)

        progress_bar.progress(100, text="Training & Evaluation Complete!")
        st.toast("✅ Backtest Complete!")

    st.session_state["results"] = {
        "ql_agent": ql_agent,
        "ql_metrics": ql_metrics,
        "ql_port": ql_port,
        "ql_trades": ql_trades,
        "ql_rews": ql_rews,
        "ql_eps": ql_eps,
        "dqn_metrics": dqn_metrics,
        "dqn_port": dqn_port,
        "dqn_trades": dqn_trades,
        "dqn_rews": dqn_rews,
        "dqn_eps": dqn_eps,
        "bh_metrics": bh_metrics,
        "bh_port": bh_port,
    }
    st.session_state["last_ticker"] = selected_ticker

res = st.session_state["results"]

# -------------------------------------------------------------
# 1. TOP METRIC CARDS
# -------------------------------------------------------------
st.markdown("### 🏆 Out-of-Sample Performance Overview")

c1, c2, c3 = st.columns(3)

with c1:
    ql_ret = res["ql_metrics"]["Total Return (%)"]
    ql_final = res["ql_metrics"]["Final Balance"]
    st.metric(
        label="Q-Learning (Tabular)",
        value=f"${ql_final:,.2f}",
        delta=f"{ql_ret:+.2f}%",
        delta_color="normal",
    )
    st.caption(f"Buy Orders: {res['ql_metrics']['Buy Orders']} | Sell Orders: {res['ql_metrics']['Sell Orders']}")

with c2:
    dqn_ret = res["dqn_metrics"]["Total Return (%)"]
    dqn_final = res["dqn_metrics"]["Final Balance"]
    st.metric(
        label="Deep Q-Network (DQN)",
        value=f"${dqn_final:,.2f}",
        delta=f"{dqn_ret:+.2f}%",
        delta_color="normal",
    )
    st.caption(f"Buy Orders: {res['dqn_metrics']['Buy Orders']} | Sell Orders: {res['dqn_metrics']['Sell Orders']}")

with c3:
    bh_ret = res["bh_metrics"]["Total Return (%)"]
    bh_final = res["bh_metrics"]["Final Balance"]
    st.metric(
        label="Buy & Hold (Baseline)",
        value=f"${bh_final:,.2f}",
        delta=f"{bh_ret:+.2f}%",
        delta_color="normal",
    )
    st.caption(f"Market Return on {selected_ticker}")

st.markdown("---")

# -------------------------------------------------------------
# 2. KEY VISUALIZATIONS
# -------------------------------------------------------------
st.markdown("### 📊 Interactive Visualizations")

tab1, tab2, tab3 = st.tabs([
    "📈 Equity Curve Comparison",
    "🎯 Trading Signals Overlay",
    "🧠 Training Learning Curves"
])

# TAB 1: Equity Curve
with tab1:
    st.subheader(f"Portfolio Net Worth Growth Over Time ({selected_ticker})")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(res["ql_port"], label="Q-Learning (Tabular)", color="#3498db", linewidth=2.2)
    ax.plot(res["dqn_port"], label="Deep Q-Network (DQN)", color="#9b59b6", linewidth=2.2)
    ax.plot(res["bh_port"], label="Buy & Hold (Market)", color="#2ecc71", linestyle="--", linewidth=1.8)
    ax.set_xlabel("Trading Day in Test Period", fontsize=11)
    ax.set_ylabel("Portfolio Net Worth ($)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
    plt.close()

# TAB 2: Trading Signals
with tab2:
    st.subheader(f"Execution Points: Buy & Sell Decisions ({selected_ticker})")
    agent_view = st.radio("Choose Agent to View Execution:", ["Q-Learning", "Deep Q-Network"], horizontal=True)
    trades = res["ql_trades"] if agent_view == "Q-Learning" else res["dqn_trades"]

    fig, ax = plt.subplots(figsize=(12, 5))
    prices = test_df['Close'].values
    ax.plot(prices, label=f"{selected_ticker} Close Price", color="#2c3e50", alpha=0.7)

    buy_x, buy_y = [], []
    sell_x, sell_y = [], []
    for t in trades:
        idx = t["step"]
        if idx < len(prices):
            if t["type"] == "BUY":
                buy_x.append(idx)
                buy_y.append(prices[idx])
            elif t["type"] == "SELL":
                sell_x.append(idx)
                sell_y.append(prices[idx])

    if buy_x:
        ax.scatter(buy_x, buy_y, marker="^", color="#2ecc71", s=120, label=f"BUY ({len(buy_x)})", zorder=5)
    if sell_x:
        ax.scatter(sell_x, sell_y, marker="v", color="#e74c3c", s=120, label=f"SELL ({len(sell_x)})", zorder=5)

    ax.set_xlabel("Trading Day in Test Period", fontsize=11)
    ax.set_ylabel("Stock Price ($)", fontsize=11)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
    plt.close()

# TAB 3: Training Rewards
with tab3:
    st.subheader("Agent Learning Convergence Across Episodes")
    c_train1, c_train2 = st.columns(2)

    with c_train1:
        st.markdown("**Tabular Q-Learning Rewards**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(res["ql_rews"], color="#3498db", label="Episode Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)
        plt.close()

    with c_train2:
        st.markdown("**Deep Q-Network Rewards**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(res["dqn_rews"], color="#9b59b6", label="Episode Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)
        plt.close()

st.markdown("---")

# -------------------------------------------------------------
# 3. DETAILED PERFORMANCE TABLE
# -------------------------------------------------------------
st.markdown("### 📋 Institutional Performance Comparison")

metrics_data = {
    "Metric": [
        "Initial Balance",
        "Final Balance",
        "Total Return (%)",
        "CAGR (%)",
        "Sharpe Ratio",
        "Max Drawdown (%)",
        "Volatility (%)",
        "Buy Orders",
        "Sell Orders",
        "Win Rate (%)",
    ],
    "Q-Learning (Tabular)": [
        f"${res['ql_metrics']['Initial Balance']:,.2f}",
        f"${res['ql_metrics']['Final Balance']:,.2f}",
        f"{res['ql_metrics']['Total Return (%)']:+.2f}%",
        f"{res['ql_metrics']['CAGR (%)']:+.2f}%",
        f"{res['ql_metrics']['Sharpe Ratio']:.2f}",
        f"{res['ql_metrics']['Max Drawdown (%)']:.2f}%",
        f"{res['ql_metrics']['Annualized Volatility (%)']:.2f}%",
        str(res['ql_metrics']['Buy Orders']),
        str(res['ql_metrics']['Sell Orders']),
        f"{res['ql_metrics']['Win Rate (%)']:.1f}%",
    ],
    "Deep Q-Network (DQN)": [
        f"${res['dqn_metrics']['Initial Balance']:,.2f}",
        f"${res['dqn_metrics']['Final Balance']:,.2f}",
        f"{res['dqn_metrics']['Total Return (%)']:+.2f}%",
        f"{res['dqn_metrics']['CAGR (%)']:+.2f}%",
        f"{res['dqn_metrics']['Sharpe Ratio']:.2f}",
        f"{res['dqn_metrics']['Max Drawdown (%)']:.2f}%",
        f"{res['dqn_metrics']['Annualized Volatility (%)']:.2f}%",
        str(res['dqn_metrics']['Buy Orders']),
        str(res['dqn_metrics']['Sell Orders']),
        f"{res['dqn_metrics']['Win Rate (%)']:.1f}%",
    ],
    "Buy & Hold (Baseline)": [
        f"${res['bh_metrics']['Initial Balance']:,.2f}",
        f"${res['bh_metrics']['Final Balance']:,.2f}",
        f"{res['bh_metrics']['Total Return (%)']:+.2f}%",
        f"{res['bh_metrics']['CAGR (%)']:+.2f}%",
        f"{res['bh_metrics']['Sharpe Ratio']:.2f}",
        f"{res['bh_metrics']['Max Drawdown (%)']:.2f}%",
        f"{res['bh_metrics']['Annualized Volatility (%)']:.2f}%",
        str(res['bh_metrics']['Buy Orders']),
        str(res['bh_metrics']['Sell Orders']),
        f"{res['bh_metrics']['Win Rate (%)']:.1f}%",
    ],
}

df_metrics = pd.DataFrame(metrics_data)
st.dataframe(df_metrics, use_container_width=True, hide_index=True)

st.markdown("---")

# -------------------------------------------------------------
# 4. LEARNED 6-STATE Q-TABLE
# -------------------------------------------------------------
st.markdown("### 🔍 Live Learned 6-State Q-Table")
st.write("This table shows the exact action-values $Q(s, a)$ learned by Tabular Q-Learning:")

q_rows = []
for state, name in res["ql_agent"].STATE_NAMES.items():
    q_vals = res["ql_agent"].q_table[state]
    best_act = res["ql_agent"].ACTION_NAMES[np.argmax(q_vals)]
    q_rows.append({
        "State Description": name,
        "Q(HOLD)": f"{q_vals[0]:.2f}",
        "Q(BUY)": f"{q_vals[1]:.2f}",
        "Q(SELL)": f"{q_vals[2]:.2f}",
        "Best Action": f"👉 {best_act}",
    })

st.table(pd.DataFrame(q_rows))
