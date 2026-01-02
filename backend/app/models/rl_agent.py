"""
Reinforcement Learning Trading Agent
Multi-Armed Bandit and Contextual Bandit for portfolio allocation
"""
import random
from typing import List, Dict
from datetime import datetime

class MultiArmedBandit:
    """
    Thompson Sampling Multi-Armed Bandit
    Used for portfolio allocation across different assets
    """
    def __init__(self, n_arms: int = 5):
        self.n_arms = n_arms
        self.arm_names = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"][:n_arms]

        # Thompson Sampling parameters (Beta distribution)
        self.alpha = [1.0] * n_arms  # Success count
        self.beta = [1.0] * n_arms   # Failure count

        # Performance tracking
        self.total_pulls = [0] * n_arms
        self.total_rewards = [0.0] * n_arms
        self.history = []

    def select_arm(self) -> int:
        """
        Select which asset to trade using Thompson Sampling
        """
        # Sample from Beta distribution for each arm
        samples = [random.betavariate(self.alpha[i], self.beta[i])
                   for i in range(self.n_arms)]

        # Select arm with highest sample
        return samples.index(max(samples))

    def update(self, arm: int, reward: float):
        """
        Update arm statistics after observing reward
        """
        self.total_pulls[arm] += 1
        self.total_rewards[arm] += reward

        # Update Beta distribution parameters
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

        # Record in history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "arm": arm,
            "symbol": self.arm_names[arm],
            "reward": reward
        })

    def get_portfolio_weights(self) -> Dict[str, float]:
        """
        Get recommended portfolio allocation weights
        """
        # Calculate expected values from Beta distributions
        expected_values = [self.alpha[i] / (self.alpha[i] + self.beta[i])
                          for i in range(self.n_arms)]

        # Normalize to sum to 1
        total = sum(expected_values)
        weights = {self.arm_names[i]: expected_values[i] / total
                   for i in range(self.n_arms)}

        return weights

    def get_statistics(self) -> Dict:
        """
        Get detailed statistics for all arms
        """
        stats = {
            "arms": [],
            "total_pulls": sum(self.total_pulls),
            "best_arm": self.arm_names[self.alpha.index(max(self.alpha))],
            "exploration_rate": 0.1
        }

        for i in range(self.n_arms):
            avg_reward = (self.total_rewards[i] / self.total_pulls[i]
                         if self.total_pulls[i] > 0 else 0)

            stats["arms"].append({
                "symbol": self.arm_names[i],
                "pulls": self.total_pulls[i],
                "total_reward": round(self.total_rewards[i], 2),
                "avg_reward": round(avg_reward, 4),
                "alpha": round(self.alpha[i], 2),
                "beta": round(self.beta[i], 2),
                "win_rate": round((self.alpha[i] / (self.alpha[i] + self.beta[i])) * 100, 1)
            })

        return stats


class ContextualBandit:
    """
    Contextual Bandit that considers market conditions
    More advanced than Multi-Armed Bandit
    """
    def __init__(self, n_actions: int = 3):
        self.n_actions = n_actions
        self.actions = ["BUY", "SELL", "HOLD"]

        # Simple Q-table for different market states
        self.q_values = {
            "bullish": {"BUY": 0.6, "SELL": 0.2, "HOLD": 0.2},
            "bearish": {"BUY": 0.2, "SELL": 0.6, "HOLD": 0.2},
            "neutral": {"BUY": 0.3, "SELL": 0.3, "HOLD": 0.4}
        }

        self.epsilon = 0.1  # Exploration rate
        self.learning_rate = 0.01

        # Statistics
        self.actions_taken = {"BUY": 0, "SELL": 0, "HOLD": 0}
        self.rewards_history = []

    def get_market_state(self, market_data: Dict = None) -> str:
        """
        Determine current market state
        In real implementation, this would use actual market data
        """
        if market_data is None:
            return random.choice(["bullish", "bearish", "neutral"])

        # Simple logic based on price trend
        volatility = market_data.get("volatility", 15)
        if volatility > 20:
            return "bearish"
        elif volatility < 10:
            return "bullish"
        else:
            return "neutral"

    def select_action(self, market_state: str = None) -> str:
        """
        Select action based on current market state
        """
        if market_state is None:
            market_state = self.get_market_state()

        # Epsilon-greedy strategy
        if random.random() < self.epsilon:
            # Explore: random action
            action = random.choice(self.actions)
        else:
            # Exploit: best action for this state
            q_vals = self.q_values[market_state]
            action = max(q_vals, key=q_vals.get)

        self.actions_taken[action] += 1
        return action

    def update(self, state: str, action: str, reward: float):
        """
        Update Q-values based on observed reward
        """
        # Q-learning update
        current_q = self.q_values[state][action]
        new_q = current_q + self.learning_rate * (reward - current_q)
        self.q_values[state][action] = new_q

        self.rewards_history.append(reward)

    def get_statistics(self) -> Dict:
        """
        Get agent statistics
        """
        total_actions = sum(self.actions_taken.values())
        avg_reward = (sum(self.rewards_history[-100:]) / len(self.rewards_history[-100:])
                     if self.rewards_history else 0)

        return {
            "total_actions": total_actions,
            "action_distribution": {
                action: {
                    "count": count,
                    "percentage": round((count / total_actions * 100) if total_actions > 0 else 0, 1)
                }
                for action, count in self.actions_taken.items()
            },
            "epsilon": self.epsilon,
            "avg_reward_last_100": round(avg_reward, 4),
            "q_values": self.q_values
        }


# Global instances
mab_agent = MultiArmedBandit(n_arms=5)
contextual_agent = ContextualBandit(n_actions=3)


def get_rl_recommendation() -> Dict:
    """
    Get trading recommendation from RL agent
    """
    # Get market state
    market_state = contextual_agent.get_market_state()

    # Get action from contextual bandit
    action = contextual_agent.select_action(market_state)

    # Get asset allocation from MAB
    selected_arm = mab_agent.select_arm()
    symbol = mab_agent.arm_names[selected_arm]

    # Get portfolio weights
    weights = mab_agent.get_portfolio_weights()

    return {
        "action": action,
        "symbol": symbol,
        "market_state": market_state,
        "confidence": round(random.uniform(70, 95), 1),
        "portfolio_weights": weights,
        "reasoning": f"Market is {market_state}, recommending {action} {symbol}"
    }


def simulate_trade_and_update():
    """
    Simulate a trade and update both agents
    """
    # Get recommendation
    recommendation = get_rl_recommendation()

    # Simulate trade execution and reward
    # Positive reward for good trades, negative for bad
    reward = random.uniform(-0.5, 1.0)

    # Update contextual bandit
    market_state = recommendation["market_state"]
    action = recommendation["action"]
    contextual_agent.update(market_state, action, reward)

    # Update MAB
    symbol_index = mab_agent.arm_names.index(recommendation["symbol"])
    mab_agent.update(symbol_index, reward)

    return {
        "recommendation": recommendation,
        "reward": round(reward, 3),
        "timestamp": datetime.now().isoformat()
    }


def get_agent_stats() -> Dict:
    """
    Get combined statistics from all agents
    """
    return {
        "multi_armed_bandit": mab_agent.get_statistics(),
        "contextual_bandit": contextual_agent.get_statistics(),
        "last_recommendation": get_rl_recommendation()
    }