from __future__ import annotations

from dataclasses import dataclass, field
from random import Random


class MultiArmedBandit:
    def __init__(self, reward_probabilities: list[float], seed: int = 7) -> None:
        self.reward_probabilities = reward_probabilities
        self.rng = Random(seed)

    def pull(self, arm: int) -> int:
        return 1 if self.rng.random() < self.reward_probabilities[arm] else 0


@dataclass
class EpsilonGreedyAgent:
    num_arms: int
    epsilon: float = 0.1
    counts: list[int] = field(init=False)
    values: list[float] = field(init=False)
    rng: Random = field(default_factory=lambda: Random(11))

    def __post_init__(self) -> None:
        self.counts = [0] * self.num_arms
        self.values = [0.0] * self.num_arms

    def select_arm(self) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.num_arms)
        return max(range(self.num_arms), key=lambda arm: self.values[arm])

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        step_size = 1 / self.counts[arm]
        self.values[arm] += step_size * (reward - self.values[arm])


class LineWorld:
    def __init__(self) -> None:
        self.goal = 4
        self.start = 0

    def step(self, state: int, action: int) -> tuple[int, float, bool]:
        next_state = max(0, min(self.goal, state + action))
        done = next_state == self.goal
        reward = 1.0 if done else -0.02
        return next_state, reward, done


def q_learning_demo(episodes: int = 150, seed: int = 17) -> dict[tuple[int, int], float]:
    rng = Random(seed)
    env = LineWorld()
    actions = [-1, 1]
    q_values: dict[tuple[int, int], float] = {(state, action): 0.0 for state in range(env.goal + 1) for action in actions}
    learning_rate = 0.2
    gamma = 0.95
    epsilon = 0.15

    for _ in range(episodes):
        state = env.start
        done = False
        while not done:
            if rng.random() < epsilon:
                action = rng.choice(actions)
            else:
                action = max(actions, key=lambda candidate: q_values[(state, candidate)])
            next_state, reward, done = env.step(state, action)
            best_next = max(q_values[(next_state, candidate)] for candidate in actions)
            current = q_values[(state, action)]
            q_values[(state, action)] = current + learning_rate * (reward + gamma * best_next - current)
            state = next_state
    return q_values


def main() -> None:
    bandit = MultiArmedBandit([0.15, 0.35, 0.6])
    agent = EpsilonGreedyAgent(num_arms=3)
    total_reward = 0
    for _ in range(200):
        arm = agent.select_arm()
        reward = bandit.pull(arm)
        agent.update(arm, reward)
        total_reward += reward

    print("Bandit estimates")
    for arm, value in enumerate(agent.values):
        print(f"- arm={arm} estimated_value={value:.3f} pulls={agent.counts[arm]}")
    print(f"- total_reward={total_reward}")

    print("\nQ-learning values")
    q_values = q_learning_demo()
    for state in range(4):
        left = q_values[(state, -1)]
        right = q_values[(state, 1)]
        best_action = -1 if left > right else 1
        print(f"- state={state} q_left={left:.3f} q_right={right:.3f} best_action={best_action}")


if __name__ == "__main__":
    main()
