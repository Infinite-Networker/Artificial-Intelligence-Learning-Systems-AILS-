"""
AILS Reinforcement Learning Agent — Deep Q-Network (DQN)
Supports custom environments: trading, game AI, robotics, and more.
Artificial Intelligence Learning System — Created by Cherry Computer Ltd.
"""

import numpy as np
import random
import logging
from collections import deque
from typing import Optional, Tuple, List


class AILSRLAgent:
    """
    AILS Deep Q-Network (DQN) Reinforcement Learning Agent.

    Implements epsilon-greedy exploration, experience replay memory,
    and target network for stable Q-learning.

    Example:
        agent = AILSRLAgent(state_size=8, action_size=4)
        for episode in range(500):
            state = env.reset()
            for step in range(200):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                state = next_state
                if done:
                    break
            agent.update_target_model()
    """

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 2000,
                 batch_size: int = 32):
        """
        Args:
            state_size: Dimension of the observation/state space.
            action_size: Number of discrete actions available.
            learning_rate: Adam optimizer learning rate.
            gamma: Discount factor for future rewards (0–1).
            epsilon: Initial exploration rate (1.0 = fully random).
            epsilon_min: Minimum exploration rate.
            epsilon_decay: Multiplicative decay per episode.
            memory_size: Maximum capacity of replay buffer.
            batch_size: Mini-batch size for training.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.logger = logging.getLogger("AILS.RL.DQNAgent")

        # Build online and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.logger.info(
            f"✅ DQN Agent initialized — "
            f"states={state_size}, actions={action_size}"
        )

    def _build_model(self):
        """Build the Q-network (state → Q-values for each action)."""
        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                128, input_dim=self.state_size, activation="relu"
            ),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(self.action_size, activation="linear"),
        ])
        model.compile(
            loss="huber",
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            ),
        )
        return model

    def update_target_model(self) -> None:
        """Copy weights from online model to target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state: np.ndarray, action: int,
                  reward: float, next_state: np.ndarray,
                  done: bool) -> None:
        """
        Store a (state, action, reward, next_state, done) transition.

        Args:
            state: Current environment state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state after action.
            done: Whether the episode ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state array, shape (1, state_size).

        Returns:
            Integer action index.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(
            np.reshape(state, [1, self.state_size]), verbose=0
        )
        return int(np.argmax(q_values[0]))

    def replay(self) -> Optional[float]:
        """
        Sample a mini-batch from memory and train the Q-network.

        Returns:
            Training loss for this batch, or None if not enough samples.
        """
        if len(self.memory) < self.batch_size:
            return None

        minibatch = random.sample(self.memory, self.batch_size)
        states      = np.array([e[0] for e in minibatch])
        actions     = np.array([e[1] for e in minibatch])
        rewards     = np.array([e[2] for e in minibatch])
        next_states = np.array([e[3] for e in minibatch])
        dones       = np.array([e[4] for e in minibatch])

        # Q-target using target network (Double DQN style)
        q_values_next = self.target_model.predict(next_states, verbose=0)
        targets = rewards + self.gamma * np.max(q_values_next, axis=1) * (1 - dones)

        q_values = self.model.predict(states, verbose=0)
        q_values[np.arange(self.batch_size), actions] = targets

        history = self.model.fit(states, q_values, epochs=1,
                                  batch_size=self.batch_size, verbose=0)
        loss = history.history["loss"][0]

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, path: str) -> None:
        """Save the Q-network weights."""
        self.model.save_weights(path)
        self.logger.info(f"💾 DQN weights saved to '{path}'")

    def load(self, path: str) -> None:
        """Load Q-network weights from file."""
        self.model.load_weights(path)
        self.update_target_model()
        self.logger.info(f"📂 DQN weights loaded from '{path}'")

    @property
    def memory_size(self) -> int:
        """Return current number of stored experiences."""
        return len(self.memory)
