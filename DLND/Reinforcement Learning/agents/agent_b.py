# DDPG network from the course/class

# Import the memory, actor, critic, and noise classes for the agent to use
from agents.actor_b import Actor
from agents.critic_b import Critic
from agents.OUnoise import OUNoise
from agents.RBuffer import ReplayBuffer
from keras import layers

import numpy as np

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        self.score = 0

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, self.action_low, self.action_high)
        self.critic_target = Critic(self.state_size, self.action_size, self.action_low, self.action_high)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000 # Taken from paper - changed from 10000 originally
        self.batch_size = 64 # Taken from paper - changed from 64 originally
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # Changed from 0.005 used previoussly to 0.001 now from DDPG paper
        
        # Reset the episode when model set up
        self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        self.score = 0.0
        self.count = 0
        self.best_score = 0.0
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state
    
    def normalise_actions(self, actions):
        normalised_actions = (actions * self.action_range / 2) + self.action_low + self.action_range / 2
        return normalised_actions

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > 10000:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state
        self.count += 1
        self.total_reward += reward
        

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        action = self.normalise_actions(action)
        return list(action + self.noise.sample())  # add some noise for exploration
    
    def act_eval(self, state):
        """Returns actions without exploration for final evaluation."""
        state = np.reshape(state, [-1, self.state_size])
        # Scale [0, 1] output for each action dimension to proper range
        action = self.actor_local.model.predict(state)[0]
        action = self.normalise_actions(action)
        action = list(action)
        
        return list(action)  # No noise for exploration. Evaluate final performance of quadcopter

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        actions_next = self.normalise_actions(actions_next)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)
        
        # Update the score metric so it can be tracked through training
        self.score = self.total_reward / float(self.count) if self.count else 0.0


    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)