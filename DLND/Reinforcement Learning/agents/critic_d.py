# File to store the critic class to be used by the agent

from keras import layers, models, optimizers, initializers
from keras import backend as K
import numpy as np

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()
        
    def huber_loss(self,a, b, in_keras=True):
        error = a - b
        quadratic_term = error*error / 2
        linear_term = abs(error) - 1/2
        use_linear_term = (abs(error) > 1.0)
        if in_keras:
            use_linear_term = K.cast(use_linear_term, 'float32')
            return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term
   
    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        # Normalise the states entering the network
        net_states = layers.BatchNormalization()(states)
        
        actions = layers.Input(shape=(self.action_size,), name='actions')
        # Normalise the actions before being fed into the network
        net_actions = layers.Lambda(lambda x: ((x - self.action_low) / (self.action_range / 2)) - 1)(actions)

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=400, activation='linear')(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('elu')(net_states)
        net_states = layers.Dense(units=300, activation='linear')(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('elu')(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=300, activation='linear')(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('elu')(net_actions)
#         net_actions = layers.Dense(units=128, activation='linear')(net_actions)
#         net_actions = layers.BatchNormalization()(net_actions)
#         net_actions = layers.Activation('relu')(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.
        

        # Combine state and action pathways
        net = layers.Concatenate()([net_states, net_actions])
        net = layers.BatchNormalization()(net)
        net = layers.Activation('elu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        # Initialise the final weight values with random normal between -0.003 and 0.003
        Q_values = layers.Dense(units=1, name='q_values',
                                kernel_initializer=initializers.RandomUniform(minval=-0.003, maxval=0.003))(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)  # Divided gradients by batch size

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)