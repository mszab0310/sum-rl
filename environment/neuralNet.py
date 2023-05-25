import tensorflow as tf

class TrafficLightController:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        # Define the neural network
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(self.observation_space,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(self.action_space, activation='softmax')
        ])

    def act(self, state):
        # Use the neural network to predict the action
        action_probabilities = self.model.predict(tf.expand_dims(state, axis=0)).flatten()
        action_index = tf.random.categorical(tf.math.log([action_probabilities]), num_samples=1)[0, 0]
        action = self._get_action_from_index(action_index)

        # Return the action
        return action

    def train(self, states, actions, rewards):
        # Convert the actions to one-hot vectors
        actions_one_hot = tf.one_hot(actions, depth=self.action_space)

        # Compute the loss and gradients
        with tf.GradientTape() as tape:
            logits = self.model(states)
            loss = tf.keras.losses.categorical_crossentropy(actions_one_hot, logits, from_logits=True)
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Apply the gradients to the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def _get_action_from_index(self, index):
        # Map the action index to the corresponding action
        # Here, we assume the action space is a list of tuples, where each tuple contains the phase and its

