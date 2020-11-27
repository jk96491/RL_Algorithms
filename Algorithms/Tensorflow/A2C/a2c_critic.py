from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam


class Critic(object):
    def __init__(self, state_dim, action_dim, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        self.model, self.states = self.build_network()

        self.model.compile(optimizer=Adam(self.learning_rate), loss='mse')

    def build_network(self):
        state_input = Input((self.state_dim,))
        h1 = Dense(64, activation='relu')(state_input)
        h2 = Dense(32, activation='relu')(h1)
        h3 = Dense(16, activation='relu')(h2)
        v_output = Dense(1, activation='linear')(h3)
        model = Model(state_input, v_output)
        model.summary()

        return model, state_input

    def train_in_batch(self, states, td_targets):
        return self.model.train_on_batch(states, td_targets)

    def save_weights(self, path):
        self.model.save_weights(path)

    def load__weights(self, path):
        self.model.load_weights(path + "pendulum_critic.h5")

