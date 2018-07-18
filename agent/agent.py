import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque

class Agent:
    # takes state size, model name and is_eval for loading model
    def __init__(self, state_size, is_eval = False, model_name=""):
        self.state_size = state_size
        # 3 tipes of action : sell, buy, weit
        self.action_size = 3
        # memory max size vector
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        # varables for RL agent
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # we check if we load the model
        if is_eval:
            self.model = load_model("models/" + model_name)
        else:
            self.model = self._model()

    # model method
    def _model(self):
        model = Sequential()
        # https://keras.io/layers/core/ info about layers
        model.add(Dense(units=64, input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(units=32, activation = 'relu'))
        model.add(Dense(units=8, activation = 'relu'))
        # https://keras.io/activations/ about activations
        model.add(Dense(self.state_size, activation = 'linear'))
        model.compile(loss="mse", optimizer = Adam(lr=0.001))
        return model

    # prediction class
    def act(self, state):
        # if we dont load model and we explore
        # more info in yt video:
        # https://youtu.be/0g4j2k_Ggc4?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
        # np.random.rand() gives value betwen 0 and 1
        if not self.is_eval and np.random.rand() <= self.epsilon:
            # random.randrange - Return a randomly selected element-
            # -from range(start, stop, step)
            # here return value betwen 0 and self.action_size
            return random.randrange(self.action_size)
        # predict action
        options = self.model.predict(state)
        # take most predicted action
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        mini_batch = []
        memory_len = len(self.memory)
        # we add object to mini_batch forom memory
        for i in range(memory_len - batch_size + 1, memory_len):
            mini_batch.append(self.memory[i])
        # q-lerning algoritym, more info:
        # https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/16_Reinforcement_Learning.ipynb
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            # we check if we miss
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f =self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs = 1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
