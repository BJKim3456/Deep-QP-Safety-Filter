import random
import numpy as np
import pickle

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push_one_signal(self, state, action, signal, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, signal, next_state, done)
        self.position = int( (self.position + 1) % self.capacity )

    def push_two_signals(self, state, action, signal, next_state, next_signal, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, signal, next_state, next_signal, done)
        self.position = int( (self.position + 1) % self.capacity )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, save_path, add_arg="/buffer"):
        save_path_ = save_path+add_arg
        with open(save_path_, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, load_path, add_arg="/buffer"):
        with open(load_path+add_arg, "rb") as f:
            self.buffer = pickle.load(f) 
            self.capacity = len(self.buffer)
            self.position = 0 

    def load_and_concatenate_buffer(self, load_path):
        with open(load_path, "rb") as f:
            self.buffer = self.buffer + pickle.load(f)
            self.capacity = np.maximum(len(self.buffer), self.capacity)
            self.position = int(len(self.buffer) % self.capacity)