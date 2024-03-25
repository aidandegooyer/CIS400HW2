import keras.models
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import confusion_matrix


class Individual:
    def __init__(self, input_dim):
        self.q = None
        self.input_dim = input_dim
        self.model = keras.models.load_model('model.keras')
        self.weights = self.model.get_weights()
        self.vectorsize = input_dim * (2 * input_dim) + (2 * input_dim)
        self.vector = []

    def randomize_vector(self):
        half_zeros = np.zeros(self.vectorsize // 2, dtype=int)
        half_ones = np.ones(self.vectorsize // 2, dtype=int)
        vector = np.concatenate((half_zeros, half_ones))
        np.random.shuffle(vector)
        self.vector = vector

    def mask_weights(self):
        self.weights = self.model.get_weights()
        d = self.input_dim
        dd = self.input_dim * 2
        for i in range(d):
            for j in range(dd):
                self.weights[0][i][j] = self.weights[0][i][j] * self.vector[(i*d)+j]
        for i in range(dd):
            self.weights[2][i % dd][0] = self.weights[2][i % dd][0] * self.vector[i + (dd * d)]
        return

    def get_vector(self):
        return self.vector

    def set_vector(self, new_vector):
        self.vector = new_vector
        return

    def get_weights(self):
        return self.weights

    def set_weights(self, new_weights):
        self.weights = new_weights
        return

    def calculate_q(self, x_train, y_train, b1, b2):
        self.model.set_weights(self.weights)
        predictions = self.model.predict(x_train, verbose=0)
        rounded_predictions = np.rint(predictions)
        conf_matrix = confusion_matrix(y_train, rounded_predictions)  # needs help
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0]

        # Calculate q value
        self.q = (FP + FN) / (b1 + b2) + 10 * (max(0, FP / b1 - 0.1) + max(0, FN / b2 - 0.1))
        return

    def mutate(self, mutation_rate):
        for i in range(len(self.vector)):
            if np.random.rand() < mutation_rate:
                if self.vector[i]:
                    self.vector[i] = 0
                else:
                    self.vector[i] = 1
        return

    def get_model(self):
        return self.model

    def repair(self):
        num_ones = np.sum(self.vector)
        extra_ones = num_ones - (len(self.vector)/2)
        if extra_ones > 0:
            ones_indices = np.where(self.vector == 1)[0]
            flip_indices = np.random.choice(ones_indices, int(abs(extra_ones)), replace=False)
            self.vector[flip_indices] = 0
        elif extra_ones < 0:
            zero_indicies = np.where(self.vector == 0)[0]
            flip_indices = np.random.choice(zero_indicies, int(abs(extra_ones)), replace=False)
            self.vector[flip_indices] = 1


