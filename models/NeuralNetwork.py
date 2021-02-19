import numpy as np

from utils import accuracy, batch


class NeuralNetwork:
    LAYER_NEURONS_INPUT = 784
    LAYER_NEURONS_OUTPUT = 10

    def __init__(self, layer1_size, layer2_size):
        self.w1 = np.zeros((layer1_size, self.LAYER_NEURONS_INPUT))
        self.b1 = np.zeros((layer1_size, 1))

        self.w2 = np.zeros((layer2_size, layer1_size))
        self.b2 = np.zeros((layer2_size, 1))

        self.w3 = np.zeros((self.LAYER_NEURONS_OUTPUT, layer2_size))
        self.b3 = np.zeros((self.LAYER_NEURONS_OUTPUT, 1))

        self.init_layers()

    def init_layers(self):
        raise NotImplementedError

    def forward(self, x):
        self.z1 = np.dot(self.w1, x) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.w2, self.a1) + self.b2
        self.a2 = self.sigmoid(self.z2)

        self.z3 = np.dot(self.w3, self.a2) + self.b3
        self.a3 = self.softmax(self.z3)
        return self.a3

    def backward(self, x, y, alpha):
        m = x.shape[1]

        dL3 = 2 * (self.a3 - y)
        dL2 = np.dot(self.w3.T, dL3) * self.sigmoid_derivative(self.z2) / m
        dL1 = np.dot(self.w2.T, dL2) * self.sigmoid_derivative(self.z1) / m

        self.w3 -= alpha * np.dot(dL3, self.a2.T)
        self.b3 -= alpha * dL3.sum(axis=1).reshape(-1, 1)

        self.w2 -= alpha * np.dot(dL2, self.a1.T)
        self.b2 -= alpha * dL2.sum(axis=1).reshape(-1, 1)

        self.w1 -= alpha * np.dot(dL1, x.T)
        self.b1 -= alpha * dL1.sum(axis=1).reshape(-1, 1)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return NeuralNetwork.sigmoid(x) * (1 - NeuralNetwork.sigmoid(x))

    @staticmethod
    def permute_data(data):
        permutation = np.random.permutation(range(len(data[0])))
        data = (np.array(data[0])[permutation], np.array(data[1])[permutation])
        return data

    def train(self, train_data, validation_data=None, batch_size=1000, epochs=10, alpha=0.1):
        for i in range(epochs):
            acc_train, acc_val = None, None
            train_data = self.permute_data(train_data)

            for x, y in zip(batch(train_data[0], batch_size), batch(train_data[1], batch_size)):
                x, y = np.hstack(x), np.hstack(y)

                self.forward(x)
                self.backward(x, y, alpha)

            # Train acc for epoch
            x_train, y_train = np.hstack(train_data[0]), np.hstack(train_data[1])
            y_hat_train = self.predict(x_train)
            acc_train = accuracy(y_train.argmax(axis=0), y_hat_train)

            # Val acc for epoch
            if validation_data:
                x_val, y_val = np.hstack(validation_data[0]), np.hstack(validation_data[1])
                y_hat_val = self.predict(x_val)
                acc_val = accuracy(y_val, y_hat_val)

            print(f"Epoch [{i + 1}/{epochs}] Accuracy train: {acc_train}, Accuracy validation: {acc_val}")

    def predict(self, x):
        return self.forward(x).argmax(axis=0)

    def predict_proba(self, x):
        return self.forward(x)
