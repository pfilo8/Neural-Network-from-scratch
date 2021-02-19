import numpy as np

from .NeuralNetwork import NeuralNetwork


class NeuralNetworkNormal(NeuralNetwork):

    def init_layers(self):
        self.w1 = np.random.randn(*self.w1.shape)
        self.b1 = np.random.randn(*self.b1.shape)

        self.w2 = np.random.randn(*self.w2.shape)
        self.b2 = np.random.randn(*self.b2.shape)

        self.w3 = np.random.randn(*self.w3.shape)
        self.b3 = np.random.randn(*self.b3.shape)


class NeuralNetworkNormalScaled(NeuralNetwork):

    def init_layers(self):
        scaling_factor = 0.1
        self.w1 = scaling_factor * np.random.randn(*self.w1.shape)
        self.b1 = scaling_factor * np.random.randn(*self.b1.shape)

        self.w2 = scaling_factor * np.random.randn(*self.w2.shape)
        self.b2 = scaling_factor * np.random.randn(*self.b2.shape)

        self.w3 = scaling_factor * np.random.randn(*self.w3.shape)
        self.b3 = scaling_factor * np.random.randn(*self.b3.shape)


class NeuralNetworkUniform(NeuralNetwork):

    def init_layers(self):
        self.w1 = np.random.uniform(-1, 1, self.w1.shape)
        self.b1 = np.random.uniform(-1, 1, self.b1.shape)

        self.w2 = np.random.uniform(-1, 1, self.w2.shape)
        self.b2 = np.random.uniform(-1, 1, self.b2.shape)

        self.w3 = np.random.uniform(-1, 1, self.w3.shape)
        self.b3 = np.random.uniform(-1, 1, self.b3.shape)
