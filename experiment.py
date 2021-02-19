from itertools import product

import numpy as np
import pandas as pd

from models import NeuralNetworkNormal, NeuralNetworkNormalScaled, NeuralNetworkUniform
from utils import accuracy
from utils.mnist_loader import load_data_wrapper

training_data, validation_data, test_data = load_data_wrapper('data/mnist.pkl.gz')

MODELS = [NeuralNetworkNormal, NeuralNetworkNormalScaled, NeuralNetworkUniform]
LAYERS = [
    (100, 100),
    (200, 100),
    (300, 100),
    (300, 200),
    (500, 100),
    (500, 200),
    (500, 300)
]

BATCH_SIZE = 100
EPOCHS = 40
ALPHA = 0.05

results = []

for model, (layer1, layer2) in product(MODELS, LAYERS):
    nn = model(layer1, layer2)
    print(f"Model: {model}: ({layer1})({layer2})")
    nn.train(training_data, validation_data, batch_size=BATCH_SIZE, epochs=EPOCHS, alpha=ALPHA)

    x_train, y_train = np.hstack(training_data[0]), np.hstack(training_data[1])
    x_test, y_test = np.hstack(test_data[0]), np.hstack(test_data[1])
    y_training_hat = nn.predict(x_train)
    y_test_hat = nn.predict(x_test)

    res = [
        model,
        layer1,
        layer2,
        accuracy(y_train.argmax(axis=0), y_training_hat),
        accuracy(y_test, y_test_hat)
    ]
    print(res)
    results.append(res)

results = pd.DataFrame(results, columns=['model', 'layer_1_size', 'layer_2_size', 'train_acc', 'test_acc'])
results.to_csv('results/experiment.csv', index=False)
