import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import math
from sklearn.metrics import f1_score, confusion_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
ALGORITHM = "custom_net"
# ALGORITHM = "tf_net"


class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate, layers=2, act=0):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.weights = [np.random.randn(self.inputSize, self.neuronsPerLayer)]
        for i in range(layers - 2):
            self.weights.append(np.random.randn(self.neuronsPerLayer, self.neuronsPerLayer))
        self.weights.append(np.random.randn(self.neuronsPerLayer, self.outputSize))
        self.layers = layers
        self.activation = act

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def relu(self, x):
        return max(0, x)

    # Activation function.
    def __sigmoid(self, x):
        sigmoid_v = np.vectorize(self.sigmoid)
        return sigmoid_v(x)
        # TODO: implement

    def __relu(self, x):
        relu_v = np.vectorize(self.relu)
        return relu_v(x)

    # Activation prime function.
    def __sigmoidDerivative(self, x):  # TODO: implement
        return x * (1 - x)

    def __reluDerivative(self, x):
        if x > 0:
            return 1
        return 0

    # Loss function
    def mse(self, x, y):
        return ((x - y) ** 2) / 2

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i: i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs=100000, minibatches=True, mbs=100):

        n = mbs
        if not minibatches:
            n = 1
        if self.activation == 0:
            act = np.vectorize(self.__sigmoidDerivative)
        else:
            act = np.vectorize(self.__reluDerivative)

        for epoch in range(epochs):
            for i in range(0, len(xVals), n):
                inp = xVals[i: i + n]
                oup = yVals[i: i + n]
                layers = self.__forward(inp)

                o_out = layers[len(layers) - 1]
                o_error = o_out - oup

                adjL = []

                for j in range(len(o_out)):
                    temp1 = o_out[j]
                    temp2 = o_error[j]
                    temp = act(temp1) * temp2
                    adjL.append(temp)

                o_delta = np.array(adjL)
                upd = (np.dot(np.transpose(layers[len(layers) - 2]), o_delta)) / n
                updates = [upd]

                k = len(layers) - 1

                while k > 1:

                    h_error = np.dot(o_delta, np.transpose(self.weights[k]))
                    h_out = layers[k - 1]

                    adjL = []

                    for j in range(len(h_out)):
                        temp1 = h_out[j]
                        temp2 = h_error[j]
                        temp = act(temp1) * temp2
                        adjL.append(temp)

                    o_delta = np.array(adjL)
                    upd = (np.dot(np.transpose(layers[k - 2]), o_delta)) / n
                    updates.append(upd)
                    k -= 1

                h_error = np.dot(o_delta, np.transpose(self.weights[k]))
                h_out = layers[k - 1]

                adjL = []

                for j in range(len(h_out)):
                    temp1 = h_out[j]
                    temp2 = h_error[j]
                    temp = act(temp1) * temp2
                    adjL.append(temp)

                o_delta = np.array(adjL)
                upd = (np.dot(np.transpose(inp), o_delta)) / n
                updates.append(upd)

                updates.reverse()

                for j in range(len(self.weights)):
                    self.weights[j] -= self.lr * updates[j]

        pass
        # TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.

    # Forward pass.
    def __forward(self, input):
        layers = []
        temp = input
        for i in range(self.layers):
            if self.activation == 0:
                layer = self.__sigmoid(np.dot(temp, self.weights[i]))
            else:
                layer = self.__relu(np.dot(temp, self.weights[i]))
            layers.append(layer)
            temp = layer
        return layers

    # Predict.
    def predict(self, xVals):
        layers = self.__forward(xVals)
        layer = layers[len(layers) - 1]
        b = np.zeros_like(layer)
        b[np.arange(len(layer)), layer.argmax(1)] = 1
        return b


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


# =========================<Pipeline Functions>==================================


def range_reduce(x):
    return (x * 1.0) / 255


def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw  # TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    range_v = np.vectorize(range_reduce)
    train = []
    test = []
    for x in xTrain:
        x = x.flatten()
        x = range_v(x)
        train.append(x)
    train = np.array(train)
    for x in xTest:
        x = x.flatten()
        x = range_v(x)
        test.append(x)
    test = np.array(test)
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((train, yTrainP), (test, yTestP))


def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None  # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        # print("Not yet implemented.")                   # TODO: Write code to build and train your custom neural net.
        nn = NeuralNetwork_2Layer(784, 10, 50, 2.0, 2)
        nn.train(xTrain, yTrain, 20)
        return nn
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        # print("Not yet implemented.")                   # TODO: Write code to build and train your keras neural net.
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                            tf.keras.layers.Dense(512, activation=tf.nn.sigmoid),
                                            tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.fit(xTrain, yTrain, epochs=15)
        return model
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        # print("Not yet implemented.")                   # TODO: Write code to run your custom neural net.
        pred = model.predict(data)
        return pred
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        # print("Not yet implemented.")                   # TODO: Write code to run your keras neural net.
        preds = model.predict(data)
        b = np.zeros_like(preds)
        b[np.arange(len(preds)), preds.argmax(1)] = 1
        return b
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds):  # TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(yTest.argmax(axis=1), preds.argmax(axis=1)))
    print()
    print("F1 Score Matrix:")
    print(f1_score(yTest, preds, average=None))


# =========================<Main>================================================


def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)

    # Iris Dataset
    # TODO: Uncomment below code to test Custom NN on Iris dataset

    # iris = datasets.load_iris()
    # X = iris.data
    # Y = iris.target
    # xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2)
    # yTrain = to_categorical(yTrain, 3)
    # yTest = to_categorical(yTest, 3)
    #
    # nn = NeuralNetwork_2Layer(4, 3, 150, 1.0, 2)
    # nn.train(xTrain, yTrain, 20, True, 10)
    # preds = nn.predict(xTest)
    # data = (xTest, yTest)
    # evalResults(data, preds)


if __name__ == '__main__':
    main()
