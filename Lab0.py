
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import math
from sklearn.metrics import f1_score
import pandas as pd


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
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate=0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # Activation function.
    def __sigmoid(self, x):
        sigmoid_v = np.vectorize(self.sigmoid)
        return sigmoid_v(x)
        # TODO: implement

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        sig = self.__sigmoid(x)   # TODO: implement
        return sig * (1 - sig)

    # Loss function
    def mse(self, x, y):
        return ((x - y) ** 2) / 2

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i: i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs=100000, minibatches=True, mbs=100):

        mse_v = np.vectorize(self.mse)
        n = mbs

        for epoch in range(epochs):
            for i in range(0, len(xVals), n):
                inp = xVals[i: i + n]
                oup = yVals[i: i + n]
                L1_out, L2_out = self.__forward(inp)
                y_hat = L2_out.copy()
                # loss = mse_v(oup, y_hat)
                loss_prime = y_hat - oup

                list2 = []

                for j in range(len(y_hat)):
                    temp1 = y_hat[j]
                    temp2 = loss_prime[j]
                    mult = np.ones(len(temp1)) - temp1

                    temp = temp1 * mult
                    temp = temp * temp2
                    list2.append(temp)

                adj2 = np.array(list2)
                upd2 = (np.dot(np.transpose(L1_out), adj2)) / n

                h_error = np.dot(adj2, np.transpose(self.W2))

                list1 = []

                for j in range(len(L1_out)):
                    temp1 = L1_out[j]
                    temp2 = h_error[j]
                    mult = np.ones(len(temp1)) - temp1

                    temp = temp1 * mult
                    temp = temp * temp2
                    list1.append(temp)

                adj1 = np.array(list1)
                upd1 = (np.dot(np.transpose(inp), adj1)) / n

                self.W1 -= self.lr * upd1
                self.W2 -= self.lr * upd2


        pass
        # TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        b = np.zeros_like(layer2)
        b[np.arange(len(layer2)), layer2.argmax(1)] = 1
        return b


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


#=========================<Pipeline Functions>==================================


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
    ((xTrain, yTrain), (xTest, yTest)) = raw            # TODO: Add range reduction here (0-255 ==> 0.0-1.0).
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
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        # print("Not yet implemented.")                   # TODO: Write code to build and train your custom neural net.
        nn = NeuralNetwork_2Layer(784, 10, 50, 2.0)
        nn.train(xTrain, yTrain, 20)
        return nn
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        # print("Not yet implemented.")                   # TODO: Write code to build and train your keras neural net.
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                            tf.keras.layers.Dense(512, activation=tf.nn.sigmoid),
                                            tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.fit(xTrain, yTrain, epochs=10)
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


def evalResults(data, preds):   # TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()
    score = []
    for i in range(10):
        temp = [0] * 11
        score.append(temp)
    for i in range(preds.shape[0]):
        act = yTest[i]
        pred = preds[i]
        num = 0
        for val in act:
            if val == 1:
                break
            num += 1
        pred_num = 0
        for val in pred:
            if val == 1:
                break
            pred_num += 1
        if num == pred_num:
            score[num][num] += 1
        else:
            score[pred_num][num] += 1
    for i in range(10):
        temp = sum(score[i])
        score[i][10] = temp
    temp = []
    for i in range(11):
        add = 0
        for j in range(10):
            add += score[j][i]
        temp.append(add)
    score.append(temp)
    score = np.array(score)
    print("Confusion Matrix:")
    print()
    print(score)
    print()
    print("F1 Score Matrix:")
    print(f1_score(yTest, preds, average=None))


#=========================<Main>================================================


def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)


if __name__ == '__main__':
    main()
