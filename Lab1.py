
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model

random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ALGORITHM = "guesser"
ALGORITHM = "tf_net"
# ALGORITHM = "tf_conv"

DATASET = "mnist_d"
# DATASET = "mnist_f"
# DATASET = "cifar_10"
# DATASET = "cifar_100_f"
# DATASET = "cifar_100_c"

if DATASET == "mnist_d":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "cifar_10":
    # pass                                 # TODO: Add this case.
    NUM_CLASSES = 10
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
elif DATASET == "cifar_100_f":
    # pass                                 # TODO: Add this case.
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
elif DATASET == "cifar_100_c":
    # pass                                 # TODO: Add this case.
    NUM_CLASSES = 20
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072

#=========================<Classifier Functions>================================


def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def buildTFNeuralNet(x, y, eps = 6):
    # pass        # TODO: Implement a standard ANN here.
    model = keras.models.Sequential([keras.layers.Flatten(),
                                        keras.layers.Dense(512, activation=tf.nn.sigmoid),
                                        keras.layers.Dense(NUM_CLASSES, activation=tf.nn.sigmoid)])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x, y, epochs=10)
    return model


def buildTFConvNet(x, y, dataset, load = True, eps = 10, dropout = True, dropRate = 0.2):
    # pass        # TODO: Implement a CNN here. dropout option is required.
    if load:
        parent_dir = os.getcwd()
        curr = DATASET + '/model'
        final_path = os.path.join(parent_dir, curr)
        model = load_model(final_path)
        return model

    model = Sequential()
    inShape = (IH, IW, IZ)
    lossType = keras.losses.categorical_crossentropy

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=inShape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=inShape))
    model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(dropRate))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(Dropout(dropRate))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(dropRate))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(dropRate))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    if dropout:
        model.add(Dropout(dropRate))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer='adam', loss=lossType)
    model.fit(x, y, epochs=eps)

    return model

#=========================<Pipeline Functions>==================================


def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        # pass      # TODO: Add this case.
        cifar = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data()
    elif DATASET == "cifar_100_f":
        # pass      # TODO: Add this case.
        cifar = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data(label_mode='fine')
    elif DATASET == "cifar_100_c":
        # pass      # TODO: Add this case.
        cifar = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data(label_mode='coarse')
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))


def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain, DATASET, False)
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()
    return accuracy


def saveModel(model, acc, dir):
    acc_file = os.path.join(dir, 'acc')
    mod_file = os.path.join(dir, 'model')
    if not os.path.exists(acc_file):
        acc_f = open(acc_file, 'w')
        acc_f.write(str(acc))
        acc_f.close()
        save_model(model, mod_file)
    else:
        acc_f = open(acc_file, 'r+')
        old_acc = acc_f.read()
        temp = float(old_acc)
        if temp < acc:
            acc_f.seek(0)
            acc_f.write(str(acc))
            acc_f.truncate()
            acc_f.close()
            save_model(model, mod_file, overwrite=True)


#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    acc = evalResults(data[1], preds)

    if ALGORITHM == 'tf_conv':
        parent_dir = os.getcwd()
        curr_dir = DATASET
        final_dir = os.path.join(parent_dir, curr_dir)
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
        saveModel(model, acc, final_dir)
    else:
        parent_dir = os.getcwd()
        curr_dir = DATASET + '_ann'
        final_dir = os.path.join(parent_dir, curr_dir)
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
        saveModel(model, acc, final_dir)


if __name__ == '__main__':
    main()
