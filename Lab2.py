import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.backend import gradients
import warnings
import imageio
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cwd = os.getcwd()
CONTENT_IMG = "Hogwarts"
STYLE_IMG = "StarryNight"
CONTENT_IMG_PATH = os.path.join(cwd, "Hogwarts.jpg")          # TODO: Add this.
STYLE_IMG_PATH = os.path.join(cwd, "StarryNight.jpg")            # TODO: Add this.


CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.1    # Alpha weight.
STYLE_WEIGHT = 1.0      # Beta weight.
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 50


# =============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''


def deprocessImage(img):
    # img = img.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def gramMatrix(x):
    if K.image_data_format == "channels_first":
        features = K.flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


# ========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
    size = style.shape[0] * style.shape[1]
    return K.sum(K.square(gramMatrix(style) - gramMatrix(gen))) / (4.0 * (3 ** 2) * (size ** 2))   # TODO: implement.


def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalLoss(contentLoss, styleLoss):
    return CONTENT_WEIGHT * contentLoss + STYLE_WEIGHT * styleLoss   # TODO: implement.


# =========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH, target_size=(CONTENT_IMG_H, CONTENT_IMG_W))
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH, target_size=(STYLE_IMG_H, STYLE_IMG_W))
    print("      Images have been loaded.")
    return (cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W)


def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


'''
TODO: A lot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and de-processed images.
'''


def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)

    model = vgg19.VGG19(include_top=False, weights="imagenet", input_tensor=inputTensor)   # TODO: implement.
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")

    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"

    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    cLoss = contentLoss(contentOutput, genOutput)   # TODO: implement.

    sLoss = 0.0
    print("   Calculating style loss.")
    for layerName in styleLayerNames:
        styleLayer = outputDict[layerName]
        styleOutput = styleLayer[1, :, :, :]
        genOutput = styleLayer[2, :, :, :]
        sLoss += styleLoss(styleOutput, genOutput)   # TODO: implement.
    sLoss /= len(styleLayerNames)

    tLoss = totalLoss(cLoss, sLoss)   # TODO: implement.

    # TODO: Setup gradients or use K.gradients().
    gradients = K.gradients(tLoss, genTensor)[0]
    fetchLossAndGrads = K.function([genTensor], [tLoss, gradients])

    class Evaluator(object):
        def __init__(self):
            self.loss_value = None
            self.grad_values = None

        def loss(self, x):
            assert self.loss_value is None
            x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
            outs = fetchLossAndGrads([x])
            loss_value = outs[0]
            grad_values = outs[1].flatten().astype('float64')
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values

    evaluator = Evaluator()

    tData = tData.flatten()
    print("   Beginning transfer.")
    for i in range(1, TRANSFER_ROUNDS + 1):
        print("   Step %d." % i)

        # TODO: perform gradient descent using fmin_l_bfgs_b.
        tData, tLoss, info = fmin_l_bfgs_b(evaluator.loss, tData, fprime=evaluator.grads, maxfun=20, maxiter=1300)
        print("      Loss: %f." % tLoss)

        img = tData.copy().reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
        img = deprocessImage(img)
        temp = CONTENT_IMG + "_" + STYLE_IMG + "_%d.png"
        saveFile = temp % i   # TODO: Implement.
        imageio.imwrite(saveFile, img)   # Uncomment when everything is working right.
        print("      Image saved to \"%s\"." % saveFile)

    print("   Transfer complete.")


# =========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    print("Preprocessed Raw Data")
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")


if __name__ == "__main__":
    main()
