from cmath import pi
import os
from time import time
from PIL import Image, ImageOps
from joblib import dump, load
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


def loadModel():
    """
    > The function loads the model from the file `model.joblib` and returns it
    :return: The model is being returned.
    """
    return load('model.joblib')


def saveModel(model):
    """
    > The function takes in a model and saves it to a file called `model.joblib`

    :param model: The model to be saved
    """
    dump(model, 'model.joblib')


def glcmEntropy(glcm):
    """
    It takes a 4D array of GLCMs and returns a 1D array of the entropy of each GLCM

    :param glcm: the gray level co-occurrence matrix
    :return: The entropy of the GLCM matrix
    """
    entropy = []
    for i in range(0, 5):
        for j in range(0, 4):
            entropy.append(shannon_entropy(glcm[:, :, i, j]))
    return entropy


def showFeatures(image, screen):
    """
    It takes an image and returns the energy, homogeneity, and entropy of the image

    :param image: The image to be analyzed
    :param screen: The screen object that contains the image
    """
    start = time()

    numpyImage = np.array(image, dtype=np.uint8)
    glcm = graycomatrix(numpyImage, distances=[
        1, 2, 4, 8, 16], angles=[0, pi/4, pi/2, 3*pi/4], levels=256)
    energy = graycoprops(glcm, 'energy')
    homogeneity = graycoprops(glcm, 'homogeneity')
    entropy = glcmEntropy(glcm)

    end = time()

    info = ""
    info += f"\nEnergy: {round(energy[0][0],2)}\n"
    info += f"Homogeneity: {round(homogeneity[0][0],2)}\n"
    info += f"Entropy: {round(entropy[0],2)}"
    info += f"\nTime: {round(end - start, 2)}"
    screen.text.set(info)


def computeFeatures(image):
    """
    It takes an image, converts it to a numpy array, calculates the grey level co-occurrence matrix, and
    then calculates the contrast, homogeneity, and entropy of the image

    :param image: The image to be analyzed
    :return: a list of the contrast, homogeneity, and entropy of the image.
    """
    numpyImage = np.array(image, dtype=np.uint8)
    glcm = graycomatrix(numpyImage, distances=[
                        1, 2, 4, 8, 16], angles=[0, pi/4, pi/2, 3*pi/4], levels=256)
    contrast = graycoprops(glcm, 'contrast')
    homogeneity = graycoprops(glcm, 'homogeneity')
    entropy = glcmEntropy(glcm)

    numpyContrast = np.hstack(contrast)
    numpyHomogeneity = np.hstack(homogeneity)

    return list(numpyContrast) + list(numpyHomogeneity) + list(entropy)


def featuresSizes(image):
    """
    It takes an image, resizes it to three different sizes, and then computes the features for each
    of those sizes

    :param image: The image to be processed
    :return: a list of features.
    """
    size128Image = image.resize((128, 128))
    size64Image = image.resize((64, 64))
    size32Image = image.resize((32, 32))

    size128features = computeFeatures(size128Image)
    size64features = computeFeatures(size64Image)
    size32features = computeFeatures(size32Image)

    features = size32features + size64features + size128features

    return features


def featuresFile(path=None, image=None):
    """
    It takes an image, equalizes it, converts it to grayscale, quantizes it into 16 and 32 colors, and
    then returns the features for each quantized image

    :param path: The path to the image file
    :param image: The image to be processed
    :return: a list of features.
    """
    if image is None:
        image = Image.open(path)

    equalizedImage = ImageOps.equalize(image)
    grayImage = equalizedImage.convert("L")
    quant16Image = grayImage.quantize(colors=16)
    quant32Image = grayImage.quantize(colors=32)
    features = featuresSizes(
        quant16Image) + featuresSizes(quant32Image)

    return features


def featuresFolder(screen):
    """
    It reads each image from the imgs folder, and then it calls the featuresFile function to get the
    features of each image

    :param screen: the screen object from the GUI
    :return: a list of features and a list of types.
    """
    url = "imgs/"
    types = []
    features = []
    imgCount = 0

    # reads each line folder from the imgs folder, containing the types
    for i in range(1, 5):
        for file in os.scandir(f"{url}{i}/"):
            if file.path.endswith(".png") and file.is_file():
                features.append(featuresFile(file.path))
                types.append(i)
                imgCount += 1
                screen.progressBar['value'] = int((imgCount/400)*100)
                screen.update_idletasks()
                screen.text.set(f"Generaring features...{imgCount}/400")
    return features, types


def computeMetrics(confusion):
    """
    It computes the mean sensibility and specificity of the confusion matrix

    :param confusion: a 3x3 matrix where the rows are the actual class and the columns are the predicted
    class
    :return: The mean sensibility and the specificity
    """
    meanSensibility = 0
    for i in range(0, 3):
        meanSensibility += confusion[i][i] / 100

    sum = 0

    for i in range(0, 3):
        for j in range(0, 3):
            if i != j:
                sum += confusion[i][j] / 300
    specificity = 1 - sum

    return (meanSensibility, specificity)


def trainModel(screen):
    """
    It takes a screen as an argument, and returns a trained model

    :param screen: The screen where the training is happening
    :return: The classifier is being returned.
    """
    start = time()
    X, y = featuresFolder(screen)
    screen.progressBar.destroy()
    X_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=.25)

    classifier = svm.SVC(kernel="linear", probability=True,
                         gamma="scale", C=1)
    screen.text.set("Training...")
    classifier.fit(X_train, y_train)

    y_prediction = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_prediction)

    confusion = confusion_matrix(y_test, y_prediction)
    (meanSensibility, specificity) = computeMetrics(confusion)

    end = time()

    totalTime = end - start

    info = ""

    info += f"{confusion}\n"
    info += f"\nAccuracy: {round(accuracy, 2)}"
    info += f"\nMean Sensibility: {round(meanSensibility, 2)}\n"
    info += f"Specificity: {round(specificity, 2)}"
    info += f"\nTime: {round(totalTime, 2)}"
    screen.text.set(info)

    return classifier
