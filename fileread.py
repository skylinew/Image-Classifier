import samples
import Node
import random
import math


def percentageoftraining(traininglist, cats, percent):
    newlist = []
    for i in range(cats):
        newlist.append([])
    for j in range(cats):
        for k in range(int(math.floor(percent * len(traininglist[j])))):
            index = random.randrange(0, len(traininglist[j]))
            newlist[j].append(traininglist[j][index])
    return newlist




def loadtraining(n, imageloc, labelloc, x, y, cats, percent):

    training_images_digit = samples.loadDataFile(imageloc, n, x, y)
    training_labels_digit = samples.loadLabelsFile(labelloc, n)

    digit_training_list = []
    for z in range(cats):
        digit_training_list.append([])

    # Creates a list where each element contains the list of lists of pixels, label
    # number of spaces, hashtags, pluses, ratio of spaces vs. nonspaces
    # Parsing DIGITS
    for im, lab in zip(training_images_digit, training_labels_digit):
        new = Node.node(im.getPixels(), lab)
        digit_training_list[new.label].append(new)

    if percent != 1:
        digit_training_list = percentageoftraining(digit_training_list, cats, percent)
    return digit_training_list


def loadtest(n, imageloc, labelloc, x, y):

    training_images_digit = samples.loadDataFile(imageloc, n, x, y)
    training_labels_digit = samples.loadLabelsFile(labelloc, n)

    digit_training_list = []

    # Creates a list where each element contains the list of lists of pixels, label
    # number of spaces, hashtags, pluses, ratio of spaces vs. nonspaces
    # Parsing DIGITS
    for im, lab in zip(training_images_digit, training_labels_digit):
        new = Node.node(im.getPixels(), lab)
        digit_training_list.append(new)
    return digit_training_list


def digitprob(training_list, number, percent):
    return len(training_list[number])/float(math.floor(percent * 5000))


def faceprob(training_list, face, percent):
    return len(training_list[face])/float(math.floor(percent * 451))
