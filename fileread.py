import samples
import Node


def loaddigittraining():

    training_images_digit = samples.loadDataFile('digitdata/trainingimages', 5000, 28, 28)
    training_labels_digit = samples.loadLabelsFile('digitdata/traininglabels', 5000)

    digit_training_list = []
    for z in range(10):
        digit_training_list.append([])

    # Creates a list where each element contains the list of lists of pixels, label
    # number of spaces, hashtags, pluses, ratio of spaces vs. nonspaces
    # Parsing DIGITS
    for im, lab in zip(training_images_digit, training_labels_digit):
        new = Node.node(im.getPixels(), lab)
        for i in range(len(new.image)):
            for j in range(len(new.image[i])):
                if new.image[i][j] == 0:
                    new.space += 1
                elif new.image[i][j] == 1:
                    new.plus += 1
                else:
                    new.hashtag += 1
        new.ratio = new.space/float(new.plus + new.hashtag)
        digit_training_list[new.label].append(new)
    return digit_training_list


def loaddigitdata(size, imageloc, labelloc):

    training_images_digit = samples.loadDataFile(imageloc, size, 28, 28)
    training_labels_digit = samples.loadLabelsFile(labelloc, size)

    digit_training_list = []

    # Creates a list where each element contains the list of lists of pixels, label
    # number of spaces, hashtags, pluses, ratio of spaces vs. nonspaces
    # Parsing DIGITS
    for im, lab in zip(training_images_digit, training_labels_digit):
        new = Node.node(im.getPixels(), lab)
        for i in range(len(new.image)):
            for j in range(len(new.image[i])):
                if new.image[i][j] == 0:
                    new.space += 1
                elif new.image[i][j] == 1:
                    new.plus += 1
                else:
                    new.hashtag += 1
        new.ratio = new.space/float(new.plus + new.hashtag)
        digit_training_list.append(new)
    return digit_training_list



def digitprob(training_list, number):
    return len(training_list[number])/float(5000)


'''
#training_labels_face = samples.loadLabelsFile('facedata/facedatatrainlabels', 451)
#training_images_face = samples.loadDataFile('facedata/facedatatrain', len(training_labels_face), 60, 74)
#face_training_list = []
# Parsing FACES
for ima, labl in zip(training_images_face, training_labels_face):
    new = Node.node(ima.getPixels(), labl)
    for i in range(len(new.image)):
        for j in range(len(new.image[i])):
            if new.image[i][j] == 0:
                new.space += 1
            elif new.image[i][j] == 1:
                new.plus += 1
            else:
                new.hashtag += 1
    new.ratio = new.space/float(new.plus + new.hashtag)
    face_training_list.append(new)
'''




