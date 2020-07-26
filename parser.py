import samples
import Node

training_images_digit = samples.loadDataFile('digitdata/trainingimages', 5000, 28, 28)
training_labels_digit = samples.loadLabelsFile('digitdata/traininglabels', 5000)

#training_images_face = samples.loadDataFile('facedata/facedatatrain', 451, 60, 74)
#training_labels_face = samples.loadLabelsFile('facedata/facedatatrainlabels', 451)

digit_training_list = []

# Creates a list where each element contains the list of lists of pixels, label
# number of spaces, hashtags, pluses, ratio of spaces vs. nonspaces
for im, lab in zip(training_images_digit, training_labels_digit):
    new = Node.node(im.getPixels(), lab)
    for i in new.image:
        for j in new.image[i]:
            if new.image[i][j] == '':
                new.space += 1
            elif new.image[i][j] == '+':
                new.plus += 1
            else:
                new.hashtag += 1
            new.ratio = new.space/(new.plus + new.hashtag)
    digit_training_list.append(new)

for test in digit_training_list:
    print("Ratio: " + str(test.ratio))




