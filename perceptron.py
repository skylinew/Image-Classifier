from __future__ import print_function
import samples
import random
import math
import perceptronFunctions as pf
import time


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style




'''
Flag - 0 indicates weights starting at 0, otherwise initialize weights randomly from -1/sqrt(n) to 1/sqrt(n)
Returns - list of initialized weights
'''
def initialize_weights(n, flag):
    if flag == 0:
        weights = [0 for a in range(n+1)]
    else:
        MINWEIGHT = -1/math.sqrt(n)
        MAXWEIGHT = 1/math.sqrt(n)
        weights = [random.uniform(MINWEIGHT, MAXWEIGHT) for a in range(n+1)]

    return weights

'''
To be called in compute_f_values() (the perceptron algorithm)

flags: 1 for when fsum of a single featuresvector is < 0 and the datum's label is true
      -1 for when fsum of a single featuresvector is > 0 and the datum's label is false
'''
def update_weights(weights, featuresvector, flag):
    if flag == 1:
        weights[0] += 1
        for w in range(1, len(weights)):
            weights[w] += featuresvector[w]
    elif flag == -1:
        weights[0] -= 1
        for w in range(1, len(weights)):
            weights[w] -= featuresvector[w]




'''
First function to call when starting the perceptron algorithm, pass in featureslist to compute_f_values()

Returns a list of features vectors (2d list)
functions - a list of functions
data - list of datum objects
typeflag - 0 to indicate digit data, 1 for face


featureslist is a 2d list containing feature vectors
each feature vector has the same index as its corresponding image in datalist
'''
def compute_features(functionslist, datalist, typeflag):
    featureslist = []

    for data in datalist:
        featuresvector = []
        featuresvector.append(1)
        for f in functionslist:
            featuresvector.append(f(data, typeflag))
        featureslist.append(featuresvector)
    return featureslist



'''
The main perceptron algorithm
Updates weights until all fsums of a datum/image are congruent with their respective labels (ie y =  true or false)
'''
def compute_weights(weights, featureslist, labels):

    start = time.time()


    f = 0
    updateflag = False
    printcount = 0

    while True:
                                    # number indicates seconds to timeout at
        if time.time() >= (start + 30):
            break

        fsum = 0.0
        for w in range(len(weights)):
            if w == 0:
                # weights[w] is a single features vector
                fsum += weights[w]
            else:
                #print(len(featureslist))
                #print(len(featureslist[f][w]))
                fsum += (weights[w] * featureslist[f][w])

        # Check if fsum is accurate by comparing it with its corresponding label

        print('fsum, iter: ' + str(f) + ' ~~~~~~~~~~ ' + str(fsum) +' ~~~~label ~~~ ' + str(labels_sample[f]))


        if fsum < 0 and labels[f] is True:
            print('updating fsum: ' + str(fsum) +  ' < 0 since its true but actually false')
            updateflag = True
            update_weights(weights, featureslist[f], 1)

        elif fsum >= 0 and labels[f] is False:
            print('updating fsum: ' + str(fsum) +  ' >= 0 since its false but actually true')
            updateflag = True
            update_weights(weights, featureslist[f], -1)

        print(weights)

        # if you're at the last weight, reset counter f to -1
        if f == (len(featureslist)-1):
            printcount += 1
            print('-------- ' + 'End of Round ' + str(printcount) + '  ---------')

            if updateflag is False:
                # if no updates have been made after the entire pass, then the algorithm is finished
                break
            # otherwise, reset updateFlag
            updateflag = False
            f = -1

        f += 1

    return weights



'''
Sampling function for digit data

samplepercentage: a percentage of the entire training set
digit: the digit we want to train on

Digits are randomly picked so that half of the samples are y=true digits, and the other half are y=false digits.

    eg: To train on the digit '4', half the digits chosen will be '4' and the other half will be non '4's

'''
def sample_digits(digit, samplepercentage, datalist, labelslist):
    size = int(math.ceil(len(datalist) * (float(sample_percentage) / 100)))
    print(size)
    digitcount = nondigitcount = 0
    visited = []
    sampledata = []
    samplelabels = []
    cap = int(math.ceil(size / 2))
    cap2 = int(math.floor(size / 2))
    i = 0

    while i < (size - 1):

        randomindex = random.randrange(0, len(datalist))

        if labelslist[randomindex] == digit and randomindex not in visited and digitcount < cap:
            sampledata.append(datalist[randomindex])
            samplelabels.append(True)
            visited.append(randomindex)

            print(str(labels[randomindex]) + ' ~ ' + str(True) + ' ~ '+ str(randomindex + 1))

            digitcount += 1
            i += 1

        elif labelslist[randomindex] != digit and randomindex not in visited and nondigitcount < cap2:
            sampledata.append(datalist[randomindex])
            samplelabels.append(False)
            visited.append(randomindex)

            print(str(labels[randomindex]) + ' ~ ' + str(False) + ' ~ ' + str(randomindex + 1))

            nondigitcount += 1
            i += 1


    return sampledata, samplelabels, visited


'''
Need a sample faces function as well, since for example, it wouldn't be helpful for the algorithm 
to train on 1 face and 100 non-faces
'''





def plotpoints(featureslist, labels_sample):


    for i in range(len(featureslist)):
        if labels_sample[i]:
            plt.plot(featureslist[i][1], featureslist[i][2], 'b^')
        else:
            plt.plot(featureslist[i][1], featureslist[i][2], 'rx')



if __name__ == "__main__":

    # Sample percentage being too low can incur an error, make sure to check how many image datums are being sampled
    # The closer sample percentage is to 100, the more likely it is that sample_digits() will run infinitely
    #   adjust later on so that it doesn't run infinitely
    sample_percentage = .5
    n_images = 5000
    typeflag = 0

    images = samples.loadDataFile('digitdata/trainingimages', n_images, 28, 28)
    labels = samples.loadLabelsFile('digitdata/traininglabels', n_images)

    functions_list = [pf.avg_horizontal_line_length, pf.variance_horizontal_line_length]
    images_sample, labels_sample, visited = sample_digits(1, sample_percentage, images, labels)

    featureslist = compute_features(functions_list, images_sample, typeflag)
    weights = initialize_weights(len(functions_list), 0)




    '''
            ANIMATE GRAPH
    '''

    #plotpoints(featureslist, labels_sample)
    #plt.show()



    '''
            Run Perceptron
    '''
    final_weights = compute_weights(weights, featureslist, labels_sample)


    print()
    print()
    print('==============================================')
    print()
    print('Final Weights: ', end="")
    print(final_weights)
    print()
    print('==============================================')

    print()

    # check the weights on the same sample set to make sure its legit
    print('     Checking final weights...')
    print()

    accuracylist = []

    for i in range(len(images_sample)):
        fsum = final_weights[0]

        for j in range(0, len(functions_list)):
            func = functions_list[j]
            feature = func(images_sample[i], 0)
            fsum += (feature * final_weights[j+1])

        print('Image label at line ' + str(visited[i] + 1) + ': ' + str(labels_sample[i]) + ' --- ' + 'fsum: ' + str(fsum))
        if fsum < float(0) and labels_sample[i] is False:
            accuracylist.append(True)
        elif fsum >= float(0) and labels_sample[i] is True:
            accuracylist.append(True)
        else:
            accuracylist.append(False)

    accuracy_count = 0
    for i in range(len(accuracylist)):
        if accuracylist[i]:
            accuracy_count += 1


    print()
    accuracy = accuracy_count * 100 / len(accuracylist)
    print('----- Accuracy: ' + str(accuracy) + '%' + ' ------')



