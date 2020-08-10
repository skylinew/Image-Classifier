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
        weights = [float(0) for a in range(n+1)]
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
        weights[0] += float(1)
        for w in range(1, len(weights)):
            weights[w] += float(featuresvector[w])
    elif flag == -1:
        weights[0] -= float(1)
        for w in range(1, len(weights)):
            weights[w] -= float(featuresvector[w])




'''
        For custom features (will probably remove this function)
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
        Each feature is just the pixel's value, so a digit will have 28*28 + 1 features (the +1 is value of 1 to multiply with Weight0)
'''


def compute_features2(datalist):

    featureslist = []

    for data in datalist:
        featuresvector = []
        featuresvector.append(1)
        for height in range(len(data.pixels)):
            for width in range(len(data.pixels[height])):
                featuresvector.append(data.pixels[height][width])
        featureslist.append(featuresvector)
    return featureslist


'''
        The main perceptron algorithm
        Updates weights until all fsums of a datum/image are congruent with their respective labels (ie y =  true or false)
'''


def compute_weights(weights, featureslist, labels):

    start = time.time()
    updateflag = False
    printcount = 0

    f = 0
    while True:
                                    # number indicates seconds to timeout at
        #if time.time() >= (start + 120):
        #    break

        fsum = float(0)
        for w in range(len(weights)):

            #print(fsum)
            fsum += (float(weights[w]) * (1.0 * float(featureslist[f][w])))

        # Check if fsum is accurate by comparing it with its corresponding label

        #print('fsum, iter: ' + str(f) + ' ~~~~~~~~~~ ' + str(fsum) +' ~~~~label ~~~ ')


        if fsum < float(0) and labels[f] is True:
            # print('updating fsum: ' + str(fsum) +  ' < 0 since its true but actually false~~ ' + str(labels[f]))
            updateflag = True
            update_weights(weights, featureslist[f], 1)

        elif fsum >= float(0) and labels[f] is False:
            # print('updating fsum: ' + str(fsum) +  ' >= 0 since its false but actually true ~~ ' + str(labels[f]))
            updateflag = True
            update_weights(weights, featureslist[f], -1)


        # If you're at the last weight, reset counter f to -1
        if f == (len(featureslist)-1):
            printcount += 1
            #print('-------- ' + 'End of Round ' + str(printcount) + '  ---------')

            if updateflag is False:
                # If no updates have been made after the entire pass, then the algorithm is finished
                break
            # Otherwise, reset updateFlag
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


def sample_digits(digit, sample_percentage, datalist, labelslist):

    start = time.time()

    size = int(math.ceil(len(datalist) * (float(sample_percentage) / 100)))

    digitcount = nondigitcount = 0
    visited = []
    sampledata = []
    samplelabels = []
    cap = int(math.ceil(size / 2))
    cap2 = int(math.floor(size / 2))


    if sample_percentage == 100:

        for v in range(len(labelslist)):
            visited.append(v+1)

            if labelslist[v] == digit:
                samplelabels.append(True)
            else:
                samplelabels.append(False)


        joined_lists = list(zip(datalist, samplelabels, visited))
        random.shuffle(joined_lists)
        sampledata, samplelabels, visited = zip(*joined_lists)

        return sampledata, samplelabels, visited

    i = 0
    if 430 < cap:
        cap = 430
        cap2 = size - 430

    while i < (size - 1):

        randomindex = random.randrange(0, len(datalist))

        if labelslist[randomindex] == digit and randomindex not in visited and digitcount < cap:
            sampledata.append(datalist[randomindex])
            samplelabels.append(True)
            visited.append(randomindex)

            # print(str(labels[randomindex]) + ' ~ ' + str(True) + ' ~ '+ str(randomindex + 1))

            digitcount += 1
            i += 1

        elif labelslist[randomindex] != digit and randomindex not in visited and nondigitcount < cap2:
            sampledata.append(datalist[randomindex])
            samplelabels.append(False)
            visited.append(randomindex)

            #print(str(labels[randomindex]) + ' ~ ' + str(False) + ' ~ ' + str(randomindex + 1))

            nondigitcount += 1
            i += 1

    #print('sampleSize ' + str(sample_percentage) + ' time elapsed: ' + str(time.time() - start) + ' seconds')
    return sampledata, samplelabels, visited



'''
            If digit = -1, then validate on faces
            
'''



def validate_weights(digit, final_weights):

    if digit >= 0:
        images = samples.loadDataFile('digitdata/validationimages', 1000, 28, 28)
        labels = samples.loadLabelsFile('digitdata/validationlabels', 1000)
        features_list = compute_features2(images)
    else:
        images = samples.loadDataFile('facedata/facedatavalidation', 301, 60, 70)
        labels = samples.loadLabelsFile('facedata/facedatavalidationlabels', 301)
        features_list = compute_features2(images)

    accuracylist = []

    for i in range(len(images)):
        fsum = 0
        # for each already computed feature in the current image
        for j in range(len(features_list[i])):
            fsum += (final_weights[j] * features_list[i][j])
        # If working with digits
        if digit >= 0:
            #print(str(labels[i]) + ' --- ' + 'fsum: ' + str(fsum))

            if fsum < float(0) and labels[i] != digit:
                accuracylist.append(True)
            elif fsum >= float(0) and labels[i] == digit:
                accuracylist.append(True)
            else:
                accuracylist.append(False)
        # If working with faces
        else:
            if fsum < float(0) and labels[i] == 0:
                accuracylist.append(True)
            elif fsum >= float(0) and labels[i] == 1:
                accuracylist.append(True)
            else:
                accuracylist.append(False)

    accuracy_count = 0
    for i in range(len(accuracylist)):
        if accuracylist[i]:
            accuracy_count += 1

    accuracy = accuracy_count * 100 / len(accuracylist)
    return accuracy


def plotpoints(featureslist, labels_sample):

    for i in range(len(featureslist)):
        if labels_sample[i]:
            plt.plot(featureslist[i][1], featureslist[i][2], 'b^')
        else:
            plt.plot(featureslist[i][1], featureslist[i][2], 'rx')


def run_digits_n_times(digit, iterations, sample_percentage, trainingpath, labelspath):

    all_final_weights = []
    all_final_accuracies = []
    images = []
    labels = []

    '''
            Load either digit training data or face training data
    '''
    n_images == 5000;
    images = samples.loadDataFile(trainingpath, n_images, 28, 28)
    labels = samples.loadLabelsFile(labelspath, n_images)

    '''
            Wrap datum in Node object, store Node objects in a list, list index = datum/node's label
    '''

    '''
    
            Compute weights / run perceptron n times, where n = iterations parameter
    
    '''
    for a in range(iterations):

        # Functions_list = [pf.avg_horizontal_line_length, pf.variance_horizontal_line_length]
        images_sample, labels_sample, visited = sample_digits(digit, sample_percentage, images, labels)


        '''
                !!!! SUBJECT TO CHANGE !!!
        '''
        # Featureslist = compute_features(functions_list, images_sample, typeflag)
        featureslist = compute_features2(images_sample)
        # Weights = initialize_weights(len(functions_list), 0)
        weights = initialize_weights(28*28, 0)


        '''
                Run Perceptron
        '''
        start = time.time()
        final_weights = compute_weights(weights, featureslist, labels_sample)
        elapsed = time.time() - start



        '''
                Validate weights
        '''
        # Why was the first parameter 1???????? should be digit?? re run digits with digit instead of 1
        accuracy = validate_weights(digit, final_weights)


        ###
        print(str(digit) + ': ' + str(elapsed) + ' ~~ ' + 'sample percent: ' + str(sample_percentage) + ' ~~ ' + str(accuracy) + '%')
        basepath = './TrainingDigitsResults120/TrainingDigitsResults' + str(digit) + '/' + str(sample_percentage) + '_percent.txt'
        with open(basepath, 'w') as file:
            for weight in range(len(weights)):
                if weight == len(weights)-1:
                    file.write(str(weights[weight]) + '\n')
                else:
                    file.write(str(weights[weight]) + ' ')
            file.write(str(round(elapsed, 2)) + 's' + ' ' + str(round(float(accuracy) / float(100), 2)))


        '''
                Record computed weights and accuracy for this training iteration
        '''
        all_final_weights.append(final_weights)
        all_final_accuracies.append(accuracy)

    '''
            Record mean accuracy for all iterations
    '''

    '''
    accuracy_mean = sum(all_final_accuracies) / len(all_final_accuracies)

    print('{0}              {1}              {2}'.format(digit, sample_percentage, accuracy_mean))

    storagepath = './TrainDigitsResults/TrainingDigitsResults' + str(digit) +  '/' + str(sample_percentage) + '_percent_digit_train.txt'

    with open(storagepath, 'w') as file:
        for i in range(len(all_final_weights)):
            for j in range(len(all_final_weights[i])):
                file.write(str(all_final_weights[i][j]) + ' ')
            file.write(str(all_final_accuracies[i]) + '%' + '\n')
        file.write(str(accuracy_mean) + '%')
        file.close()

    '''
def sample_faces(sample_percentage, trainingpath, labelspath, n_images):
    faces = samples.loadDataFile(trainingpath, 451, 60, 70)
    labels = samples.loadLabelsFile(labelspath, 451)
    joinedlists = list(zip(faces, labels))
    random.shuffle(joinedlists)
    faces, labels = zip(*joinedlists)
    n_faces = int(float(sample_percentage) / float(100) * 451)

    return faces[:n_faces], labels[:n_faces]


def run_faces_n_times(iterations, sample_percentage, trainingpath, labelspath, n_images):

    all_final_weights = []
    all_final_accuracies = []

    for i in range(iterations):

        faces, labels = sample_faces(sample_percentage, trainingpath, labelspath, n_images)
        convertedlabels = []
        for q in range(len(labels)):
            if labels[q] == 1:
                convertedlabels.append(True)
            else:
                convertedlabels.append(False)

        features_list = compute_features2(faces)
        weights = initialize_weights(60*70, 0)

        start = time.time()

        final_weights = compute_weights(weights, features_list, convertedlabels)

        elapsed = time.time() - start

        #print(final_weights)
        #print('compute_weights(), time elapsed: ' + str(time.time() - start) + ' seconds')
        accuracy = validate_weights(-1, final_weights)

        all_final_weights.append(final_weights)
        all_final_accuracies.append(accuracy)

    '''
                Record mean accuracy for all iterations
    '''
    accuracy_mean = sum(all_final_accuracies) / len(all_final_accuracies)

    print('{0}      {1}     {2}'.format(sample_percentage, accuracy_mean, round(elapsed, 2)))

    '''
    if sample_percentage == 100:
        with open('./TrainFaceNoCap/SizeAccuracyTime.txt', 'a') as file:
            bestweights = all_final_weights[9]
            file.write('\n')
            for w in bestweights:
                file.write(str(w))
    
    
    '''

    '''
    storagepath = './TrainFaceResults' + '/' + str(sample_percentage) + '_percent_face_train.txt'

    with open(storagepath, 'w') as file:
        for i in range(len(all_final_weights)):
            for j in range(len(all_final_weights[i])):
                file.write(str(all_final_weights[i][j]) + ' ')
            file.write(str(all_final_accuracies[i]) + '%' + '\n')
        file.write(str(accuracy_mean) + '%')
        file.close()
    '''

'''
        Takes in a file and returns a set of weights that yield the best accuracy
'''
def choose_best_weights(file):
    content = file.readlines()
    best_accuracy = float(-1)
    index = 0


    lines = [line.split() for line in content]
    for i in range(0, len(lines)):

        if lines[i][-1] > best_accuracy:
            best_accuracy = lines[i][-1]
            index = i

    weights = [float(j) for j in lines[index][:-1]]

    #print('Chose weights with accuracy of ' + str(lines[index][-1]))
    return weights



def demo_digits(n_images, datapath, labelspath, flag):

    start = time.time()

    weights_vectors = []
    base_path = './TrainDigitsResults/TrainingDigitsResults'

    for i in range(0, 10):
        load_path = base_path + str(i) + '/100_percent_digit_train.txt'
        with open(load_path, 'r') as file:
            weights = choose_best_weights(file)
            weights_vectors.append(weights)
            file.close()


    images = samples.loadDataFile(datapath, n_images, 28, 28)
    labels = samples.loadLabelsFile(labelspath, n_images)
    featureslist = compute_features2(images)

    '''
            For each image, apply all weight vectors to image
            Keep track of each sum in a sums list for the current image,
             Choose the sum that:
                The highest value
                
            The sum's index in the sums dictionary is the same as the weight vector's index
            The weight vector's index is the same as the digit's designated/chosen weights
            This index will be what digit you are guessing the image to be
            
            Compare this index/guessed number with the labels[image] value, if they are the same, append True to results
                if false, append False to results
    '''

    results = []
    for image in range(len(images)):
        sums = []

        for weightsvector in range(len(weights_vectors)):
            sum = float(0)
            for weight in range(len(weights_vectors[weightsvector])):
                sum += (weights_vectors[weightsvector][weight] * featureslist[image][weight])
            # Add the sum to the sums dictionary
            sums.append(sum)

        max = float('-inf')
        index  = -1
        for z in range(10):
            if sums[z] > max:
                max = sums[z]
                index = z

        #print(index, end=" --> ")
        #print(labels[image])

        # Keep track of all guesses/index vs. labels in a tuples list (guess, label)
        results.append((index, labels[image]))

    correctcount = float(0)
    for t in results:
        if t[0] == t[1]:
            correctcount += float(1)


    if flag:
        rand = random.randint(0, len(results))
        print('Guessed: ' + str(results[rand][0]) + ', Actual: ' + str(results[rand][1]) + ' (line ' + str(rand) + ' of digit testlabel)')
    else:
        print(results)
        print('Digits accuracy: ' + str(round((float(correctcount) * 100 / float(len(labels))), 1)) + '%')
        print('Time elapsed: ' + str(round(time.time() - start, 2)) + 's')


def demo_faces(n_images, datapath, labelspath, flag):
    start = time.time()

    base_path = './TrainFaceResults/100_percent_face_train.txt'

    with open(base_path, 'r') as file:
        weights = choose_best_weights(file)
        file.close()

    images = samples.loadDataFile(datapath, n_images, 60, 70)
    labels = samples.loadLabelsFile(labelspath, n_images)
    featureslist = compute_features2(images)

    results = []

    for image in range(len(images)):
        sum = float(0)
        for weight in range(len(weights)):
            sum += weights[weight] * featureslist[image][weight]

        if sum >= float(0):
            results.append((True, labels[image]))
        else:
            results.append((False, labels[image]))

    correctcount = 0
    for t in results:
        if t[0] >= float(0) and t[1] == 1:
            correctcount += 1
        elif t[0] < float(0) and t[1] == 0:
            correctcount += 1

    if flag:
        rand = random.randint(0, len(results))
        actual = False
        if results[rand][1] == 1:
            actual = True
        else:
            actual = False
        print('Guessed: ' + str(results[rand][0]) + ', Actual: ' + str(actual) + ' (line ' + str(rand) + ' of face testlabel)')
    else:
        print(results)
        print('Faces accuracy: ' + str(round((float(correctcount) * 100 / float(len(labels))), 1)) + '%')
        print('Time elapsed: ' + str(round(time.time() - start, 2)) + 's')

def validate_digits120():
    start = time.time()

    weights_vectors = []
    base_path = './TrainingDigitsResults120/TrainingDigitsResults'

    for i in range(0, 10):
        load_path = base_path + str(i) + '/100_percent.txt'
        with open(load_path, 'r') as file:
            lines = file.readlines()
            weightsstrings = lines[0].split()
            weights =  [float(weight) for weight in weightsstrings]
            print(weights)
            weights_vectors.append(weights)
            file.close()

    images = samples.loadDataFile('digitdata/validationimages', 1000, 28, 28)
    labels = samples.loadLabelsFile('digitdata/validationlabels', 1000)
    featureslist = compute_features2(images)

    '''
            For each image, apply all weight vectors to image
            Keep track of each sum in a sums list for the current image,
             Choose the sum that:
                The highest value

            The sum's index in the sums dictionary is the same as the weight vector's index
            The weight vector's index is the same as the digit's designated/chosen weights
            This index will be what digit you are guessing the image to be

            Compare this index/guessed number with the labels[image] value, if they are the same, append True to results
                if false, append False to results
    '''

    results = []
    for image in range(len(images)):
        sums = []

        for weightsvector in range(len(weights_vectors)):
            sum = float(0)
            for weight in range(len(weights_vectors[weightsvector])):
                sum += (weights_vectors[weightsvector][weight] * featureslist[image][weight])
            # Add the sum to the sums dictionary
            sums.append(sum)

        max = float('-inf')
        index = -1
        for z in range(10):
            if sums[z] > max:
                max = sums[z]
                index = z

        # print(index, end=" --> ")
        # print(labels[image])

        # Keep track of all guesses vs. labels in a tuples list (guess, label)
        results.append((index, labels[image]))

    correctcount = float(0)
    for t in results:
        if t[0] == t[1]:
            correctcount += float(1)

    print('Digits accuracy: ' + str(round((float(correctcount) * 100 / float(len(labels))), 1)) + '%')


'''
    For saving the overall times and accuracies per sample size
'''
def digits_overall():

    for i in range(10):
        demo_digits(1000, 'digitdata/validationimages', 'digitdata/validationlabels')



if __name__ == "__main__":


    training_percent_cap = 100 # ie all of them, 10%, 20%, etc... 100%
    iterations_per_sample = 1



    # Paths must always refer to the files in the data.zip archive
    trainingpath = 'digitdata/trainingimages'
    labelspath = 'digitdata/traininglabels'
    n_images = 5000
    

    '''
    
    #for each digit 0 - 9 to train on
    for z in range(0, 10):
        # for each training percentage 10%, 20%, ... etc. to 100%
        for b in range(1, (training_percent_cap/10) + 1):
            #run perceptron iterations_per_sample times
            run_digits_n_times(z, iterations_per_sample, b*10, trainingpath, labelspath)
    

    '''

    #for f in range(1, (training_percent_cap/10) + 1):
    #    run_faces_n_times(iterations_per_sample, f*10, 'facedata/facedatatrain', 'facedata/facedatatrainlabels', 451)



    demo_digit_data_path = 'digitdata/validationimages'
    demo_digit_labels_path = 'digitdata/validationlabels'
    demo_faces_data_path = 'facedata/facedatavalidation'
    demo_faces_labels_path = 'facedata/facedatavalidationlabels'


    '''
    demo_digit_data_path = 'digitdata/testimages'
    demo_digit_labels_path = 'digitdata/testlabels'
    demo_faces_data_path = 'facedata/facedatatest'
    demo_faces_labels_path = 'facedata/facedatatestlabels'
    '''


    '''     - Change n_images parameter for test data
            - Current demos are set to run on validation data
            - Demo paths will change to test paths during demo
    '''
    #flag = true means 1 image only, False means all images
    print('Digits ~ ')
    demo_digits(1000, demo_digit_data_path, demo_digit_labels_path, False)
    print()
    print('Faces ~ ')
    demo_faces(150, demo_faces_data_path, demo_faces_labels_path, False)



