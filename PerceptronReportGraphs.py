from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import samples
import time
import perceptron
from scipy.stats import norm
import statistics
import math
import random

def demo_faces(weights):
    n_images = 301
    start = time.time()

    data_path = 'facedata/facedatavalidation'
    labels_path = 'facedata/facedatavalidationlabels'


    images = samples.loadDataFile(data_path, n_images, 60, 70)
    labels = samples.loadLabelsFile(labels_path, n_images)
    featureslist = perceptron.compute_features2(images)

    results = []

    for image in range(len(images)):
        sum = 0
        for weight in range(len(weights)):
            sum += weights[weight] * featureslist[image][weight]

        results.append((sum, labels[image]))

    correctcount = 0
    for t in results:
        if t[0] >= float(0) and t[1] == 1:
            correctcount += 1
        elif t[0] < float(0) and t[1] == 0:
            correctcount += 1


    return (float(correctcount) * 100 / float(len(labels)))

def compute_weights(weights, featureslist, labels):

    start = time.time()
    updateflag = False
    printcount = 0

    f = 0
    while True:
                                    # number indicates seconds to timeout at
        if time.time() >= (start + 5):
            break

        fsum = float(0)
        for w in range(len(weights)):

            #print(fsum)
            fsum += (float(weights[w]) * (float(featureslist[f][w])))

        # Check if fsum is accurate by comparing it with its corresponding label

        #print('fsum, iter: ' + str(f) + ' ~~~~~~~~~~ ' + str(fsum) +' ~~~~label ~~~ ')


        if fsum < float(0) and labels[f] is True:
            # print('updating fsum: ' + str(fsum) +  ' < 0 since its true but actually false~~ ' + str(labels[f]))
            updateflag = True
            perceptron.update_weights(weights, featureslist[f], 1)

        elif fsum >= float(0) and labels[f] is False:
            # print('updating fsum: ' + str(fsum) +  ' >= 0 since its false but actually true ~~ ' + str(labels[f]))
            updateflag = True
            perceptron.update_weights(weights, featureslist[f], -1)


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



def demo_digits(weights_vectors):
    n_images = 1000
    datapath = 'digitdata/validationimages'
    labelspath = 'digitdata/validationlabels'

    images = samples.loadDataFile(datapath, n_images, 28, 28)
    labels = samples.loadLabelsFile(labelspath, n_images)
    featureslist = perceptron.compute_features2(images)


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

    return round((float(correctcount) * 100 / float(len(labels))), 2)


def graph_face_results():
    base_path_faces = './TrainFaceResults/'

    fig1, ax1 = plt.subplots()
    ax1.set_title('Face Training Accuracies per Sample Size')
    datas = []
    means = []

    for i in range(1, 11):
        path = base_path_faces + str(i*10) + '_percent_face_train.txt'

        with open(path, 'r') as file:
            lines = file.read().split('\n')
            data = []

            for line in range(len(lines)-1):
                percentstring = lines[line].split()[-1][:-1]
                data.append(float(percentstring) / 100.0)
            datas.append(data)

            means.append(float(lines[-1][:-1]) / 100)

            file.close()
    print(means)
    plt.plot([1,2,3,4,5,6,7,8,9, 10], means,linestyle='--', marker='o', label='Mean Accuracies')
    plt.boxplot(datas)
    plt.xticks([1,2,3,4,5,6,7,8,9, 10], ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    plt.xlabel('Training Data Sample Size')
    plt.ylabel('Accuracy of Computed Weights')
    plt.legend(loc='upper left');
    plt.show()




def graph_digit_results():
    base_path_digits = './TrainDigitsResults/TrainingDigitsResults'
    colors = ['r', 'b', 'g', 'orange', 'k', 'c', 'm', 'y', 'grey', 'pink']
    markers = ['x','d','s','o','v','p','*','1','2','3']
    datas = [] #each index is a digit's training results (ie list of accuracies for each percentage)

    std_devs = []


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Standard Deviation Digit Training Accuracies')
    for i in range(0, 10):

        data = []
        for j in range(1, 11):
            path = base_path_digits + str(i) + '/' +  str(j*10) + '_percent_digit_train.txt'
            with open(path, 'r') as file:
                lines = file.read().split('\n')
                accuracy = lines[-1][:-1]
                data.append(int(accuracy))
                file.close()
        # append accuracy
        datas.append(data)

        # add standard deviation of accuracies for all training sample size for digit i
        std_devs.append(round(statistics.stdev(data), 2))

        #ax1.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], datas[i], c=colors[i], marker='.',linestyle='--', label=str(i))


    plt.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], std_devs)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.xlabel('Digit Trained On')
    plt.ylabel('Accuracy Standard Deviation')
    #plt.legend(loc='lower right', title='Digits Trained On');
    plt.show()


def demo_digits_graph(n_images, datapath, labelspath):
    start = time.time()

    weights_vectors = []
    base_path = './TrainDigitsResults/TrainingDigitsResults'

    for i in range(0, 10):
        load_path = base_path + str(i) + '/100_percent_digit_train.txt'
        with open(load_path, 'r') as file:
            weights = perceptron.choose_best_weights(file)
            weights_vectors.append(weights)
            file.close()

    images = samples.loadDataFile(datapath, n_images, 28, 28)
    labels = samples.loadLabelsFile(labelspath, n_images)
    featureslist = perceptron.compute_features2(images)

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

    '''
    correctcount = float(0)
    for t in results:
        if t[0] == t[1]:
            correctcount += float(1)
    '''
    #print('Digits accuracy: ' + str(round((float(correctcount) * 100 / float(len(labels))), 1)) + '%')


    corrects = [[] for i in range(10)]
    incorrects = [[] for i in range(10)]

    for z in range(len(results)):
        if results[z][0] == results[z][1]:
            corrects[results[z][1]].append(True)
        else:
            incorrects[results[z][1]].append(True)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Digit Validation with Best Weights')

    #ax1.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], datas[i], c=colors[i], marker='.', linestyle='--', label=str(i))

    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    plt.xlabel('Training Data Sample Size')
    plt.ylabel('Accuracy of Computed Weights')
    plt.legend(loc='lower right', title='Digits Trained On');
    plt.show()


def graphdigits120():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Digit Training Times by Sample Size')
    colors = ['r', 'b', 'g', 'orange', 'k', 'c', 'm', 'y', 'grey', 'pink']

    basepath = './TrainingDigitsResults120/TrainingDigitsResults'

    # first index refers to TrainingDigitsResultsX and second index refers to Y_percent.txt runtime
    avgtrainingtimes = [[] for i in range(10)]

    for i in range(10):
        path = basepath + str(i)
        times = []
        for j in range(1,11):
            path = basepath + str(i) + '/' + str(j*10) + '_percent.txt'
            t = 0
            with open(path, 'r') as file:
                lines = file.readlines()
                t = float(lines[-1].split()[0][:-1])
                file.close()
            avgtrainingtimes[i].append(t)

    print(avgtrainingtimes)
    for i in range(10):
        ax1.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], avgtrainingtimes[i], c=colors[i], marker='.', linestyle='--', label=str(i))

    '''
    meanruntimes = []
    for i in range(len(avgtrainingtimes)):
        meanruntimes.append(statistics.mean(avgtrainingtimes[i]))

    print(meanruntimes)
    standard_dev = statistics.stdev(meanruntimes)

    print(standard_dev)
    '''


    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    plt.xlabel('Training Data Sample Size')
    plt.ylabel('Runtime in Seconds')
    plt.legend(loc='lower right', title='Digit Trained on');
    plt.show()



def graphfacetimes():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Face Training Runtimes')
    colors = ['r', 'b', 'g', 'orange', 'k', 'c', 'm', 'y', 'grey', 'pink']

    runtimes = []

    with open('./TrainFaceNoCap/SizeAccuracyTime.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            time = float(line.split()[-1])
            runtimes.append(time)
        file.close()

    print(runtimes)
    ax1.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], runtimes, c=colors[5], marker='.', linestyle='--',)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    plt.xlabel('Training Data Sample Size')
    plt.ylabel('Runtime in Seconds')
    #plt.legend(loc='lower right', title='Face Train Runtime');
    plt.show()




def get_digit_acc():

    # list of weight vectors
    weights10 = [[] for i in range(10)] #weights10[0] has weights vector for 0 digit at 10%
    weights20 = [[] for i in range(10)] #weights20[0] has weights vector for 0 digit at 20%
    weights30 = [[] for i in range(10)]
    weights40 = [[] for i in range(10)]
    weights50 = [[] for i in range(10)]
    weights60 = [[] for i in range(10)]
    weights70 = [[] for i in range(10)]
    weights80 = [[] for i in range(10)]
    weights90 = [[] for i in range(10)]
    weights100 = [[] for i in range(10)]


    allweights = []
    for i in range(10):
        allweights.append([])

    # first index of allweights refers to sample size
    # second index of allweights refers to list of weights vectors
    # third index of allweights refers to a specific weight vector (list) ordered by digit

    #for r in range(10):
    #    allweights[i]

    basepath = 'TrainingDigitsResults120/TrainingDigitsResults'

    for i in range(10): # for each folder TrainingDigitsResultsX
        for j in range(1, 11): # for each Y_percent.txt
            weights = []
            with open(basepath + str(i) + '/' + str(j*10) + '_percent.txt') as file:
                line = file.readline()
                tokens = line.split()
                weights = [float(token) for token in tokens]
                file.close()

            allweights[j-1].append(weights)
    #print(allweights)


    for k in range(10): #for each percentage / set of weight vectors (allweights[j] = all_weights 2d list)
            demo_digits((k + 1)*10, allweights[k])


def run_digits_n_times():

    all_final_stddevs = []
    all_final_accs = []

    colors = ['r', 'b', 'g', 'orange', 'k', 'c', 'm', 'y', 'grey', 'pink']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Face Training Runtimes')

    images = samples.loadDataFile('digitdata/trainingimages', 5000, 28, 28)
    labels = samples.loadLabelsFile('digitdata/traininglabels', 5000)

    # For each percentage
    for v in range(1, 11):
        sample_percentage = v*10
        acc_list = []
        # For all 5 iterations
        for j in range(0, 5):

            weights_vectors = []
            # For each digit 0-9
            for y in range(0, 10):
                digit = y

                images_sample, labels_sample, visited = perceptron.sample_digits(digit, sample_percentage, images, labels)


                # Featureslist = compute_features(functions_list, images_sample, typeflag)
                featureslist = perceptron.compute_features2(images_sample)
                # Weights = initialize_weights(len(functions_list), 0)
                weights = perceptron.initialize_weights(28 * 28, 0)

                '''
                        Run Perceptron / Learn weights
                '''
                start = time.time()
                final_weights = compute_weights(weights, featureslist, labels_sample)
                elapsed = time.time() - start
                weights_vectors.append(final_weights)


            # Test all weights (weight vectors 0-9), add acc to acc list
            acc = demo_digits(weights_vectors)
            acc_list.append(acc)


        # For percentage sample size v, get the mean accuracy and standard deviation accuracy
        mean = statistics.mean(acc_list)
        stddev = statistics.stdev(acc_list)
        print('Mean accuracy for sample percent-' + str(v) + ' ~ ' + str(mean))
        print('STD Dev for sample percent-' + str(v) + ' ~ ' + str(stddev))

        all_final_stddevs.append(stddev)
        all_final_accs.append(mean)

    print(stddev)
    ax1.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], all_final_accs, color='blue', marker='.', linestyle='--')
    #ax1.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], all_final_stddevs, color='red',marker='.', linestyle='--')
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    plt.xlabel('Training Data Sample Size')
    plt.ylabel('Standard Deviation of Accuracies')
    # plt.legend(loc='lower right', title='Face Train Runtime');
    plt.show()







def get_digit_acc2():
    images = samples.loadDataFile('digitdata/trainingimages', 5000, 28, 28)
    labels = samples.loadLabelsFile('digitdata/traininglabels', 5000)

    accuracies = []
    times = []


    def convertlabelslist(digit, labelslist):
        booleanlist = [False for i in range(len(labelslist))]
        for i in range(len(labelslist)):
            if labelslist[i] == digit:
                booleanlist[i] = True
            else:
                booleanlist[i] = False
        return booleanlist

    # For each percentage
    for i in range(10):
        sample_percentage = float((i+1)/ 10.0)
        sample_size = int(math.floor(sample_percentage * float(len(labels))))

        joinedlists = list(zip(images, labels))
        images_sample = []
        labels_sample = []

        # Shuffle until sample has all 9 digits
        while True:

            joinedlists = list(zip(images, labels))
            random.shuffle(joinedlists)
            images_sample, labels_sample = zip(*joinedlists)
            images_sample = images_sample[:sample_size]
            labels_sample = labels_sample[:sample_size]
            #print(labels_sample)
            if 0 in labels_sample and 1 in labels_sample and 2 in labels_sample\
                    and 3 in labels_sample and 4 in labels_sample and 5 in labels_sample\
                    and 6 in labels_sample and 7 in labels_sample and 8 in labels_sample\
                    and 9 in labels_sample:
                break

        # Have 10 different labels list for each digit
        # Convert labelslist into true/false instead of numbers
        labelslist0 = convertlabelslist(0, labels_sample)
        labelslist1 = convertlabelslist(1, labels_sample)
        labelslist2 = convertlabelslist(2, labels_sample)
        labelslist3 = convertlabelslist(3, labels_sample)
        labelslist4 = convertlabelslist(4, labels_sample)
        labelslist5 = convertlabelslist(5, labels_sample)
        labelslist6 = convertlabelslist(6, labels_sample)
        labelslist7 = convertlabelslist(7, labels_sample)
        labelslist8 = convertlabelslist(8, labels_sample)
        labelslist9 = convertlabelslist(9, labels_sample)

        all_labels = [labelslist0, labelslist1, labelslist2, labelslist3,
                      labelslist4, labelslist5, labelslist6, labelslist7,
                      labelslist8, labelslist9]

        start = time.time()

        # Compute weight vectors for all digits 0-9
        all_weight_vectors = []
        for j in range(10):
            featureslist = perceptron.compute_features2(images_sample)
            weights = perceptron.initialize_weights(28*28, 0)
            computed_weights = perceptron.compute_weights(weights, featureslist, all_labels[j])
            all_weight_vectors.append((computed_weights))
        elapsed = round(time.time() - start, 2)

        acc = demo_digits(all_weight_vectors)
        print(str((i + 1) * 10) + ' ' + str(elapsed) + ' ' + str(acc))


def graph_digit_acc_dev():
    lines = []
    with open('./PercepDigitSampleSizesAccuracy/final_digit_acc_dev.txt', 'r') as f:
        lines = f.readlines()
        f.close()


    stddevs = []
    accs = []
    for l in range(len(lines)):
        line = lines[l]
        tokens = line.split()
        if 'Mean' in tokens:
            accs.append(float(tokens[-1]))
        elif 'STD' in tokens:
            stddevs.append(float(tokens[-1]))

    print('----')
    print(stddevs)


def face_dev():

    stddevs = []
    # For each percentage
    for v in range(1, 11):
        sample_percentage = v * 10
        acc_list = []

        # for all 10 iterations
        for j in range(0, 10):

            images_sample, labels_sample = perceptron.sample_faces(sample_percentage,'facedata/facedatatrain', 'facedata/facedatatrainlabels' , 451)
            featureslist = perceptron.compute_features2(images_sample)

            converted_labels = []
            for t in range(len(labels_sample)):
                if labels_sample[t] == 1:
                    converted_labels.append(True)
                else:
                    converted_labels.append(False)

            # Weights = initialize_weights(len(functions_list), 0)
            weights = perceptron.initialize_weights(60 * 70, 0)
            '''
                    Run Perceptron / Learn weights
            '''
            start = time.time()
            final_weights = perceptron.compute_weights(weights, featureslist, converted_labels)
            elapsed = time.time() - start


            # Test all weights (weight vectors 0-9), add acc to acc list
            acc = demo_faces(final_weights)
            acc_list.append(acc)
            print(acc)


        stddev = statistics.stdev(acc_list)
        stddevs.append(round(stddev, 2))


    print(stddevs)


def graph_digit_acc():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Digit Run Times by Sample Size')
    colors = ['r', 'b', 'g', 'orange', 'k', 'c', 'm', 'y', 'grey', 'pink']

    accuracies = []
    times = []

    with open('./TrainDigitNoCap/SizeTimeAccuracy.txt', 'r') as file:
        lines = file.readlines()

        for i in range(len(lines)):
            accuracies.append(float(lines[i].split()[2]))
            times.append(float(lines[i].split()[1]) / 60.0)
        file.close()

    print(statistics.stdev(accuracies))
    '''
    ax1.plot([1, 2, 3, 4, 5, 6, 7, 8], times, c=colors[5], marker='.', linestyle='--', )
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%'])
    plt.yticks([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150])

    plt.xlabel('Training Data Sample Size')
    plt.ylabel('Run Time in Minutes')
    # plt.legend(loc='lower right', title='Face Train Runtime');
    plt.show()

    '''




if __name__ == "__main__":
    #graph_face_results()
    #graph_digit_results()

    demo_digit_data_path = 'digitdata/validationimages'
    demo_digit_labels_path = 'digitdata/validationlabels'

    #demo_digits_graph(1000, demo_digit_data_path, demo_digit_labels_path)
    #graphdigits120()
    #graphfacetimes()
    #get_digit_acc()
    #graph_digit_acc()
    #get_digit_acc2()


    face_dev()