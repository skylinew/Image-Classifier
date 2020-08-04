import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import samples
import time
import perceptron
from scipy.stats import norm
import statistics



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





if __name__ == "__main__":
    #graph_face_results()
    #graph_digit_results()

    demo_digit_data_path = 'digitdata/validationimages'
    demo_digit_labels_path = 'digitdata/validationlabels'

    #demo_digits_graph(1000, demo_digit_data_path, demo_digit_labels_path)
    graphdigits120()
    #graphfacetimes()
