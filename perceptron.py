from __future__ import print_function
import samples
import random
import math


'''
Flag - 0 indicates weights starting at 0, otherwise initialize weights randomly from -1/sqrt(n) to 1/sqrt(n)
Returns - list of initialized weights
'''
def initialize_weights(n, flag):
    if flag == 0:
        MINWEIGHT = 0
        MAXWEIGHT = 0
        weights = [random.uniform(MINWEIGHT, MAXWEIGHT) for a in range(n)]
    else:
        MINWEIGHT = -1/math.sqrt(n)
        MAXWEIGHT = 1/math.sqrt(n)
        weights = [random.uniform(MINWEIGHT, MAXWEIGHT) for a in range(n)]
    return weights

'''
Function 1, counts total number of non-empty pixels
'''
def f1(datum, width, height):
    count = 0
    for i in range(width):
        for j in range(height):
            if datum.pixels[i][j] == 2:
                count += 1
    return count


'''
Returns a list of features vectors (2d list)
functions - a list of functions
data - list of datum objects

***The order of features vectors follows the same order in the list of datum objects
'''
def compute_features(functionslist, datalist):
    featureslist = []

    for data in datalist:
        featuresvector = []
        for f in functionslist:
            featuresvector.append(f(data))
        featureslist.append(featuresvector)

    return featureslist



'''
For determining the f-values
Returns a list of all f-values, each f-value's index corresponds with the each datum's index in the datalist
'''
def compute_f_values(weights, featureslist):
    for featuresvector in featureslist:
        fsum = 0
        for wfunction in weights:
            if type(wfunction) is float:
                fsum += wfunction*



'''
THE main perceptron algorithm, iterates through all f-values in flist and updates them as needed.
When there are no more f-values to update, a list of all the final weights are returned
'''
def update_f_values(flist, labels):





if __name__ == "__main__":
    # 451
    nfaces = 451

    phi_functions = []
    faces = samples.loadDataFile('facedata/facedatatrain', nfaces, 60, 74)
    print(str(len(faces)) + '----------------------------------------------nfaces')
    exit()
    labels = samples.loadLabelsFile('./data/facedata/facedatatrainlabels', nfaces)
    weights = initialize_weights(nfaces, 0)

    for i in range (nfaces):
        print('----------------------------------  '  + str(labels[i]) + '  -----------------------------------')
        for j in range(74):
            for k in range(60):
                if faces[i].pixels[k][j] != 2:
                    print(' ', end="")
                else:
                    print('#',end="")
            print()



