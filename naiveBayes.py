import fileread
import Featuretable

 # cats is number of categories
 # 10 for digits
 # 2 for faces

def main(cats):
    # trainingdata is a list of node objects
    if cats == 10:
        trainingdata = fileread.loadtraining(5000, 'digitdata/trainingimages', 'digitdata/traininglabels', 28, 28, cats)
        features = Featuretable.FeatureTable(cats, 28, 28)
        features.filltable(trainingdata, cats)
        validationdata = fileread.loadtest(1000, 'digitdata/validationimages', 'digitdata/validationlabels', 28, 28)
        validationprob = []
        results = []
        for z in range(cats):
            validationprob.append(0.0)
            results.append(0.0)
    else:
        trainingdata = fileread.loadtraining(451, 'facedata/facedatatrain', 'facedata/facedatatrainlabels', 60, 70, cats)
        features = Featuretable.FeatureTable(cats, 70, 60)
        features.filltable(trainingdata, cats)
        validationdata = fileread.loadtest(1000, 'facedata/facedatavalidation', 'facedata/facedatavalidationlabels', 60, 70)
        validationprob = []
        results = []
        for z in range(cats):
            validationprob.append(0.0)
            results.append(0.0)

    correctcount = 0
    totalcount = 0
    for i in range(len(validationdata)):
        for q in range(cats):
            probability = fileread.digitprob(trainingdata, q)
            for j in range(len(validationdata[i].image)):
                for k in range(len(validationdata[i].image[j])):
                    if validationdata[i].image[j][k] == 0:
                        probability *= features.table[q][j][k][0]
                    elif validationdata[i].image[j][k] == 1:
                        probability *= features.table[q][j][k][1]
                    else:
                        probability *= features.table[q][j][k][2]
            validationprob[q] = probability

        maxp = 0.0
        labelv = -1
        for maxv in range(len(validationprob)):
            if validationprob[maxv] > maxp:
                maxp = validationprob[maxv]
                labelv = maxv
        if validationdata[i].label == labelv:
            correctcount += 1
        totalcount += 1
        print("Actual label: " + str(validationdata[i].label) + " Predicted label: " + str(
            labelv) + " Probability: " + str(maxp))
    print("Accuracy: " + str(correctcount / float(totalcount)))


'''
    spnum = 0
    pnum = 0
    hshnum = 0
    countfeaturemap = []
    for q in range(3):
        countfeaturemap.append([])
        for w in range(10):
            countfeaturemap[q].append([])
            for e in range(784):
                countfeaturemap[q][w].append(0.0)

    for j in range(10):
        for count in range(784):
            spnum = 0
            pnum = 0
            hshnum = 0
            for z in range(len(trainingdata[j])):
                if trainingdata[j][z].space == count:
                    spnum += 1
                if trainingdata[j][z].plus == count:
                    pnum += 1
                if trainingdata[j][z].hashtag == count:
                    hshnum += 1
            countfeaturemap[0][j][count] = spnum/float(len(trainingdata[j]))
            countfeaturemap[1][j][count] = pnum/float(len(trainingdata[j]))
            countfeaturemap[2][j][count] = hshnum/float(len(trainingdata[j]))

    validationdata = fileread.loaddigitdata(1000, 'digitdata/validationimages', 'digitdata/validationlabels')

    validationprob = []
    for it in range(10):
        validationprob.append(0.0)

    correctcount = 0
    totalcount = 0
    out = 1.0
    for val in range(len(validationdata)):
        for i in range(10):
            out = fileread.digitprob(trainingdata, i)
            for feat in range(3):
                if feat == 0:
                    lpar = validationdata[val].space
                elif feat == 1:
                    lpar = validationdata[val].plus
                else:
                    lpar = validationdata[val].hashtag
                out *= countfeaturemap[feat][i][lpar]
            validationprob[i] = out

        maxp = 0.0
        labelv = -1
        for maxv in range(len(validationprob)):
            if validationprob[maxv] > maxp:
                maxp = validationprob[maxv]
                labelv = maxv
        if validationdata[val].label == labelv:
            correctcount += 1
        totalcount += 1
        print("Actual label: " + str(validationdata[val].label) + " Predicted label: " + str(labelv) + " Probability: " + str(maxp))
    print("Accuracy: " + str(correctcount/float(totalcount)))


    #print("Label: " + str(labelv) + " with probability: " + str(maxp))

    #sumcheck = 0
    #for m in range(len(countfeaturemap[1][1])):
    #    sumcheck += countfeaturemap[0][9][m]
    #    print("Probability of " + str(m) + " in number " + str(n) + " is: " + str(countfeaturemap[1][n][m]))
    #print("Should be 1: " + str(sumcheck))
'''

if __name__ == "__main__":
    main(2)
