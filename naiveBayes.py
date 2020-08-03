import fileread
import Featuretable
import math

 # cats is number of categories
 # 10 for digits
 # 2 for faces

def main(cats, percent):
    # trainingdata is a list of node objects
    if cats == 10:
        trainingdata = fileread.loadtraining(5000, 'digitdata/trainingimages', 'digitdata/traininglabels', 28, 28, cats, percent)
        features = Featuretable.FeatureTable(cats, 28, 28)
        features.filltable(trainingdata, cats)
        #features.printtable(cats)
        validationdata = fileread.loadtest(1000, 'digitdata/testimages', 'digitdata/testlabels', 28, 28)
        validationprob = []
        results = []
        for z in range(cats):
            validationprob.append(0.0)
            results.append(0.0)
    else:
        trainingdata = fileread.loadtraining(451, 'facedata/facedatatrain', 'facedata/facedatatrainlabels', 60, 70, cats, percent)
        features = Featuretable.FeatureTable(cats, 60, 70)
        features.filltable(trainingdata, cats)
        validationdata = fileread.loadtest(150, 'facedata/facedatatest', 'facedata/facedatatestlabels', 60, 70)
        validationprob = []
        results = []
        for z in range(cats):
            validationprob.append(0.0)
            results.append(0.0)

    correctcount = 0
    totalcount = 0
    for i in range(len(validationdata)):
        for q in range(cats):
            if cats == 10:
                probability = math.log(fileread.digitprob(trainingdata, q, percent))
            else:
                probability = math.log(fileread.faceprob(trainingdata, q, percent))
            for j in range(len(validationdata[i].image)):
                for k in range(len(validationdata[i].image[j])):
                    if validationdata[i].image[j][k] == 0:
                        if features.table[q][j][k][0] == 0:
                            continue
                        else:
                            probability += math.log(features.table[q][j][k][0])
                    elif validationdata[i].image[j][k] == 1:
                        if features.table[q][j][k][1] == 0:
                            continue
                        else:
                            probability += math.log(features.table[q][j][k][1])
                    else:
                        if features.table[q][j][k][2] == 0:
                            continue
                        else:
                            probability += math.log(features.table[q][j][k][2])
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
        #print("Actual label: " + str(validationdata[i].label) + " Predicted label: " + str(
        #    labelv) + " Probability: " + str(maxp))
    storagepath = './FaceResults/' + str(int(percent * 100)) + 'percent/result.txt'
    f = open(storagepath, 'a')
    f.write(str(correctcount / float(totalcount)) + ", ")
    f.close()
    #print("Accuracy: " + str(correctcount / float(totalcount)))


if __name__ == "__main__":
    for i in range(1, 11):
        if i == 11:
            for k in range(5):
                main(2, 1)
        else:
            for j in range(5):
                main(2, (.1 * i))
