import fileread


def main():
    # trainingdata is a list of node objects
    trainingdata = fileread.loaddigittraining()

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

    #sumcheck = 0
    #for m in range(len(countfeaturemap[1][1])):
    #    sumcheck += countfeaturemap[0][9][m]
    #    print("Probability of " + str(m) + " in number " + str(n) + " is: " + str(countfeaturemap[1][n][m]))
    #print("Should be 1: " + str(sumcheck))


if __name__ == "__main__":
    main()
