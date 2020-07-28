import parser


def main():
    # trainingdata is a list of node objects
    trainingdata = parser.loaddigittraining()

    spacefeature = []
    plusfeature = []
    hashtagfeature = []
    ratiofeature = []


    for i in range(10):
        spacefeature.append([])
        plusfeature.append([])
        hashtagfeature.append([])
        ratiofeature.append([])


    spnum = 0
    pnum = 0
    hshnum = 0
    countfeaturemap = []
    for q in range(3):
        for w in range(10):
            for e in range(784):
                countfeaturemap[q][w][e] = 0.0


    for j in range(len(trainingdata)):
        for count in range(784):
            for z in range(trainingdata[j]):
                if trainingdata[j][z].space == count:
                    spnum += 1
                if trainingdata[j][z].plus == count:
                    pnum += 1
                if trainingdata[j][z].hshnum == count:
                    hshnum += 1
            countfeaturemap[0][j][count] = spnum/len(trainingdata[j])
            countfeaturemap[1][j][count] = pnum/len(trainingdata[j])
            countfeaturemap[2][j][count] = hshnum/len(trainingdata[j])

    print("Ratio: " + str(len(countfeaturemap[0][2][300])))


if __name__ == "__main__":
    main()
