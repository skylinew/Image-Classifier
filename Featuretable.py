class FeatureTable:

    # numtables is 10 for digits, 2 for face
    # digits: xsize and ysize should be 28
    # face: xsize 70 and y size 60
    def __init__(self, numtables, xsize, ysize):
        self.table = []
        for i in range(numtables):
            self.table.append([])
            for z in range(xsize):
                self.table[i].append([])
                for t in range(ysize):
                    self.table[i][z].append([])
                    for last in range(3):
                        self.table[i][z][t].append(0.0)

    def filltable(self, trainingdata):
        for i in range(10):
            for z in range(len(trainingdata[i])):
                for j in range(len(self.table[i])):
                    for k in range(len(self.table[i][j])):
                        if trainingdata[i][z].image[j][k] == 0:
                            self.table[i][j][k][0] += (1/float(len(trainingdata[i])))
                        elif trainingdata[i][z].image[j][k] == 1:
                            self.table[i][j][k][1] += (1/float(len(trainingdata[i])))
                        else:
                            self.table[i][j][k][2] += (1/float(len(trainingdata[i])))

    def printtable(self):
        for i in range(10):
            for j in range(len(self.table[i])):
                for k in range(len(self.table[i][j])):
                    print("space: " + str(self.table[i][j][k][0]) + " plus: " + str(self.table[i][j][k][1]) + " hashtag: " + str(self.table[i][j][k][2]))