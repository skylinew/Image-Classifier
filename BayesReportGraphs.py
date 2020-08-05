import matplotlib.pyplot as plt
import statistics

def plots(percent):
    avgs = []
    avgst = []
    stdt = []
    times = []
    accavg = []
    avgtime = []
    z = -1
    for j in range(10):
        avgs.append([])
        avgst.append([])
        times.append([])
        stdt.append([])

    for zep in range(10):
        print("\n\n")
        f = open('./FaceResults/' + str(int((percent + (zep * percent)) * 100)) + 'percent/result.txt', 'r')
        line = f.readlines()
        count = 0
        for eachline in line:
            info = eachline.split(',')
            c = [float(b) for b in info]
            if count == 10:
                avgst.append(c)
                continue
            elif count == 11:
                stdt.append(c)
                break
            avg, dur = c
            avgs[zep].append(avg)
            times[zep].append(dur)
            count += 1
    for zepilo in range(10):
        avgtime.append(statistics.mean(times[zepilo]))
        accavg.append(statistics.mean(avgs[zepilo]))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Digit: Average training runtime of Naive Bayes based on sample size')
    for k in range(10):
        ax1.scatter([k+1]*10, times[k], marker='.', color='b')
    ax1.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], avgtime)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    plt.xlabel('Training Data Sample Size')
    plt.ylabel('Runtime')
    plt.show()
    f.close()


if __name__ == "__main__":
        plots(.1)
