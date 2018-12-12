from numpy import *
import operator
import importlib
from os import listdir

#image to vector
def img2vector(filename) :
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32) :
        lineStr = fr.readline()
        for j in range(32) :
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#k-Nearest Neighbours
def classify0 (inX, dataSet, labels, k) :
    #distance Calculation Start   
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    #distance Calculation End
    classCount = {}
    for i in range(k) :
        voteIlabel = labels[sortedDistIndicies[i]]
        #voting with Lowest k distances
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1), reverse = True) #sort Dictionary
    return sortedClassCount[0][0]

#Handwritten digits testing code
def handwritingClassTest() :
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m) :
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest) :
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("The classifier came back with : %d, the real answer is : %d"\
              % (classifierResult, classNumStr))
        if (classifierResult != classNumStr) : errorCount += 1.0
    print("\n The total number of errors is : %d" % errorCount)
    print("\n The total error rate is : %f" % (errorCount/float(mTest)))
