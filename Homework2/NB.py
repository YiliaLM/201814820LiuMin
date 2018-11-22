import os
import math
from os import path, listdir
from math import log

#训练样本每个类的单词总数以及每个单词出现的次数
def getCateWordsFre(data_path):
    #data_path = 'D:/repository/filteredTrainData/0/'  # 训练样本路径
    cateWordsNum = {}
    cateWordsFre = {}
    sampleFilesList = listdir(data_path)
    for i in range(len(sampleFilesList)):
        count = 0  # 记录每个类的单词总数
        sampleFilesDir = data_path + sampleFilesList[i]
        sampleList = listdir(sampleFilesDir)
        for j in range(len(sampleList)):
            sampleDir = sampleFilesDir + '/' + sampleList[j]
            for line in open(sampleDir).readlines():
                count = count + 1
                word = line.strip('\n')
                keyName = sampleFilesList[i] + '_' + word
                cateWordsFre[keyName] = cateWordsFre.get(keyName, 0) + 1  # 记录每个类中每个单词出现的次数
        cateWordsNum[sampleFilesList[i]] = count
        print('The category %s contains %d words.' % (sampleFilesList[i], cateWordsNum[sampleFilesList[i]]))
    print('Words size in this cate is %d' % len(cateWordsFre))
    return cateWordsNum, cateWordsFre

def NBProcess(train_path, test_path, classifyResultByNB):
    #classifyResultByNB = 'classifyResultByNB.txt'
    f = open(classifyResultByNB, 'w')
    #train_path = 'D:/repository/filteredTrainData/0/'
    #test_path = 'D:/repository/filteredTestData/0/'
    cateWordsNum, cateWordsFre = getCateWordsFre(train_path)  # 返回类中单词总数，每个单词出现次数
    trainTotalNum = sum(cateWordsNum.values())  # 训练样本中总词数
    print('Total words num in train set is %d' % trainTotalNum)
    testFilesList = listdir(test_path)
    for i in range(len(testFilesList)):
        testFilesDir = test_path + testFilesList[i]
        testList = listdir(testFilesDir)
        for j in range(len(testList)):
            testFilesWords = []
            testDir = testFilesDir + '/' + testList[j]
            for line in open(testDir).readlines():
                word = line.strip('\n')
                testFilesWords.append(word)

            maxProb = 0.0
            trainFilesList = listdir(train_path)
            for k in range(len(trainFilesList)):
                prob = computeCateProb(trainFilesList[k], testFilesWords, cateWordsNum, trainTotalNum, cateWordsFre)
                if k == 0:
                    maxProb = prob
                    bestCate = trainFilesList[k]
                    continue
                if prob > maxProb:
                    maxProb = prob
                    bestCate = trainFilesList[k]
            f.write('%s %s\n' % (testList[j], bestCate))
    f.close()

# 以单词为粒度计算概率
def computeCateProb(trainCate, testFilesWords, cateWordsNum, trainTotalNum, cateWordsFre):
    conditionProb = 0
    wordNumInCate = cateWordsNum[trainCate]  # 类k中单词总数
    for i in range(len(testFilesWords)):
        keyName = trainCate + '_' + testFilesWords[i]
        if keyName in cateWordsFre.keys():
            testWordsNumInCate = cateWordsFre[keyName]  # 类k中单词出现次数
        else:
            testWordsNumInCate = 0.0
        eachConditionProb = log((testWordsNumInCate + 0.0001)/(wordNumInCate + trainTotalNum))  # 条件概率
        conditionProb = conditionProb + eachConditionProb
    preProb = log(wordNumInCate/trainTotalNum)  # 先验概率
    prob = conditionProb + preProb
    return prob

def computeAcc(rightCate, resultCate):
    #rightCate = 'D:/repository/classifyRightCate0.txt'
    #resultCate = 'D:/repository/classifyResultByNB.txt'
    rightCateDict = {}
    resultCateDict = {}
    rightCount = 0.0
    for line in open(rightCate).readlines():
        (sampleFile, cate) = line.strip('\n').split(' ')
        rightCateDict[sampleFile] = cate

    for line in open(resultCate).readlines():
        (sampleFile, cate) = line.strip('\n').split(' ')
        resultCateDict[sampleFile] = cate

    for sampleFile in rightCateDict.keys():
        if(rightCateDict[sampleFile] == resultCateDict[sampleFile]):
            rightCount += 1.0

    print('The rightCount is : %d The rightCate is : %d' % (rightCount, len(rightCateDict)))
    acc = rightCount/(len(rightCateDict))
    print('The accuracy of NB classifier is : %f' % acc)

if __name__ == '__main__':
    train_path = 'D:/repository/filteredTrainData/0/'
    test_path = 'D:/repository/filteredTestData/0/'
    classifyResultByNB = 'D:/repository/classifyResultByNB.txt'
    NBProcess(train_path, test_path, classifyResultByNB)
    rightCate = 'D:/repository/classifyRightCate0.txt'
    resultCate = 'D:/repository/classifyResultByNB.txt'
    computeAcc(rightCate, resultCate)