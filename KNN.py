import operator
import time
import numpy as np
from operator import itemgetter

def doProcess():
    trainDataPath = 'D:/repository/TrainTFIDF.txt'
    testDataPath = 'D:/repository/TestTFIDF.txt'
    KNNResultFile = 'D:/repository/KNNClassifyResult.txt'

    trainDocWordMap = {}  # 字典<key, value> key=category:doc, value={{word1,tfidf1},...}
    for line in open(trainDataPath).readlines():
        lineSplitBlock = line.strip('\n').split(' ')
        trainWordMap = {}
        m = len(lineSplitBlock) - 1
        for i in range(2, m, 2):  # 步长为2，从2到m，即在每个文档向量中提取(word, tfidf)存入字典
            trainWordMap[lineSplitBlock[i]] = lineSplitBlock[i + 1]

        cateDoc_key = lineSplitBlock[0] + '_' + lineSplitBlock[1]  # 在每个文档向量中提取category和doc
        trainDocWordMap[cateDoc_key] = trainWordMap  # <类_文件名，<word,tfidf>>

    testDocWordMap = {}
    for line in open(testDataPath).readlines():
        lineSplitBlock = line.strip('\n').split(' ')
        testWordMap = {}
        m = len(lineSplitBlock) - 1
        for i in range(2, m, 2):
            testWordMap[lineSplitBlock[i]] = lineSplitBlock[i + 1]

        cateDoc_key = lineSplitBlock[0] + '_' + lineSplitBlock[1]
        testDocWordMap[cateDoc_key] = testWordMap

    # 遍历测试样本，计算与所有训练样本的距离，进行分类
    count = 0  # 测试样本数目
    rightCount = 0  # 正确分类数目
    resultWriter = open(KNNResultFile, 'w')
    k = 25
    for item in testDocWordMap.items():
        classifyResult = KNN(item[0], item[1], trainDocWordMap)  # 调用KNN()函数
        count += 1
        print('This is %d round!' % count)

        classifyRight = item[0].split('_')[0]
        resultWriter.write('%s %s\n' %(classifyRight, classifyResult))  # 正确类别，KNN分类类别
        if classifyRight == classifyResult:
            rightCount += 1
        print('RightCount:%d' % rightCount)

    accuracy = float(rightCount)/float(count)
    print('KNN classifier\'s accuracy based %d is: %.6f' % (k, accuracy))
    return accuracy

def KNN(testKey, testValue, trainDocWordMap):
    similarityMap = {}  # <类_文件名，距离>,之后将该hashmap按照value排序
    for item in trainDocWordMap.items():
        similarity = computeSim(testValue, item[1])  # 调用computeSim()函数
        similarityMap[item[0]] = similarity
    sortedSimilarityMap = sorted(similarityMap.items(), key=itemgetter(1), reverse=True)  # 按value排序

    k = 25
    cateSimilarityMap = {}  # <类，距离和>
    for i in range(k):
        category = sortedSimilarityMap[i][0].split('_')[0]
        cateSimilarityMap[category] = cateSimilarityMap.get(category, 0) + sortedSimilarityMap[i][1]
    sortedCateSimMap = sorted(cateSimilarityMap.items(), key=itemgetter(1), reverse=True)

    return sortedCateSimMap[0][0]


def computeSim(testValue, trainValue):
    testList = []  # 测试向量与训练向量共有的词在测试向量中的tfidf值
    trainList = []  # 测试向量与训练向量共有的词在训练向量中的tfidf值
    for word, weight in testValue.items():
        if trainValue.__contains__(word):
            testList.append(float(weight))  # float将字符型数据转换为数值型数据
            trainList.append(float(trainValue[word]))
    testVec = np.mat(testList)  # 列表转成矩阵，将之后的向量相乘运算和使用numpy的范式函数计算
    trainVec = np.mat(trainList)
    num = float(testVec * trainVec.T)  # 分子
    denom = float(np.linalg.norm(testVec) * np.linalg.norm(trainVec))
    cosSim = num/(1.0+float(denom))
    return cosSim

if __name__ == '__main__':
    start = time.clock()
    doProcess()
    end = time.clock()
    print('Runtime is :' + str(end - start))