import createDict
import os
import math
import time
from os import listdir
from math import log

def computeIDF():
    sampleFilesDir = 'D:/repository/filteredTrainData/0/'  # 训练集
    #sampleFilesDir = 'D:/repository/filteredTestData/0/' # 测试集
    wordDocMap = {}  # <word, set(docM,...,docN)>
    wordIDFMap = {}  # <word, IDF值>
    docNum = 0  # 文档总数
    wordDocNum = 0  # 出现某单词的文档数目
    sampleFilesList = listdir(sampleFilesDir)
    for i in range(len(sampleFilesList)):
        sampleDir = sampleFilesDir + sampleFilesList[i]
        sampleList = listdir(sampleDir)
        for j in range(len(sampleList)):
            sample = sampleDir + '/' + sampleList[j]
            docNum += 1
            for line in open(sample).readlines():
                word = line.strip('\n')
                if word in wordDocMap.keys():
                    wordDocMap[word].add(sampleList[j])  # set保存word出现过的文档
                else:
                    wordDocMap.setdefault(word, set())
                    wordDocMap[word].add(sampleList[j])
        print('This is %d round!' % (i + 1))

    for word in wordDocMap.keys():
        wordDocNum = len(wordDocMap[word])  # 计算set中文档个数
        IDF = log(docNum/wordDocNum)/log(10)
        wordIDFMap[word] = IDF
    return wordIDFMap

# 将IDF值写入文件保存
def writeIDF():
    start = time.clock()
    wordIDFMap = computeIDF()
    end = time.clock()
    print('Runtime is :' + str(end - start))
    f = open('D:/repository/TrainWordIDF.txt', 'w')
    #f = open('D:/repository/TestWordIDF.txt', 'w')
    for word, IDF in wordIDFMap.items():
        f.write('%s %.6f\n' % (word, IDF))
    f.close()

def computeTFIDF():
    wordIDF = {}  # <word, IDF值>从文件中读取的数据保存在此字典结构中
    data_path = 'D:/repository/TrainWordIDF.txt'
    #data_path = 'D:/repository/TestWordIDF.txt'
    for line in open(data_path).readlines():
        (word, IDF) = line.strip('\n').split(' ')
        wordIDF[word] = IDF
    #wordIDF = computeIDF()

    sampleFilesDir = 'D:/repository/filteredTrainData/0/'
    #sampleFilesDir = 'D:/repository/filteredTestData/0/'
    f = open('D:/repository/TrainTFIDF.txt', 'w')
    #f = open('D:/repository/TestTFIDF.txt', 'w')
    sampleFilesList = listdir(sampleFilesDir)
    for i in range(len(sampleFilesList)):
        sampleDir = sampleFilesDir + sampleFilesList[i]
        sampleList = listdir(sampleDir)
        for j in range(len(sampleList)):
            wordTF = {}  # <word, doc下该word出现次数>
            sumPerDoc = 0  # doc中单词总数
            sample = sampleDir + '/' + sampleList[j]
            for line in open(sample).readlines():
                sumPerDoc += 1  # 每行一个word
                word = line.strip('\n')
                wordTF[word] = wordTF.get(word, 0) + 1

            f.write('%s %s ' % (sampleFilesList[i], sampleList[j]))  # 写入类别，文档名
            for word, count in wordTF.items():
                TF = float(count)/float(sumPerDoc)
                TFIDFfweight = TF * float(wordIDF[word])
                f.write('%s %f ' % (word, TFIDFfweight))  # 继续写入单词及其tfidf权重
            f.write('\n')
        print('This is %d round!' % (i + 1))
    f.close()

if __name__=='__main__':
    writeIDF()
    computeTFIDF()