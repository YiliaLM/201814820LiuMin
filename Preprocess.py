import os
import nltk
import string
import codecs
from os import listdir, mkdir, path
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer

srcPath = 'D:/repository/20news-18828/'
processedPath = 'D:/repository/Processed20news-18828/'

def createFiles():
    srcFilesList = listdir(srcPath)
    for i in range(len(srcFilesList)):
        # 20个文件夹的路径
        dataFilesDir = srcPath + srcFilesList[i]
        dataFilesList = listdir(dataFilesDir)
        # print(dataFilesList)
        # 预处理后20个文件夹的路径
        targetDir = processedPath + srcFilesList[i]
        if not path.exists(targetDir):
            mkdir(targetDir)
        else:
            print('%s exits!' % targetDir)
        for j in range(len(dataFilesList)):
            # 调用createProcessFile()
            createProcessFile(srcFilesList[i], dataFilesList[j])
            print('%s,%s' % (srcFilesList[i], dataFilesList[j]))
    print('Preprocessing finished!')

def createProcessFile(srcFilesName, dataFilesName):
    srcFile = srcPath + srcFilesName + '/' + dataFilesName
    targetFile = processedPath + srcFilesName + '/' + dataFilesName
    f = open(targetFile, 'w')
    dataList = open(srcFile, encoding='utf8', errors='ignore').readlines()# 之前报错，所以加上encoding和errors
    for line in dataList:
        # 调用lineProcess()处理每行文本
        processedLine = lineProcess(line)
        for word in processedLine:
            # 每行一个单词
            f.write("%s\n" % word)
    f.close()

def lineProcess(line):
    stopWords = stopwords.words('english')
    stemmer = PorterStemmer()
    # lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')

    delSymbols = string.digits
    remove = str.maketrans('', '', delSymbols)
    # 去除数字
    cleanLine = line.translate(remove)

    loweredLine = cleanLine.lower()
    wordTokens = tokenizer.tokenize(loweredLine)
    stemmed = []
    for word in wordTokens:
        stemmed.append(stemmer.stem(word))
    filteredWords_1 = [w for w in stemmed if not w in stopWords]
    filteredWords = [w for w in filteredWords_1 if not len(str(w)) < 3]
    return filteredWords

# @param indexOfSample第k次实验
# @param trainSampleRatio训练集占比
def splitData(indexOfSample,trainSampleRatio):
    data_path = 'D:/repository/Processed20news-18828/'
    traindata_path = 'D:/repository/TrainData/'
    testdata_path = 'D:/repository/TestData/'
    #f = open(classifyRightCate, 'w')
    sampleFilesList = listdir(data_path)
    for i in range(len(sampleFilesList)):
        sampleFilesDir = data_path + sampleFilesList[i]
        sampleList = listdir(sampleFilesDir)
        m = len(sampleList)
        testBeginIndex = indexOfSample * (m * (1 - trainSampleRatio))
        testEndIndex = (indexOfSample + 1) * (m * (1 - trainSampleRatio))
        for j in range(m):
            if (j > testBeginIndex) and (j < testEndIndex):
                print('Start generating test data!')
                #f.write('%s %s\n' %(sampleList[j],sampleFilesList[i]))# 写入每篇文档序号，其所在的文档名称即类别
                targetDir = testdata_path + str(indexOfSample) + '/' + sampleFilesList[i]
            else:
                print('Start generating train data!')
                targetDir = traindata_path + str(indexOfSample) + '/' +sampleFilesList[i]
            if not path.exists(targetDir):
                makedirs(targetDir)
            sampleDir = sampleFilesDir + '/' + sampleList[j]
            sample = open(sampleDir).readlines()
            sampleWriter = open(targetDir + '/' + sampleList[j], 'w')
            for line in sample:
                sampleWriter.write('%s\n' %(line.strip('\n')))
            sampleWriter.close()
    #f.close()

if __name__ == '__main__':
    createFiles()
    #生成5组训练集与测试集
    for i in range(5):
        #classifyRightCate = 'classifyRightCate' + str(i) + '.txt'
        splitData(i,0.8)
