import os
from os import listdir,path,mkdir

#在第0组训练集上构建字典
def createWordMap():
    wordMap = {}
    newWordMap = {}
    data_path = 'D:/repository/TrainData/0/'
    sampleFilesList = listdir(data_path)
    for i in range(len(sampleFilesList)):
        sampleFilesDir = data_path + sampleFilesList[i]
        sampleList = listdir(sampleFilesDir)
        for j in range(len(sampleList)):
            sampleDir = sampleFilesDir + '/' + sampleList[j]
            for line in open(sampleDir).readlines():
                word = line.strip('\n')
                wordMap[word] = wordMap.get(word, 0.0) + 1.0
    # 输出字典大小
    print('The size of wordMap is : %d' % len(wordMap))

    # 返回出现次数大于7的单词
    for key, value in wordMap.items():
        if value > 7:
            newWordMap[key] = value
    dict_path = 'D:/repository/dict.txt'
    #wordCount = 0
    f = open(dict_path, 'w')
    for key, value in newWordMap.items():
        f.write('%s %.1f\n' % (key, value))
        #wordCount += 1
    print('The size of newWordMap is : %d' % len(newWordMap))  # 过滤词频为19时，字典规模为9356；13时字典11879
    return newWordMap

def filteredTrainData():
    src_path = 'D:/repository/TrainData/0/'
    target_path = 'D:/repository/filteredTrainData/0/'
    dict = createWordMap()
    sampleFilesList = listdir(src_path)
    for i in range(len(sampleFilesList)):
        sampleFilesDir = src_path + sampleFilesList[i]
        sampleList = listdir(sampleFilesDir)
        targetFilesDir = target_path + sampleFilesList[i]
        if not path.exists(targetFilesDir):
            mkdir(targetFilesDir)
        for j in range(len(sampleList)):
            targetDir = targetFilesDir + '/' + sampleList[j]
            f = open(targetDir, 'w')
            sampleDir = sampleFilesDir + '/' + sampleList[j]
            for line in open(sampleDir).readlines():
                word = line.strip('\n')
                if word in dict.keys():
                    f.write('%s\n' % word)
            f.close()

def filteredTestData():
    src_path = 'D:/repository/TestData/0/'
    target_path = 'D:/repository/filteredTestData/0/'
    dict = createWordMap()
    sampleFilesList = listdir(src_path)
    for i in range(len(sampleFilesList)):
        sampleFilesDir = src_path + sampleFilesList[i]
        sampleList = listdir(sampleFilesDir)
        targetFilesDir = target_path + sampleFilesList[i]
        if not path.exists(targetFilesDir):
            mkdir(targetFilesDir)
        for j in range(len(sampleList)):
            targetDir = targetFilesDir + '/' + sampleList[j]
            f = open(targetDir, 'w')
            sampleDir = sampleFilesDir + '/' + sampleList[j]
            for line in open(sampleDir).readlines():
                word = line.strip('\n')
                if word in dict.keys():
                    f.write('%s\n' % word)
            f.close()

if __name__=='__main__':
    filteredTrainData()
    filteredTestData()
