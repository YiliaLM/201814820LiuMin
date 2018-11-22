Data Mining Homeworks<br>
<br>
Homework1:VSM and KNN<br>
1.预处理文本数据集，并且得到每个文本的VSM表示<br>
2.实现KNN分类器，测试其在20news-18828上的分类效果<br>
Dataset:<br>
The 20 Newsgroups dataset is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. <br>
[20news-18828.tar.gz](http://qwone.com/~jason/20Newsgroups/) : duplicates removed, only "From" and "Subject" headers (18828 documents)<br>
Experiments:<br>
Preprocess.py firstly preprocesses the whole dataset which including word tokenization,stemming and removing the stopwords.Then the dataset was divided into five groups of different trainsets and testsets by 8:2.<br>
createDict.py constructed a dictionary on one of the trainsets, where the dictionary was shown in dict.txt.Then filtered out the words in the trainset and testset that didn't appear in the dictionary.<br>
VSM.py computed the tfidf weight on both trainset and testset and obtained their VSM representation.<br>
Finally KNN.py implemented the KNN algorithm.<br>

Homework2:NBC<br>
实现朴素贝叶斯分类器，测试其在20 Newsgroups数据集上的效果<br>
Dataset:同Homework1<br>
Experiments:<br>
NB.py fristly counted the total number of words in each category and the number of times each word appears in the train samples.Then calculated the probability to determine which category the test sample belongs to.Finally computed the accuracy of NBC.<br>
