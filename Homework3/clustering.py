import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score
from collections import Counter

def kmeans(weight, labels, k):
    km = KMeans(n_clusters=k).fit(weight)
    labels_predict = km.predict(weight)
    nmi = normalized_mutual_info_score(labels, labels_predict)
    print('The nmi of K-Means is :', nmi)

def affinitypropagation(weight, labels):
    ap = AffinityPropagation().fit(weight)
    labels_predict = ap.predict(weight)
    nmi = normalized_mutual_info_score(labels, labels_predict)
    print('The nmi of Affinity propagation is :', nmi)

def meanshift(weight, labels):
    ms = MeanShift().fit(weight)
    labels_predict = ms.predict(weight)
    nmi = normalized_mutual_info_score(labels, labels_predict)
    print('The nmi of Mean-Shift is :', nmi)

def spectralclustering(weight, labels, k):
    sc = SpectralClustering(n_clusters=k).fit(weight)
    labels_predict = sc.fit_predict(weight)
    nmi = normalized_mutual_info_score(labels, labels_predict)
    print('The nmi of Spectral clustering is :', nmi)

def wardhierarchicalclustering(weight, labels, k):
    whc = AgglomerativeClustering(n_clusters=k).fit(weight)
    labels_predict = whc.fit_predict(weight)
    nmi = normalized_mutual_info_score(labels, labels_predict)
    print('The nmi of Ward hierarchical clustering is :', nmi)

def agglomerativeclustering(weight, labels, k):
    ac = AgglomerativeClustering(linkage='complete', n_clusters=k).fit(weight)
    labels_predict = ac.fit_predict(weight)
    nmi = normalized_mutual_info_score(labels, labels_predict)
    print('The nmi of Agglomerative clustering is :', nmi)

def dbscan(weight, labels):
    db = DBSCAN().fit(weight)
    labels_predict = db.fit_predict(weight)
    nmi = normalized_mutual_info_score(labels, labels_predict)
    print('The nmi of DBSCAN is :', nmi)

def gaussianmixtures(weight, labels, k):
    gm = GaussianMixture(n_components=k, covariance_type='diag').fit(weight)
    labels_predict = gm.predict(weight)
    nmi = normalized_mutual_info_score(labels, labels_predict)
    print('The nmi of Gaussian mixtures is :', nmi)

if __name__ == '__main__':
    data_path = 'D:/repository/Homework3/Tweets.txt'
    texts = []
    labels = []
    for line in open(data_path, 'r').readlines():
        tweets = json.loads(line)
        # print(tweets)
        texts.append(tweets['text'])
        labels.append(tweets['cluster'])

    k = len(Counter(labels))  # k为总类别数

    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(texts))
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

    '''8种聚类算法使用NMI评价聚类效果'''
    kmeans(weight, labels, k)
    affinitypropagation(weight, labels)
    meanshift(weight, labels)
    spectralclustering(weight, labels, k)
    wardhierarchicalclustering(weight, labels, k)
    agglomerativeclustering(weight, labels, k)
    dbscan(weight, labels)
    gaussianmixtures(weight, labels, k)
