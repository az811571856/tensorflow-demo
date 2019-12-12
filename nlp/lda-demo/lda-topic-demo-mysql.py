# encoding:utf-8
# reference:https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
# https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import MySQLdb
from sqlalchemy import create_engine
import pandas as pd

n_samples = 10
n_features = 1000
n_components = 5
n_top_words = 20

def read_mysql():
    mysql_cn = MySQLdb.connect(host='', port=33061, user='', passwd='', db='gene')
    df = pd.read_sql('SELECT t_article.id, t_article_corpus.corpus from t_article_corpus, t_article  where t_article.id = t_article_corpus.article_id limit '+str(n_samples)+';', con=mysql_cn)
    mysql_cn.close()
    return df

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

print("Loading dataset...")
t0 = time()
df_raw = read_mysql()
data_samples = df_raw['corpus']
print("读取语料。done in %0.3fs." % (time() - t0))

# Use tf-idf features for NMF.
print("抽取 tf-idf 特征，用于非负的矩阵分解NMF")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("完成tf-idf特征构建，done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("抽取 tf 特征，用于LDA")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("抽取 tf 特征完成 in %0.3fs." % (time() - t0))
print()

# Fit the NMF model
print("用tf-idf特征训练NMF模型(范数)，, "
      "文章个数=%d and 特征个数=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
print("训练完成。done in %0.3fs." % (time() - t0))

print("\n在非负的矩阵分解模型(范数)的主题:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

# Fit the NMF model
print("用ft-idf特征训练非负的矩阵分解模型(普通的KL散度), 文章个数=%d and 特征个数=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_components, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)
print("完成训练 in %0.3fs." % (time() - t0))

print("\n在NMF模型的主题 (普通的KL散度):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

print("用tf特征训练LDA模型, "
      ", 文章个数=%d and 特征个数=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\n在LDA模型的主题:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
