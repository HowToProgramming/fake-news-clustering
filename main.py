import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

tfidf = TfidfVectorizer()
svd = TruncatedSVD(n_components=1000)

news = pd.read_csv('cleaned_dataset.csv').astype(str)

tf_idf_fit_transform = tfidf.fit_transform(news['text'])
print(tf_idf_fit_transform.shape)
# i swear i will not do this again this is a f**king pain to my computer
svd_res = svd.fit_transform(tf_idf_fit_transform)
var_explained = svd.explained_variance_ratio_.sum()
print(var_explained) 
# 1000 components explain about 43.5% of the dataset
# but if i increase to 10k components my pc get fucked boom

def plot(svd_res, label):
    fake_svd = svd_res[label == "fake"]
    real_svd = svd_res[label == "real"]
    russian_svd = svd_res[label == "russian"]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(fake_svd[:, 0], fake_svd[:, 1], fake_svd[:, 2], label='fake')
    ax.scatter(real_svd[:, 0], real_svd[:, 1], real_svd[:, 2], label='real')
    ax.scatter(russian_svd[:, 0], russian_svd[:, 1], russian_svd[:, 2], label='russian')
    ax.legend()

plot(svd_res, news['label'].values)
plt.savefig('hi.png')