import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from wordcloud import WordCloud

# EDA

def generate_wordcloud(text, title=""):
    wc = WordCloud().generate(text)
    plt.figure()
    plt.title(title)
    plt.imshow(wc, interpolation = 'bilinear')
    plt.axis('off')

news = pd.read_csv('cleaned_dataset.csv')

fake_news = "\n".join(news[news['label'] == "fake"]['text'].astype(str).values)
real_news = "\n".join(news[news['label'] == "real"]['text'].astype(str).values)
russian = "\n".join(news[news['label'] == "russian"]['text'].astype(str).values)

generate_wordcloud(fake_news, "fake news")
generate_wordcloud(real_news, "real news")
generate_wordcloud(russian, "russian propaganda")

plt.show()