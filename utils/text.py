from string import punctuation
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

ps = PorterStemmer()
english_stopwords = stopwords.words('english')

def lowercase(text: str):
    return text.lower()

def remove_punctuation(text):
    no_punct=[words for words in text if words not in punctuation]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct

def stem(text):
    return " ".join([ps.stem(w) for w in text.split(" ")])

def remove_stopwords(text):
    text=[word for word in text.split(" ") if word not in english_stopwords]
    return " ".join(text)

def clean(text: str):
    lower_text = lowercase(text)
    rm_punc = remove_punctuation(lower_text)
    rm_stopwords = remove_stopwords(rm_punc)
    stem_ = stem(rm_stopwords)
    x = stem_
    return x