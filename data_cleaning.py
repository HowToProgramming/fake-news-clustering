import pandas as pd
from utils import clean
from utils.constants import dataset_dir

fake_news = pd.read_csv(dataset_dir.format('DataSet_Misinfo_FAKE.csv'))[['text']]
real_news = pd.read_csv(dataset_dir.format('DataSet_Misinfo_TRUE.csv'))[['text']]
russian_propaganda = pd.read_csv(dataset_dir.format('EXTRA_RussianPropagandaSubset.csv'))[['text']]

fake_news['label'] = 'fake'
real_news['label'] = 'real'
russian_propaganda['label'] = 'russian'

dataset = pd.concat([fake_news, real_news, russian_propaganda])
dataset['text'] = dataset['text'].astype(str).apply(clean)
dataset.to_csv('cleaned_dataset.csv', index=False)