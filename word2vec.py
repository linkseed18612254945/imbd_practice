#import gensim
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

labeled_data = pd.read_csv('./data/labeledTrainData.tsv', delimiter='\t', quoting=3)
unlabeled_data = pd.read_csv('./data/unlabeledTrainData.tsv', delimiter='\t', quoting=3)

reviews = list(labeled_data['review'].values) + list(unlabeled_data['review'].values)
reviews = [BeautifulSoup(review, 'lxml').get_text().strip() for review in reviews]

data = []
for review in reviews:
    sentences = sent_tokenize(review)
    for sentence in sentences:
        words = word_tokenize(sentence)
        data.append(words)
print(data)