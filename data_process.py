import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

labeled_data = './data/labeledTrainData.tsv'
clean_review_data = './data/clean_reviews.txt'
test_data = './data/testData.tsv'

def clean(reviews):
    words_list = []
    for review in reviews:
        clean_text = BeautifulSoup(review, 'lxml').get_text()
        clean_text = re.sub('[^a-zA-Z]', ' ', clean_text).lower()
        words = clean_text.split(' ')
        stops = set(stopwords.words('english'))
        words = [word for word in words if word not in stops and word != '']
        words_list.append(' '.join(words))
    return words_list

df = pd.read_csv(labeled_data, delimiter='\t').dropna()
test = pd.read_csv(test_data, delimiter='\t').dropna()

# to one hot
with open(clean_review_data, 'r', encoding='utf-8') as f:
    reviews = f.read().splitlines()
cv = CountVectorizer(max_features=10000)
one_hot_reviews = cv.fit_transform(reviews).toarray()

# randomforest model
forest = RandomForestClassifier(n_estimators=400)
forest.fit(one_hot_reviews, df['sentiment'])

# test
clean_test = clean(test['review'].values)
one_hot_test = cv.transform(clean_test)
res = forest.predict(one_hot_test)

#output
output = pd.DataFrame(data={'id': test['id'], 'sentiment': res})
output.to_csv('./output/bagofwords_randomforest_400tree.csv', index=False)

