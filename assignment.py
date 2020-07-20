import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer


data = pd.read_csv(r'C:\Users\abhay.jamwal\Downloads\4133_8841_bundle_archive\twcs\twcs.csv')

x = data[pd.isnull(data.in_response_to_tweet_id) & data.inbound]
data2 = pd.merge(x, data, left_on='tweet_id', 
                                  right_on='in_response_to_tweet_id')

authors = data2.groupby('author_id_y').agg('count').sort_values(['tweet_id_x'])

dropped_cols = ['tweet_id_x', 'inbound_x','response_tweet_id_x', 'in_response_to_tweet_id_x', 
          'tweet_id_y', 'inbound_y','response_tweet_id_y', 'in_response_to_tweet_id_y','created_at_x','created_at_y']

data2.drop(dropped_cols, axis=1, inplace=True)


rev_count = data2.groupby("author_id_y")["text_x"].count()
rev_count[rev_count>10000].plot(kind = 'bar')

rev_count.max()

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

data2['scores'] = data2['text_x'].apply(lambda text_x: sid.polarity_scores(text_x))
data2['compound']  = data2['scores'].apply(lambda score_dict: score_dict['compound'])

data2['comp_score'] = data2['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')

def rem_urls(text):
    temp = re.compile(r'https?://\S+|www\.\S+')
    return temp.sub(r'', text)

data2.text_x = data2.text_x.apply(rem_urls)

data2.text_x.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
data2.text_x.replace('\s+[a-zA-Z]\s+'," ",regex=True, inplace=True)
data2.text_x.replace('\^[a-zA-Z]\s+'," ",regex=True, inplace=True)
data2.text_x.replace('^b\s+'," ",regex=True, inplace=True)
data2.text_x.replace('\d+'," ",regex=True, inplace=True)

data2.text_y.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
data2.text_y.replace('\s+[a-zA-Z]\s+'," ",regex=True, inplace=True)
data2.text_y.replace('\^[a-zA-Z]\s+'," ",regex=True, inplace=True)
data2.text_y.replace('^b\s+'," ",regex=True, inplace=True)
data2.text_y.replace('\d+'," ",regex=True, inplace=True)

texts = data2[['text_x','text_y']]

for i in texts.columns:
    texts[i] =  texts[i].str.lower()
    
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

msg = []
for i in range(0,len(texts.columns)):
    msg.append(' '.join(x for x in texts.iloc[i,0:2]))
    
lemmatizer = WordNetLemmatizer()
corpus = []


for i in range(0,len(msg)):
    words = msg[i].split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    msg[i] = ' '.join(words)
    corpus.append(words)
    
#companies = data2.author_id_y.unique()


stop_words = set(stopwords.words('english'))
from nltk.tokenize import sent_tokenize, word_tokenize

def main_words(text):
    letters = re.sub("[^a-zA-Z]", " ",text) 
    tokens = word_tokenize(letters)[2:]
    lower_case = [l.lower() for l in tokens]
    filtered_text = list(filter(lambda l: l not in stop_words, lower_case))
    lem = [lemmatizer.lemmatize(t) for t in filtered_text]
    return lem

data2['main words'] = data2.text_x.apply(main_words)

data2.head(2)

from nltk import ngrams

def n_grams(input_1):
    
    bigrams = [' '.join(t) for t in list(zip(input_1, input_1[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_1, input_1[1:], input_1[2:]))]
    return bigrams+trigrams

data2['grams'] = data2['main words'].apply(n_grams)

less = data2[100:300]
less.head()

import collections

def count_words(text):
    cnt = collections.Counter()
    for i in input:
        for word in i:
            cnt[word] += 1
    return cnt


data2[(data2.airline_sentiment == 'neg')][['grams']].apply(count_words)['grams'].most_common(30)

data2[(data2.comp_score == 'pos')][['grams']].apply(count_words)['grams'].most_common(15)
data2.comp_score = data2.comp_score.map({'pos':1,'neg':0})

text = data2['text_x'].values
target = data2['comp_score'].values

text_train, text_test, y_train, y_test = train_test_split(text, target, test_size=0.30, random_state=444)

tfidf = TfidfVectorizer()
tfidf.fit(text_train)

X_train = tfidf.transform(text_train)
X_test  = tfidf.transform(text_test)

models = [rf,cart,logreg,xgb,naive_bayes]

for i in models:
    i.fit(X_train,y_train)
    prediction = i.predict(X_test)
    accuracy = accuracy_score(y_test,prediction)
    roc_auc = roc_auc_score(y_test,prediction)
    print('-'*90)
    print("Model Score for Training data: {}".format(i.score(X_train,y_train)))
    print("Accuracy of the model: {}".format(accuracy_score(y_test,prediction)))
    print("ROC score {}".format(roc_auc_score(y_test,prediction)))
    

cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

doc_matrix = cv.fit_transform(data2['text_x'])

from sklearn.decomposition import LatentDirichletAllocation

LDA = LatentDirichletAllocation(n_components=10)

LDA.fit(doc_matrix)
len(LDA.components_)
first_topic = LDA.components_[0]

# Printing top 10 words from 10 topics
for index,topic in enumerate(LDA.components_):
    print([cv.get_feature_names()[i] for i in topic.argsort()[-10:]])
