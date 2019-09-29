import pandas as pd
import numpy as np
import re

stock = pd.read_csv(r'C:\Users\ABHAY JAMWAL\Desktop\Classes\Projects\Combined_News_DJIA.csv\Combined_News_DJIA.csv',
                    encoding = "ISO-8859-1")

stock.isna().sum()

stock.drop(277,inplace=True)
stock.drop(348,inplace=True)
stock.drop(681,inplace=True)

import nltk
import nltk.corpus
from nltk.corpus import stopwords
from nltk import PorterStemmer


train = stock.iloc[1:1200,:]
test = stock.iloc[1200:,:]

stock_onlycols_copy = stock_onlycols.copy()
stock_onlycols=train.iloc[:,2:27]
stock_onlycols.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
stock_onlycols.replace('\s+[a-zA-Z]\s+'," ",regex=True, inplace=True)
stock_onlycols.replace('\^[a-zA-Z]\s+'," ",regex=True, inplace=True)
stock_onlycols.replace('^b\s+'," ",regex=True, inplace=True)
stock_onlycols.replace('\d+'," ",regex=True, inplace=True)


#stock_onlycols['Top1'] = stock_onlycols['Top1'].str.lower()

for i in stock_onlycols.columns:
    stock_onlycols[i] =  stock_onlycols[i].str.lower()
    
len(stock_onlycols.columns)
#from nltk.corpus import stopwords
#stop = stopwords.words('english')

#stock_onlycols['Top1'] = stock_onlycols['Top1'].apply(lambda x: [item for item in x if item not in stop])
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer


headlines = []
for i in range(0,len(stock_onlycols.columns)):
    headlines.append(' '.join(str(x) for x in stock_onlycols.iloc[i,0:25]))
corpus = []
for i in range(0,len(headlines)):
    words = headlines[i].split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    headlines[i] = ' '.join(words)
    corpus.append(words)
 
    
  
lemmatizer = WordNetLemmatizer()

from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features=2500)
X = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False,max_features=1199).fit_transform(corpus).toarray()
y = train['Label']
X = X.reshape(1199,25)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import accuracy_score,f1_score
print("Accuracy of model: {}".format(accuracy_score(y_test,y_pred)))


