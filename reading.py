import os
import nltk
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfTransformer
import numpy
from sklearn.feature_extraction.text import CountVectorizer
import re
import codecs
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.porter import *
import string
from string import digits

#path = '/opt/datacourse/data/parts'
token_dict = {}
stemmer = PorterStemmer()
#nltk.download('punkt')

qbfile = open("eng.txt","r")
count = 0
lines = []
index = []
page = ""

for aline in qbfile.readlines():

    lowers = str(aline.lower())
    no_punctuation = lowers.translate(None, string.punctuation)
    no_digits = no_punctuation.translate(None, digits)
    tokens = nltk.word_tokenize(no_digits)
    stemmed = []
    #for item in tokens:
    #    stemmed.append(stemmer.stem(item))
    alines = tokens 
    values = aline.split()
    
    if len(values) >=1 and values[0] == '[PAGE' and values[1] == 'SEPRATOR]':
      count=count+1

      lines.append(page)
      index.append(count)
      page=""
    elif len(values) >= 0 and (str(aline) =="\n" or str(aline) == " \n"):
      continue
    else :
      page = page + " " + str(alines)
      
print count
#data_frame = DataFrame(lines, index=index)
data_frame = DataFrame(lines,index)
qbfile.close()
data_frame.columns = ['text']


count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(data_frame['text'].values)
#print count_vectorizer.vocabulary_
#print counts
transformer = TfidfTransformer(smooth_idf=False)
stop_words = nltk.corpus.stopwords.words('english')
tfidf = transformer.fit_transform(counts).toarray()
#print tfidf

num_clusters = 4
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf)
clusters = km.labels_.tolist()
#print clusters
films = { 'index': index,'cluster' : clusters  }
frame = pd.DataFrame(films, index = [clusters] , columns = ['index','cluster'])
print frame
#print frame['cluster'].value_counts()
#from __future__ import print_function

#print("Top terms per cluster:")
#print()
#sort cluster centers by proximity to centroid
#order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

#for i in range(num_clusters):
#    print("Cluster %d words:" % i, end='')
#    
#    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
#        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
#    print() #add whitespace
#    print() #add whitespace
#    
#    print("Cluster %d titles:" % i, end='')
#    for title in frame.ix[i]['title'].values.tolist():
#        print(' %s,' % title, end='')
#    print() #add whitespace
#    print() #add whitespace
    
#print()
#print()


