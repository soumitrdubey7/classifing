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
import csv
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

token_dict = {}
stemmer = PorterStemmer()

import glob
lines = []
index = []
file_name = []
page_labels = []
labels_add = []
page = ""

for file in glob.glob("*.txt"):
  #print(file)
  qbfile = open(file,"r")
  count = 0
  page =""
  labels =[]
  labels_string =""
  for aline in qbfile.readlines():
    lowers = str(aline.lower())
    no_punctuation = lowers.translate(None, string.punctuation)
    no_digits = no_punctuation.translate(None, digits)
    tokens = nltk.word_tokenize(no_digits)
    alines = tokens 
    for i in range(0,len(tokens)) :
      if (tokens[i] == 'invoice' or tokens[i] == 'commercial' or tokens[i] == 'packing' or tokens[i] == 'list' or tokens[i] == 'lading' or tokens[i] == 'bill') :
        labels.append(tokens[i])  
    values = aline.split()
    
    if len(values) >=1 and values[0] == '[PAGE' and values[1] == 'SEPRATOR]':
      count=count+1

      lines.append(page)
      index.append(count)
      file_name.append(file)
      page_labels.append(labels)
      labels_add.append(labels_string)

      page=""
      labels=[]
      labels_string = ""

    elif len(values) >= 0 and (str(aline) =="\n" or str(aline) == " \n"):
      continue
    else :
      page = page + " " + str(alines)
      if str(labels) == "[]":
        continue
      else :
        labels_string = labels_string + " " + str(labels)
  qbfile.close()    
  #print count

labels_frame = DataFrame({'lines':pd.Series(labels_add),'name': pd.Categorical(file_name)})
labels_frame['page_number'] = index
labels_frame.columns = ['labels','file_name','page_count']
labels_frame.set_index(['file_name','page_count'])
labels_frame.to_csv(r'~/Downloads/invoices/classifing/other_txt_files/labels_frame.csv')


data_frame = DataFrame({'lines':pd.Series(lines),'name': pd.Categorical(file_name)})
data_frame['page_number'] = index
data_frame.columns = ['text','file_name','page_count']
data_frame.set_index(['file_name','page_count'])
data_frame.to_csv(r'~/Downloads/invoices/classifing/other_txt_files/data_frame.txt')

count_vectorizer = CountVectorizer()
labels_use = 0
if labels_use == 0 :
  counts = count_vectorizer.fit_transform(data_frame['text'].values)
else :
  counts = count_vectorizer.fit_transform(labels_frame['labels'].values)

counts2 = DataFrame(counts.A, columns=count_vectorizer.vocabulary_)
transformer = TfidfTransformer(smooth_idf=False)
stop_words = nltk.corpus.stopwords.words('english')
tfidf = transformer.fit_transform(counts).toarray()
######################################## k-means ###############################
num_clusters = 4
km = KMeans(n_clusters=num_clusters,max_iter = 5000,random_state=10)
km.fit(tfidf)
clusters = km.labels_.tolist()

df = pd.read_csv('labels_original.csv')
original_label = df['cluster_out']
print "Accuracy of k-means is", accuracy_score(original_label,clusters)
  
############################### debugging purpose ###############################
frame_buff = { 'page_count': index,'cluster' : clusters }
frame = pd.DataFrame(frame_buff, index = [file_name] , columns = ['page_count','cluster'])

if labels_use == 0 :
  frame.to_csv(r'~/Downloads/invoices/classifing/other_txt_files/output.txt')
  counts2.to_csv(r'~/Downloads/invoices/classifing/other_txt_files/counts2.csv')
  numpy.savetxt('tfidf.csv', tfidf, delimiter=",")
else :
  frame.to_csv(r'~/Downloads/invoices/classifing/other_txt_files/output_label.txt')
  counts2.to_csv(r'~/Downloads/invoices/classifing/other_txt_files/counts2_label.csv')
  numpy.savetxt('tfidf_label.csv', tfidf, delimiter=",")

########################### SVM ##################################################
X_train, X_test, y_train, y_test = train_test_split(
        tfidf, original_label, test_size=0.2, random_state=42
    )
def train_svm(X, y):
    svm = SVC(C=1000000.0, kernel='rbf')
    svm.fit(X, y)
    return svm

svm = train_svm(X_train, y_train)

pred = svm.predict(X_test)

print "Accuracy of SVM is" , accuracy_score(y_test,pred)
#################################### KNN ###################################################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print "Accuracy of kNN is" , accuracy_score(y_test, pred)

############################### decision tree ################################################33
from sklearn.tree import DecisionTreeClassifier
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
y_pred_en = clf_entropy.predict(X_test)
print "Accuracy of decision tree is " , accuracy_score(y_test,y_pred_en)
