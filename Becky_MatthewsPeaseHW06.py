# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 21:13:20 2021

@author: Becky Matthews-Pease
"""
############################################################
#Packages
import nltk
import sklearn
import re  
import os
import string
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords


RawfileName = "C:/Users/becky/Desktop/deception.csv"
AllReviewsList=[]   #content
AllLabelsList=[]    #labels

#For loop to split reviews and save to lists
with open(RawfileName,'r') as FILE:
    FILE.readline() 
    for row in FILE:   
        print(row)
        NextLabel,NextReview=row.split(",", 1)
        AllReviewsList.append(NextReview)
        AllLabelsList.append(NextLabel) 
print(AllReviewsList)  
print(AllLabelsList)

#InsertVectorizer
MyCV1 = CountVectorizer(input = "content",   
                        stop_words = "english")
MyFile = "C:/Users/becky/Desktop/deception.csv"

#Stemmer
STEMMER = PorterStemmer()

# Use NLTK's PorterStemmer in a function
def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(word) for word in words]
    return words

#Create different vectorizers:
MyVect_STEM = CountVectorizer(input = 'content',
                        analyzer = 'word',
                        stop_words = 'english',
                        tokenizer = MY_STEMMER, 
                        lowercase = True
                        )
MyVect_IFIDF_STEM = TfidfVectorizer(input = 'content',
                        analyzer = 'word',
                        stop_words = 'english',
                        tokenizer = MY_STEMMER,
                        lowercase = True
                        )
MyVect_STEM_Bern=CountVectorizer(input='content',
                        analyzer = 'word',
                        stop_words='english',
                        tokenizer=MY_STEMMER,
                        lowercase = True,
                        binary=True    
                        )

FinalDF_STEM = pd.DataFrame()
FinalDF_TFIDF_STEM = pd.DataFrame()
FinalDF_STEM_Bern = pd.DataFrame()

X1 = MyVect_STEM.fit_transform(AllReviewsList)
X2 = MyVect_IFIDF_STEM.fit_transform(AllReviewsList)
XB = MyVect_STEM_Bern.fit_transform(AllReviewsList)

#Lisa of Tokens
ColumnNames1 = MyVect_STEM.get_feature_names()
ColumnNames2 = MyVect_IFIDF_STEM.get_feature_names()
ColumnNamesB = MyVect_STEM_Bern.get_feature_names()
NumFeatures4 = len(ColumnNamesB)
#Place in data table
builderS = pd.DataFrame(X1.toarray(),columns = ColumnNames1)
builderTS = pd.DataFrame(X2.toarray(),columns = ColumnNames2)
builderB = pd.DataFrame(XB.toarray(),columns = ColumnNamesB)
#Add column
builderS["Label"] = AllLabelsList
builderTS["Label"] = AllLabelsList
builderB["Label"] = AllLabelsList
#Convert to Data frame
FinalDF_STEM = FinalDF_STEM.append(builderS)
FinalDF_TFIDF_STEM = FinalDF_TFIDF_STEM.append(builderTS)
FinalDF_STEM_Bern = FinalDF_STEM_Bern.append(builderB)
## Replace NA with 0 
FinalDF_STEM = FinalDF_STEM.fillna(0)
FinalDF_TFIDF_STEM = FinalDF_TFIDF_STEM.fillna(0)
FinalDF_STEM_Bern = FinalDF_STEM_Bern.fillna(0)
#Remove columns with numbers
MyList=[]
for col in FinalDF_TFIDF_STEM.columns:
    LogR=col.isdigit()  
    if(LogR==True):
        MyList.append(str(col))
print(MyList)       
FinalDF_TFIDF_STEM.drop(MyList, axis = 1, inplace = True)

def RemoveNums(SomeDF):
    temp = SomeDF
    MyList = []
    for col in temp.columns:
        Logical2=str.isalpha(col) 
        if(Logical2==False):
            MyList.append(str(col))
    temp.drop(MyList, axis = 1, inplace = True)
    return temp

FinalDF_STEM = RemoveNums(FinalDF_STEM)
FinalDF_STEM_Bern = RemoveNums(FinalDF_STEM_Bern)
FinalDF_TFIDF_STEM = RemoveNums(FinalDF_TFIDF_STEM) 
print(FinalDF_STEM)
print(FinalDF_TFIDF_STEM)
print(FinalDF_STEM_Bern) 




## Create the testing set with a random sample.
rd.seed(1234)
sklearn.model_selection.StratifiedKFold(n_splits = 10, shuffle = False, random_state = None)
TrainDF1, TestDF1 = train_test_split(FinalDF_STEM, test_size = 0.3, random_state = 10)
TrainDF2, TestDF2 = train_test_split(FinalDF_TFIDF_STEM, test_size = 0.3, random_state = 10)
TrainDF3, TestDF3 = train_test_split(FinalDF_STEM_Bern, test_size = 0.3)
print(FinalDF_STEM)
print(TrainDF1)
print(TrainDF2)
print(TrainDF3)
print(TestDF1)
print(TestDF2)
print(TestDF3)

## Separate and save labels; Remove labels from test set
Test1Labels = TestDF1["Label"]
Test2Labels = TestDF2["Label"]
Test3Labels = TestDF3["Label"]
## remove labels
TestDF1 = TestDF1.drop(["Label"], axis = 1)
TestDF2 = TestDF2.drop(["Label"], axis = 1)
TestDF3 = TestDF3.drop(["Label"], axis = 1)
#Remove from Training set
Train1Labels = TrainDF1["Label"]
Train2Labels = TrainDF2["Label"]
Train3Labels = TrainDF3["Label"]
## remove labels
TrainDF1 = TrainDF1.drop(["Label"], axis = 1)
TrainDF2 = TrainDF2.drop(["Label"], axis = 1)
TrainDF3 = TrainDF3.drop(["Label"], axis = 1)
### Create the NB model ###
MyModelNB1 = MultinomialNB()
MyModelNB2 = MultinomialNB()
MyModelNB3 = MultinomialNB()

MyModelNB1.fit(TrainDF1, Train1Labels)
MyModelNB2.fit(TrainDF2, Train2Labels)
MyModelNB3.fit(TrainDF3, Train3Labels)

Prediction1 = MyModelNB1.predict(TestDF1)
Prediction2 = MyModelNB2.predict(TestDF2)
Prediction3 = MyModelNB3.predict(TestDF3)

### Predition vs labeled data ### 
print("\nThe prediction from NB is:")
print(Prediction1)
print("\nThe actual labels are:")
print(Test1Labels)
print("\nThe prediction from NB is:")
print(Prediction2)
print("\nThe actual labels are:")
print(Test2Labels)
print("\nThe prediction is:")
print(Prediction3)
print("\nThe actual labels are:")
print(Test3Labels)


### Confusion Matrix ### 
cnf_matrix1 = confusion_matrix(Test1Labels, Prediction1)
print("\nThe confusion matrix is:")
print(cnf_matrix1)

cnf_matrix2 = confusion_matrix(Test2Labels, Prediction2)
print("\nThe confusion matrix is:")
print(cnf_matrix2)

cnf_matrix3 = confusion_matrix(Test3Labels, Prediction3)
print("\nThe confusion matrix is:")
print(cnf_matrix3)

### Prediction Probabilities ###
print(np.round(MyModelNB1.predict_proba(TestDF1),2))
print(np.round(MyModelNB2.predict_proba(TestDF2),2))
print(np.round(MyModelNB3.predict_proba(TestDF3),2))

SVM_Model = LinearSVC(C = 10)
SVM_Model.fit(TrainDF1, Train1Labels)
print("SVM prediction:\n", SVM_Model.predict(TestDF1))
print("Actual:")
print(Test1Labels)
SVM_matrix = confusion_matrix(Test1Labels, SVM_Model.predict(TestDF1))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")

SVM_Model.fit(TrainDF2, Train2Labels)
print("SVM prediction:\n", SVM_Model.predict(TestDF2))
print("Actual:")
print(Test2Labels)
SVM_matrix = confusion_matrix(Test2Labels, SVM_Model.predict(TestDF2))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")

SVM_Model.fit(TrainDF3, Train3Labels)
print("SVM prediction:\n", SVM_Model.predict(TestDF3))
print("Actual:")
print(Test3Labels)
SVM_matrix = confusion_matrix(Test3Labels, SVM_Model.predict(TestDF3))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")


###Frequency Distribution#######
def plot_coefficients(MODEL = SVM_Model, COLNAMES = TrainDF1.columns, top_features = 10):
    coef = MODEL.coef_.ravel()
    top_positive_coefficients = np.argsort(coef,axis = 0)[-top_features:]
    top_negative_coefficients = np.argsort(coef,axis = 0)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
    plt.bar(  x =  np.arange(2 * top_features)  , height=coef[top_coefficients], width = .5,  color = colors)
    feature_names = np.array(COLNAMES)
    plt.xticks(np.arange(0, (2*top_features)), feature_names[top_coefficients], rotation = 60, ha = "right")
    plt.show()
plot_coefficients()


###Create Word Cloud#######
mystopwords = set(stopwords.words('english'))
# Read a text file and calculate frequency of words in it
with open("C:/Users/becky/Desktop/deceptions.txt", "r") as f:
    words = f.read().split()
data = dict()
for word in words:
    word = word.lower()
    if word in stop_words:
        continue
    data[word] = data.get(word, 0) + 1
word_cloud = WordCloud(
    background_color = background_color,
    width = width,
    height=height, 
    collocations = False,
    stopwords = mystopwords
)
word_cloud.generate_from_frequencies(data)
word_cloud.to_file('image7.png')





