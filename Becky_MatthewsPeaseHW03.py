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

## Label is in the first column and reviews in multiple columns. 
#File has a header, "setinment" and "review" on the first row, remove using "with open" and readline()
#Split string into two parts so 1 split after the first comma
RawfileName = "C:/Users/becky/Desktop/Restaurant.csv"
#RawfileName = "C:/Users/becky/Desktop/FilmClean.csv"
#Create a list of labels and a list of reviews
AllReviewsList=[]   #content
AllLabelsList=[]    #labels

#For loop to split reviews and save to lists
with open(RawfileName,'r') as FILE:    # "a", "w"
    FILE.readline() # skip header line - skip row 1
    for row in FILE:   #starts on row 2
        print(row)
        NextLabel,NextReview=row.split(",", 1)
        AllReviewsList.append(NextReview)
        AllLabelsList.append(NextLabel) 
print(AllReviewsList)  
print(AllLabelsList)

#InsertVectorizer
MyCV1 = CountVectorizer(input = "content",   
                        stop_words = "english")
MyFile = "C:/Users/becky/Desktop/FilmClean.csv"
MyFile = "C:/Users/becky/Desktop/Restaurant.csv"

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
                        ##stop_words=["and", "or", "but"],
                        #token_pattern='(?u)[a-zA-Z]+',
                        #token_pattern=pattern,
                        tokenizer = MY_STEMMER,
                        #strip_accents = 'unicode', 
                        lowercase = True
                        )
MyVect_IFIDF = TfidfVectorizer(input = 'content',
                        analyzer = 'word',
                        stop_words ='english',
                        lowercase = True,
                        #binary=True
                        )
MyVect_IFIDF_STEM = TfidfVectorizer(input = 'content',
                        analyzer = 'word',
                        stop_words = 'english',
                        tokenizer = MY_STEMMER,
                        #strip_accents = 'unicode', 
                        lowercase = True,
                        #binary=True
                        )
FinalDF_STEM = pd.DataFrame()
FinalDF_TFIDF = pd.DataFrame()
FinalDF_TFIDF_STEM = pd.DataFrame()

X1 = MyVect_STEM.fit_transform(AllReviewsList)
X2 = MyVect_IFIDF.fit_transform(AllReviewsList)
X3 = MyVect_IFIDF_STEM.fit_transform(AllReviewsList)

#Lisa of Tokens
ColumnNames1 = MyVect_STEM.get_feature_names()
ColumnNames2 = MyVect_IFIDF.get_feature_names()
ColumnNames3 = MyVect_IFIDF_STEM.get_feature_names()

#Place in data table
builderS = pd.DataFrame(X1.toarray(),columns = ColumnNames1)
builderT = pd.DataFrame(X2.toarray(),columns = ColumnNames2)
builderTS = pd.DataFrame(X3.toarray(),columns = ColumnNames3)

#Add column
print("Adding new column....")
builderS["Label"] = AllLabelsList
builderT["Label"] = AllLabelsList
builderTS["Label"] = AllLabelsList

#Convert to Data frame
FinalDF_STEM = FinalDF_STEM.append(builderS)
FinalDF_TFIDF = FinalDF_TFIDF.append(builderT)
FinalDF_TFIDF_STEM = FinalDF_TFIDF_STEM.append(builderTS)

## Replace NA with 0 
FinalDF_STEM = FinalDF_STEM.fillna(0)
FinalDF_TFIDF = FinalDF_TFIDF.fillna(0)
FinalDF_TFIDF_STEM = FinalDF_TFIDF_STEM.fillna(0)

#Remove columns with numbers
MyList=[]
for col in FinalDF_TFIDF.columns:
    #print(col)
    LogR=col.isdigit()  ## any numbers
    if(LogR==True):
        #print(col)
        MyList.append(str(col))
print(MyList)       
FinalDF_TFIDF.drop(MyList, axis = 1, inplace = True)

## Create the testing set with a random sample.
rd.seed(1234)
TrainDF1, TestDF1 = train_test_split(FinalDF_STEM, test_size = 0.3)
print(FinalDF_STEM)
print(TrainDF1)
print(TestDF1)
TrainDF2, TestDF2 = train_test_split(FinalDF_TFIDF, test_size = 0.3)
TrainDF3, TestDF3 = train_test_split(FinalDF_TFIDF_STEM, test_size = 0.3)

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

#Predition vs labeled data
print("\nThe prediction from NB is:")
print(Prediction1)
print("\nThe actual labels are:")
print(Test1Labels)
print("\nThe prediction from NB is:")
print(Prediction2)
print("\nThe actual labels are:")
print(Test2Labels)
print("\nThe prediction from NB is:")
print(Prediction3)
print("\nThe actual labels are:")
print(Test3Labels)

### Confusion Matrix ### 
## The confusion matrix is square and is labels X labels
#The matrix shows rows are the true labels, columns are predicted labels
## it is alphabetical.  The numbers are how many 
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
## columns are the labels in alphabetical order, decinal in the matrix are the prob of being that label
print(np.round(MyModelNB1.predict_proba(TestDF1),2))
print(np.round(MyModelNB2.predict_proba(TestDF2),2))
print(np.round(MyModelNB3.predict_proba(TestDF3),2))

SVM_Model = LinearSVC(C = 10)
SVM_Model.fit(TrainDF1, Train1Labels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
print("SVM prediction:\n", SVM_Model.predict(TestDF1))
print("Actual:")
print(Test1Labels)
SVM_matrix = confusion_matrix(Test1Labels, SVM_Model.predict(TestDF1))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")

##########
TRAIN = TrainDF1
TRAIN_Labels = Train1Labels
TEST = TestDF1
TEST_Labels = Test1Labels

SVM_Model1 = LinearSVC(C = 50)
SVM_Model1.fit(TRAIN, TRAIN_Labels)

print("SVM prediction:\n", SVM_Model1.predict(TEST))
print("Actual:")
print(TEST_Labels)

SVM_matrix = confusion_matrix(TEST_Labels, SVM_Model1.predict(TEST))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")


## RBF
SVM_Model2 = sklearn.svm.SVC(C = 10, kernel = 'rbf', 
                           verbose = True, gamma = "auto")
SVM_Model2.fit(TRAIN, TRAIN_Labels)
print("SVM prediction:\n", SVM_Model2.predict(TEST))
print("Actual:")
print(TEST_Labels)
SVM_matrix = confusion_matrix(TEST_Labels, SVM_Model2.predict(TEST))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")

## POLY
SVM_Model3=sklearn.svm.SVC(C = 10000, kernel = 'poly', degree = 2,
                           gamma = "auto", verbose = True)
print(SVM_Model3)
SVM_Model3.fit(TRAIN, TRAIN_Labels)

print("SVM prediction:\n", SVM_Model3.predict(TEST))
print("Actual:")
print(TEST_Labels)

SVM_matrix = confusion_matrix(TEST_Labels, SVM_Model3.predict(TEST))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")


###Frequency Distribution#######
## Define a function to visualize the TOP words (variables)
def plot_coefficients(MODEL = SVM_Model, COLNAMES = TrainDF1.columns, top_features = 10):
    ## Model if SVM MUST be SVC, RE: SVM_Model=LinearSVC(C=10)
    coef = MODEL.coef_.ravel()
    top_positive_coefficients = np.argsort(coef,axis = 0)[-top_features:]
    top_negative_coefficients = np.argsort(coef,axis = 0)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
    plt.bar(  x =  np.arange(2 * top_features)  , height=coef[top_coefficients], width = .5,  color = colors)
    feature_names = np.array(COLNAMES)
    plt.xticks(np.arange(0, (2*top_features)), feature_names[top_coefficients], rotation = 60, ha = "right")
    plt.show()
plot_coefficients()




####Cleaning the Film Review Dataset##############################
##Read in the file, convert to text
RawfileName = "C:/Users/becky/Desktop/Film.csv"
FILE = open(RawfileName,"r")
##Empty csv file
filename = "FilmCleanSet.csv"
NEWFILE = open(filename,"w")
##Create a column called Label and a column Text
ToWrite = "Label,Text\n"
## Write to empty csv file and then close
NEWFILE.write(ToWrite)
NEWFILE.close()
### open the file for append and create a variable (NEWFILE) that we can use to access and control the file.
NEWFILE = open(filename, "a")
### Blank data frame
MyFinalDF = pd.DataFrame()
OutputFile = "MyOutputFile.txt"
## Open the file with "w" to create it. Then reopen with "a" to add
OUTFILE = open(OutputFile,"w")
OUTFILE.close()
OUTFILE = open(OutputFile,"a") 
for row in FILE:
    RawRow="The next row is: \n" + row +"\n"
    print(RawRow)
    OUTFILE.write(RawRow) 
    row=row.lstrip()  ## strip all spaces from the left
    row=row.rstrip()  ## strip all spaces from the right
    row=row.strip()   ## strip all extra spaces in general
    ## Split up the row of text by space - TOKENIZE IT into a LIST
    Mylist = row.split(" ")
    print(Mylist)
    ##Place the results (cleaned) into a new list
    #Create empty list
    NewList = []
    for word in Mylist:
        PlaceInOutputFile = "The next word BEFORE is: " +  word + "\n"
        OUTFILE.write(PlaceInOutputFile)
        word=word.lower()
        word=word.lstrip()
        word=word.replace(",","")
        word=word.replace(" ","")
        word=word.replace("_","")
        word=re.sub('\+', ' ',word)
        word=re.sub('.*\+\n', '',word)
        word=re.sub('zz+', ' ',word)
        word=word.replace("\t","")
        word=word.replace(".","")
        word=word.strip()
        if word not in ["", "\\", '"', "'", "*", ":", ";"]:
            if len(word) >= 3:
                if not re.search(r'\d', word): ##remove digits
                    NewList.append(word)
                    PlaceInOutputFile = "The next word AFTER is: " +  word + "\n"
                    OUTFILE.write(PlaceInOutputFile)
    print(NewList)    
    print(NewList[-1])  ## last element (label)
    label=NewList[-1]
    if "pos" in label:
        label="pos"
    else:
        label="neg"
    PlaceInOutputFile = "\nThe label is: " +  label + "\n"
    OUTFILE.write(PlaceInOutputFile)
    NewList.pop() ## removes last item
    Text=" ".join(NewList)
    Text=Text.replace("\\n","")
    Text=Text.strip("\\n")
    Text=Text.replace("\\'","")
    Text=Text.replace("\\","")
    Text=Text.replace('"',"")
    Text=Text.replace("'","")
    Text=Text.replace("s'","")
    Text=Text.lstrip()  
    #Write to the NEWFILE...
    OriginalRow = "ORIGINAL" + RawRow
    OUTFILE.write(OriginalRow)
    ToWrite=label+","+Text+"\n"
    NEWFILE.write(ToWrite)
    OUTFILE.write(ToWrite)
FILE.close()  
NEWFILE.close()
OUTFILE.close()
## Read the new csv file you created into a DF or into CounterVectorizer
MyTextDF = pd.read_csv("C:/Users/becky/Desktop/FilmClean.csv")
## remove any rows with NA
MyTextDF = MyTextDF.dropna(how = 'any', axis  = 0)  ## axis 0 is rowwise
##Save Labels
MyLabel = MyTextDF["Label"]
## Remove the labels from the DF
DF_noLabel = MyTextDF.drop(["Label"], axis = 1)  #axis 1 is column
## Create a list where each element in the list is a row from the file/DF
print("length: ", len(DF_noLabel))
#Build the list
MyList=[]  
for i in range(0,len(DF_noLabel)):
    NextText=DF_noLabel.iloc[i,0]  
    MyList.append(NextText)
print(MyList)
########## 
MycountVect = CountVectorizer(input = "content")
CV = MycountVect.fit_transform(MyList)
MyColumnNames = MycountVect.get_feature_names()
VectorizedDF_Text = pd.DataFrame(CV.toarray(),columns = MyColumnNames)
print(VectorizedDF_Text)
### Put the labels back
## Make copy
print(MyLabel)
print(type(MyLabel))  
NEW_Labels = MyLabel.to_frame()   #index to 0
print(type(NEW_Labels))
NEW_Labels.index = NEW_Labels.index-1
print(NEW_Labels)
LabeledCLEAN_DF = VectorizedDF_Text
LabeledCLEAN_DF["LABEL"] = NEW_Labels
print(LabeledCLEAN_DF)





