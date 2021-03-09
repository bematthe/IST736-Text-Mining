# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 22:04:51 2021

@author: becky
"""
import os
import nltk
import sklearn
import string
import collections
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from wordcloud import WordCloud



## Create CountVectorizer 
MyCV1 = CountVectorizer(input = "filename",   
                        stop_words = "english")

PosNegList = ["C:\\Users\\becky\\Desktop\\Corpus\\AmanoPos.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\AmedeiChuaoPos.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\AmedeiMedPos.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\AretePos.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\BessonePos.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\CoppeneurPos.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\GuittardPos.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\HotelChocolatPos.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\PatricPos.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\Pitch DarkPos.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\RoguePos.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\ValrhonaPos.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\ZotterPos.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\AmedeiNeg.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\AnahataNeg.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\AskinosieNeg.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\CallebautNeg.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\DantaNeg.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\MarouNeg.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\MesocacaoNeg.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\PierreMarcoliniNeg.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\QuatuNeg.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\SacredChocNeg.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\Soma2Neg.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\TerriorNeg.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\ValrhonaPos.txt",
        "C:\\Users\\becky\\Desktop\\Corpus\\ZotterNeg.txt",]

## Apply CountVectorizer
MyMat = MyCV1.fit_transform(PosNegList)
print(type(MyMat))
print(MyMat)

##Column names
MyCols = MyCV1.get_feature_names()
print(MyCols)

## Convert to array
MyDF = pd.DataFrame(MyMat.toarray(), columns = MyCols)
print(MyDF)

## Set path to Corpus
path = "C:\\Users\\becky\\Desktop\\Corpus\\"
print("calling os...")
print(os.listdir(path))

##Save list of files 
FileNameList = os.listdir(path)
print(type(FileNameList))
print(FileNameList)

##Empty list for file paths
ListOfCompleteFilePaths=[]  

## Empty list for file names for labels 
ListOfJustFileNames=[]

for name in os.listdir(path):
    print(path+ "/" + name)
    
## Loop and add to list
for name in os.listdir(path):
    print(path+ "/" + name)  
    nextfile=path+ "/" + name   
    ListOfCompleteFilePaths.append(nextfile)  
    
## Get file name w/o extension
    nextnameL=name.split(".")  
    print(nextnameL[0])  
    ListOfJustFileNames.append(nextnameL[0])

## View file lists and name of files
print("List of complete file paths...\n")
print(ListOfCompleteFilePaths)
print("list of just names:\n")
print(ListOfJustFileNames)

## Convert to DTF then data frame
MyCV3 = CountVectorizer(input = 'filename', 
                        stop_words = 'english',
                        )
pathMatrix = MyCV3.fit_transform(ListOfCompleteFilePaths)

## View DTM
print(pathMatrix)  

## Get column names 
ColumnNames3 = MyCV3.get_feature_names()
print(ColumnNames3) 

## Convert to array  then data frame
ChocolateCorpus = pd.DataFrame(pathMatrix.toarray(),columns = ColumnNames3)
print(ChocolateCorpus)  

## Empty dictionary to add labels
MyDict={}

## Loop to add to dictionary
for i in range(0, len(ListOfJustFileNames)):
    MyDict[i] = ListOfJustFileNames[i]     
print("MY DICT:", MyDict)

## Labels to categories
for i in range(0, len(ListOfJustFileNames)):
    MyDict[i] = MyDict[i].rstrip('ly')
print("MY DICT:", MyDict) 
 
## Place in data frame   
ChocolateCorpus = ChocolateCorpus.rename(MyDict, axis = "index")
print(ChocolateCorpus)

# Convert DataFrame to matrix
MyChocolateMatrix = ChocolateCorpus.values
print(type(MyChocolateMatrix))
print(MyChocolateMatrix)

## Access columns
for name in ColumnNames3:
    print(name)
for name in ColumnNames3:
    print(ChocolateCorpus[name])

## Combine terms
name1 = "wonderful"
name2 = "wonderfully"
if(name1 == name2):
    print("TRUE")
else:
    print("FALSE")

name1 = name1.rstrip("y")
print(name1)
if(name1 == name2):
    print("TRUE")
else:
    print("FALSE")
 
 ##User specified stop words   
print("The initial column names:\n", ColumnNames3)
print(type(ColumnNames3)) 
MyStops = ["also", "and", "are", "you", "of", "let", "to", "a", "they", "not", "the", "for", "why", "there", "one", "cacao", "chocolate", "bar", "flavor", "flavors", "which"]   

CleanDF = ChocolateCorpus
print("START\n", CleanDF)

## Build a new columns list
ColNames=[]

for name in ColumnNames3:
    if ((name in MyStops) or (len(name)<3)):
        CleanDF = CleanDF.drop([name], axis=1)
    else:
        ColNames.append(name)            
print("The ending column names:\n", ColNames)

#Clean data frame
for name1 in ColNames:
    for name2 in ColNames:
        if(name1 == name2):
            print("skip")
        elif(name1.rstrip("y") in name2):
            print("combining: ", name1, name2)
            print(ChocolateCorpus[name1])
            print(ChocolateCorpus[name2])
            print(ChocolateCorpus[name1] + ChocolateCorpus[name2])        
            CleanDF[name1] = CleanDF[name1] + CleanDF[name2]            
            CleanDF=CleanDF.drop([name2], axis = 1)
print(CleanDF.columns.values)
print(CleanDF["wonderful"])
print(CleanDF)

# Convert data frame to matrix
MyMatrixClean = CleanDF.values
print(type(MyMatrixClean))
print(MyMatrixClean)


##K means with 2 clusters
kmeans_object = sklearn.cluster.KMeans(n_clusters = 2)
kmeans_object.fit(MyMatrixClean)
# Get cluster assignment labels
labels = kmeans_object.labels_
print("K means with k = 2\n", labels)
# Format results as a data frame
Myresults = pd.DataFrame([CleanDF.index,labels]).T
print("k means RESULTS\n", Myresults)

#Word cloud by frequency
fullText = "sublime kind awesome perfect enduring sublime awesome stunning sublime gorgeous refreshing treat awesome classic awesome sublime robust delicious irredeemable flatliner fantastic wonderful bright awesome poor technique abuses worshipful perfect awesome incongruous discordant vague murky haze soap contaminated unpleasant phenomenal awesome well-harmonized greatness astonishing wonderful awesome sublime malpractice pity sublime exquisite precision awesome tainted mold mildew unsavory inexcusable scared bitter foul awesome exceptional sublime mass market foul generic boring mass market undeveloped downcast demoralized off-origin off-flavor bad mix beautiful sublime awesome control phenomenal extraordinary awesome exceptional sublime misleads poor shoddy abysmal" 
chocWordCloud = WordCloud(background_color="white").generate(fullText)
plt.figure(figsize = (20,20))
plt.imshow(wordcloud_spam, interpolation = 'bilinear')
plt.axis("off")
plt.show()

##Create histogram
word_list = ["sublime", "kind", "awesome", "perfect", "enduring", "sublime", "awesome", "stunning", "sublime", "gorgeous", "refreshing", "treat", "awesome", "classic", "awesome", "sublime", "robust", "delicious", "irredeemable", "flatliner", "fantastic", "wonderful", "bright", "awesome", "poor technique", "abuses", "worshipful", "perfect", "awesome", "incongruous", "discordant", "vague", "murky", "haze", "soap", "contaminated", "unpleasant", "phenomenal", "awesome", "well-harmonized", "greatness", "astonishing", "wonderful", "awesome", "sublime", "malpractice", "pity", "sublime", "exquisite", "precision", "awesome", "tainted", "mold", "mildew", "unsavory", "inexcusable", "scared", "bitter", "foul", "awesome", "exceptional", "sublime", "mass market", "foul", "generic", "boring", "mass market", "undeveloped", "downcast", "demoralized", "off-origin", "off-flavor", "bad mix", "beautiful", "sublime", "awesome", "control", "phenomenal", "extraordinary", "awesome", "exceptional", "sublime", "misleads", "poor", 
"shoddy", "abysmal"]
counts = Counter(word_list)
labels, values = zip(*counts.items())
##Sort in descending order 
indSort = np.argsort(values)[::-1]
# ReShape as Array
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]
indexes = np.arange(len(labels))
bar_width = 0.35
plt.bar(indexes, values)
# add labels
plt.xticks(indexes + bar_width, labels)
plt.xticks(rotation = 45, ha = 'right')
plt.show()


##Word Sentiments
vader = SentimentIntensityAnalyzer()
#Positive Words
vader.polarity_scores("awesome")  #.62
vader.polarity_scores("astonishing") #0.0
vader.polarity_scores("bright") #0.4404
vader.polarity_scores("classic") #0.0
vader.polarity_scores("control") #0.0
vader.polarity_scores("delicious") #0.5719
vader.polarity_scores("enduring") #0.0
vader.polarity_scores("exceptional") #0.0
vader.polarity_scores("exquisite") #0.0
vader.polarity_scores("extraordinary") #0.0
vader.polarity_scores("fantastic") #0.5574
vader.polarity_scores("gorgeous") #0.6124
vader.polarity_scores("greatness") #0.0
vader.polarity_scores("perfect") #0.5719
vader.polarity_scores("phenomenal") #0.0
vader.polarity_scores("precision") #0.0
vader.polarity_scores("refreshing") #0.0
vader.polarity_scores("robust") #0.34
vader.polarity_scores("stunning") #0.3818
vader.polarity_scores("sublime") #0.0
vader.polarity_scores("treat") #0.4019
vader.polarity_scores("well-harmonized") #0.0
vader.polarity_scores("wonderful") #0.5719
vader.polarity_scores("worshipful") #0.179

##Negative List
vader.polarity_scores("abuses") #-0.5574
vader.polarity_scores("abysmal")  #0.0
vader.polarity_scores("bad mix") #-0.5423
vader.polarity_scores("bitter") #-0.4215
vader.polarity_scores("boring") #-0.3182
vader.polarity_scores("contaminated") #0.0
vader.polarity_scores("demoralized") #-0.3818
vader.polarity_scores("discordant") #0.0
vader.polarity_scores("downcast") #-0.4215
vader.polarity_scores("flatliner")  #0.0
vader.polarity_scores("foul") #0.0
vader.polarity_scores("generic") #0.0
vader.polarity_scores("haze")  #0.0
vader.polarity_scores("incongruous") #0.0
vader.polarity_scores("inexcusable") #0.0
vader.polarity_scores("irredeemable") #0.0
vader.polarity_scores("malpractice") #0.0
vader.polarity_scores("mass market") #0.0
vader.polarity_scores("mildew") #0.0
vader.polarity_scores("misleads") #0.0
vader.polarity_scores("mold") #0.0
vader.polarity_scores("murky") #0.0
vader.polarity_scores("off-origin") #0.0
vader.polarity_scores("off-flavor") #0.0
vader.polarity_scores("pity") #-0.296
vader.polarity_scores("poor") #-0.4767
vader.polarity_scores("poor technique") #-0.4767
vader.polarity_scores("scared") #-0.4404
vader.polarity_scores("shoddy") #0.0
vader.polarity_scores("soap") #0.0
vader.polarity_scores("tainted") #0.0
vader.polarity_scores("undeveloped") #0.0
vader.polarity_scores("unpleasant") #-0.4767
vader.polarity_scores("unsavory")  #-0.4404
vader.polarity_scores("vague") #-0.1027


sid = SentimentIntensityAnalyzer()
pos_word_list = []
neg_word_list = []
for word in word_list:
    if (sid.polarity_scores(word)['compound']) >= 0.1:
        pos_word_list.append(word)
    elif (sid.polarity_scores(word)['compound']) <= -0.0:
        neg_word_list.append(word)
    else:
        neu_word_list.append(word)                
print('Positive :', pos_word_list)          
print('Negative :', neg_word_list)  


##Create histogram of positive words
counts = Counter(pos_word_list)
labels, values = zip(*counts.items())
##Sort in descending order 
indSort = np.argsort(values)[::-1]
# ReShape as Array
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]
indexes = np.arange(len(labels))
bar_width = 0.35
plt.bar(indexes, values)
# add labels
plt.xticks(indexes + bar_width, labels)
plt.xticks(rotation = 45, ha = 'right')
pos = plt.show()

##Create histogram of positive words
counts = Counter(neg_word_list)
labels, values = zip(*counts.items())
##Sort in descending order 
indSort = np.argsort(values)[::-1]
# ReShape as Array
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]
indexes = np.arange(len(labels))
bar_width = 0.35
plt.bar(indexes, values)
# add labels
plt.xticks(indexes + bar_width, labels)
plt.xticks(rotation = 45, ha = 'right')
plt.show()









    
 
