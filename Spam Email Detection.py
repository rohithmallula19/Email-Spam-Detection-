#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string


# In[2]:


df= pd.read_csv("C:/Users/itzro/OneDrive/Desktop/emails.csv")


# In[4]:


df.head(10)


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.drop_duplicates(inplace=True)
print(df.shape)


# In[8]:


print(df.isnull().sum())


# In[9]:


# download the stopwords package
nltk.download("stopwords")


# In[10]:


def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean
# to show the tokenization
df['text'].head().apply(process)


# In[11]:


from sklearn.feature_extraction.text import CountVectorizer
message = CountVectorizer(analyzer=process).fit_transform(df['text'])


# In[12]:


#split the data into 80% training and 20% testing
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(message, df['spam'], test_size=0.20, random_state=0)
# To see the shape of the data
print(message.shape)


# In[13]:


# create and train the Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(xtrain, ytrain)


# In[14]:


print(classifier.predict(xtrain))
print(ytrain.values)


# In[15]:


# Evaluating the model on the training data set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = classifier.predict(xtrain)
print(classification_report(ytrain, pred))
print()
print("Confusion Matrix: \n", confusion_matrix(ytrain, pred))
print("Accuracy: \n", accuracy_score(ytrain, pred))


# In[16]:


#print the predictions
print(classifier.predict(xtest))
#print the actual values
print(ytest.values)


# In[17]:


# Evaluating the model on the training data set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = classifier.predict(xtest)
print(classification_report(ytest, pred))
print()
print("Confusion Matrix: \n", confusion_matrix(ytest, pred))
print("Accuracy: \n", accuracy_score(ytest, pred))

