
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 3
# 
# In this assignment you will explore text message data and create models to predict if a message is spam or not. 

# In[218]:

import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


# In[219]:

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# ### Question 1
# What percentage of the documents in `spam_data` are spam?
# 
# *This function should return a float, the percent value (i.e. $ratio * 100$).*

# In[220]:

def answer_one():
    
    # count of non-spam labels
    zeros = spam_data.target.value_counts()[0]
    
    # count of spam labels
    ones = spam_data.target.value_counts()[1]
    
    return  ones / (zeros + ones) * 100 #Your answer here


# In[221]:

answer_one()


# ### Question 2
# 
# Fit the training data `X_train` using a Count Vectorizer with default parameters.
# 
# What is the longest token in the vocabulary?
# 
# *This function should return a string.*

# In[222]:

from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    
    # vectorize all the tokens
    vect = CountVectorizer().fit(X_train)
    
    # get all the tokens
    tokens = vect.get_feature_names()
    
    # convert to pandas dataframe
    df = pd.DataFrame(tokens, columns=['tokens'])
    
    # find the longest token by length
    longest = df.tokens.str.len().sort_values().index
    
    df.index = df['tokens'].str.len()
    result = df.sort_index(ascending=False).reset_index(drop=True)
    
    return result.iloc[0][0] #Your answer here


# In[223]:

answer_two()


# ### Question 3
# 
# Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
# 
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[245]:

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer

def answer_three():
    
    # vectorize all the tokens
    vect = CountVectorizer().fit(X_train)
    
    # transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train)
    
    # create a classifier object
    clf = naive_bayes.MultinomialNB(alpha=0.1)
    
    # fit the model
    clf.fit(X_train_vectorized, y_train)
    
    # Predict the transformed test documents
    predictions = clf.predict(vect.transform(X_test))

    #find the AUC
    AUC = roc_auc_score(y_test, predictions)
    
    return AUC #Your answer here


# In[246]:

answer_three()


# ### Question 4
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.
# 
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
# 
# Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.
# 
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 
# 
# *This function should return a tuple of two series
# `(smallest tf-idfs series, largest tf-idfs series)`.*

# In[243]:

from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    
    # Fit the TfidfVectorizer to the training data without specifying a minimum
    vect = TfidfVectorizer().fit(X_train)
    
    # transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train)
    
    #get the sorted score index
    values = X_train_vectorized.max(0).toarray()[0]
    sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
    
    # get the feature names as numpy array
    feature_names = np.array(vect.get_feature_names())
    
    # get the list of 20 features with the smallest coeeficients (): Not associated with spam
    smallest_coeff = feature_names[sorted_tfidf_index[:20]]
    
    # get the list of 20 features with the  largest coeeficients (): Associated with spam
    largest_coeff = feature_names[sorted_tfidf_index[:-21:-1]]
    
    # create a sorted series of feature names and their coefficients
    smallest_result = pd.Series(values[sorted_tfidf_index[:20]], index= smallest_coeff, name = None)
    smallest_result = smallest_result.iloc[np.lexsort([smallest_result.index, smallest_result.values])]
    
    largest_result = pd.Series(values[sorted_tfidf_index[:-21:-1]], index = largest_coeff, name = None)
    largest_result = largest_result.iloc[np.lexsort([largest_result.index, -largest_result.values])]
    
    return (smallest_result, largest_result) #Your answer here


# In[244]:

answer_four()


# ### Question 5
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
# 
# Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[255]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn import naive_bayes

def answer_five():
    
    # Fit the TfidfVectorizer to the training data ignoring terms that have a document frequency strictly lower than 3
    vect = TfidfVectorizer(min_df = 3).fit(X_train)
    
    # transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train)
    
    # create a classifier object
    clf = naive_bayes.MultinomialNB(alpha=0.1)
    
    # fit the model
    clf.fit(X_train_vectorized, y_train)
    
    # Predict the transformed test documents
    predictions = clf.predict(vect.transform(X_test))

    #find the AUC
    AUC = roc_auc_score(y_test, predictions)
    
    return AUC #Your answer here


# In[256]:

answer_five()


# ### Question 6
# 
# What is the average length of documents (number of characters) for not spam and spam documents?
# 
# *This function should return a tuple (average length not spam, average length spam).*

# In[299]:

def answer_six():
    
    #add a new column with the string lenths of the tokens
    spam_data['char_length'] = spam_data['text'].str.len()
    
    # calculate the means
    mean_NOT_spam = spam_data[spam_data.target == 0].char_length.mean()
    mean_spam = spam_data[spam_data.target == 1].char_length.mean()
    
    return  (mean_NOT_spam, mean_spam)  #Your answer here


# In[300]:

answer_six()


# <br>
# <br>
# The following function has been provided to help you combine new features into the training data:

# In[232]:

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ### Question 7
# 
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.
# 
# Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[328]:

from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

def answer_seven():
    
    # Fit the TfidfVectorizer to the training data ignoring terms that have a document frequency strictly lower than 3
    vect = TfidfVectorizer(min_df = 5).fit(X_train)
    
    # transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)
    
    # calculate string length to the train and test vectors
    X_train_char_len = X_train.str.len()
    X_test_char_len = X_test.str.len()
    
    # add the charachter feature to the transformed documents
    X_train_vectorized = add_feature(X_train_vectorized, X_train_char_len)
    X_test_vectorized = add_feature(X_test_vectorized, X_test_char_len)
    
    # create a classifier object
    clf = SVC(C=10000)
    
    # fit the model
    clf.fit(X_train_vectorized, y_train)
    
    # Predict the transformed test documents
    predictions = clf.predict(X_test_vectorized )

    #find the AUC
    AUC = roc_auc_score(y_test, predictions)
    
    return  AUC  #Your answer here


# In[329]:

answer_seven()


# ### Question 8
# 
# What is the average number of digits per document for not spam and spam documents?
# 
# *This function should return a tuple (average # digits not spam, average # digits spam).*

# In[411]:

def answer_eight():
    
    mean_NOT_spam = spam_data[spam_data.target == 0].text.str.count(r'\d').mean()
    mean_spam = spam_data[spam_data.target == 1].text.str.count(r'\d').mean()
    
    return (mean_NOT_spam, mean_spam) #Your answer here   


# In[412]:

answer_eight()


# ### Question 9
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * **number of digits per document**
# 
# fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[417]:

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

def answer_nine():
    
    # Fit the TfidfVectorizer to the training data ignoring terms that have a document frequency strictly lower than 3
    vect = TfidfVectorizer(min_df = 5, ngram_range = (1, 3)).fit(X_train)
    
    # transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)
    
    # calculate string length to the train and test vectors
    X_train_char_len = X_train.str.len()
    X_test_char_len = X_test.str.len()
    
    # calculate digit length to the train and test vectors
    X_train_dig_len = X_train.str.count(r'\d')
    X_test_dig_len = X_test.str.count(r'\d')
    
    # add the charachter feature to the transformed documents
    X_train_vectorized = add_feature(X_train_vectorized, X_train_char_len)
    X_train_vectorized = add_feature(X_train_vectorized, X_train_dig_len)
    X_test_vectorized = add_feature(X_test_vectorized, X_test_char_len)
    X_test_vectorized = add_feature(X_test_vectorized, X_test_dig_len)
    
    # create a classifier object
    clf = LogisticRegression(C=100)
    
    # fit the model
    clf.fit(X_train_vectorized, y_train)
    
    # Predict the transformed test documents
    predictions = clf.predict(X_test_vectorized)

    #find the AUC
    AUC = roc_auc_score(y_test, predictions)
    
    return AUC #Your answer here


# In[418]:

answer_nine()


# ### Question 10
# 
# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
# 
# *Hint: Use `\w` and `\W` character classes*
# 
# *This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*

# In[423]:

def answer_ten():

    mean_NOT_spam = spam_data[spam_data.target == 0].text.str.count(r'\W').mean()
    mean_spam = spam_data[spam_data.target == 1].text.str.count(r'\W').mean()
    
    return (mean_NOT_spam, mean_spam) #Your answer here   


# In[424]:

answer_ten()


# ### Question 11
# 
# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**
# 
# To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * number of digits per document
# * **number of non-word characters (anything other than a letter, digit or underscore.)**
# 
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
# ['length_of_doc', 'digit_count', 'non_word_char_count']
# 
# *This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*

# In[501]:

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score

def answer_eleven():
    
    #-------------------------------------------------CALCULATE AUC---------------------------------------------------#
    
    # Fit the TfidfVectorizer to the training data ignoring terms that have a document frequency strictly lower than 3
    vect = CountVectorizer(min_df = 5, ngram_range = (2, 5), analyzer='char_wb').fit(X_train)
    
    # transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)
    
    # calculate string length to the train and test vectors
    X_train_char_len = X_train.str.len()
    X_test_char_len = X_test.str.len()
    
    # calculate digit length to the train and test vectors
    X_train_dig_len = X_train.str.count(r'\d')
    X_test_dig_len = X_test.str.count(r'\d')
    
    # calculate non-word characters to the train and test vectors
    X_train_NW_len = X_train.str.count(r'\W')
    X_test_NW_len = X_test.str.count(r'\W')
    
    # add the charachter feature to the transformed documents
    X_train_vectorized = add_feature(X_train_vectorized, X_train_char_len)
    X_train_vectorized = add_feature(X_train_vectorized, X_train_dig_len)
    X_train_vectorized = add_feature(X_train_vectorized, X_train_NW_len)
    X_test_vectorized = add_feature(X_test_vectorized, X_test_char_len)
    X_test_vectorized = add_feature(X_test_vectorized, X_test_dig_len)
    X_test_vectorized = add_feature(X_test_vectorized, X_test_NW_len)
    
    # create a classifier object
    clf = LogisticRegression(C=100)
    
    # fit the model
    clf.fit(X_train_vectorized, y_train)
    
    # Predict the transformed test documents
    predictions = clf.predict(X_test_vectorized)

    #find the AUC
    AUC = roc_auc_score(y_test, predictions)
    
    #-------------------------------------------CALCULATE THE CO-EFFICIENTS-------------------------------------#
    
    #get the sorted score index
    values = clf.coef_[0]
    sorted_coef_index = clf.coef_[0].argsort()
    
    # get the feature names as numpy array
    feature_names = np.array(vect.get_feature_names()).tolist()
    new_features = ['length_of_doc', 'digit_count', 'non_word_char_count']
    feature_names = feature_names + new_features
    feature_names = np.array(feature_names)
    
    # get the list of 10 features with the smallest coeeficients (): Not associated with spam
    smallest_coeff = feature_names[sorted_coef_index[:10]]
    
    # get the list of 10 features with the  largest coeeficients (): Associated with spam
    largest_coeff = feature_names[sorted_coef_index[:-11:-1]]
    
    # create a sorted series of feature names and their coefficients
    smallest_result = pd.Series(values[sorted_coef_index[:10]], index= smallest_coeff, name = None)
    smallest_result = smallest_result.iloc[np.lexsort([smallest_result.index, smallest_result.values])]
    
    largest_result = pd.Series(values[sorted_coef_index[:-11:-1]], index = largest_coeff, name = None)
    largest_result = largest_result.iloc[np.lexsort([largest_result.index, -largest_result.values])]
    
    return (AUC, smallest_result, largest_result)   #Your answer here


# In[502]:

answer_eleven()


# In[ ]:



