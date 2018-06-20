
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.2** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# # Assignment 3 - Evaluation
# 
# In this assignment you will train several models and evaluate how effectively they predict instances of fraud using data based on [this dataset from Kaggle](https://www.kaggle.com/dalpozz/creditcardfraud).
#  
# Each row in `fraud_data.csv` corresponds to a credit card transaction. Features include confidential variables `V1` through `V28` as well as `Amount` which is the amount of the transaction. 
#  
# The target is stored in the `class` column, where a value of 1 corresponds to an instance of fraud and 0 corresponds to an instance of not fraud.

# In[1]:

import numpy as np
import pandas as pd


# ### Question 1
# Import the data from `fraud_data.csv`. What percentage of the observations in the dataset are instances of fraud?
# 
# *This function should return a float between 0 and 1.* 

# In[2]:

def answer_one():
    # Your code here
    
    #load data set
    df = pd.read_csv('fraud_data.csv')
    
    # total size
    total = df.shape[0]
    
    # count of fraudulent transactions (marked with 1 in Class)
    fraud_counts = df[df['Class'] == 1].shape[0]
    
    return fraud_counts / total # Return your answer 
    
answer_one()


# In[3]:

# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# ### Question 2
# 
# Using `X_train`, `X_test`, `y_train`, and `y_test` (as defined above), train a dummy classifier that classifies everything as the majority class of the training data. What is the accuracy of this classifier? What is the recall?
# 
# *This function should a return a tuple with two floats, i.e. `(accuracy score, recall score)`.*

# In[4]:

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    
    # Your code here
    
    # create Dummy Classifier object
    dummy_classifier = DummyClassifier(strategy="most_frequent")
    
    # train a dummy classifier to make predictions based on the most_frequent class value
    dummy_classifier.fit( X_train, y_train )
    
    # make a prediction on the y values
    y_pred = dummy_classifier.predict(X_test)
    
    # calculate the recall
    recall = recall_score(y_test, y_pred)
    
    # calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return (accuracy, recall) # Return your answer

answer_two()


# ### Question 3
# 
# Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer using the default parameters. What is the accuracy, recall, and precision of this classifier?
# 
# *This function should a return a tuple with three floats, i.e. `(accuracy score, recall score, precision score)`.*

# In[4]:

def answer_three():
    from sklearn.metrics import recall_score, precision_score, accuracy_score
    from sklearn.svm import SVC

    # Your code here
    
    # create a classifier object
    clf = SVC()
    
    # train a SVC classifier to make predictions 
    clf.fit( X_train, y_train )
    
    # make a prediction on the y values
    y_pred = clf.predict(X_test)
    
    # calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # calculate the recall
    recall = recall_score(y_test, y_pred)
    
    # calculate the accuracy
    precision = precision_score(y_test, y_pred)
    
    return (accuracy, recall, precision) # Return your answer

answer_three()


# ### Question 4
# 
# Using the SVC classifier with parameters `{'C': 1e9, 'gamma': 1e-07}`, what is the confusion matrix when using a threshold of -220 on the decision function. Use X_test and y_test.
# 
# *This function should return a confusion matrix, a 2x2 numpy array with 4 integers.*

# In[15]:

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    # Your code here
    
    # create a classifier object
    clf = SVC(C = 1e9, gamma = 1e-07)
    
    # train a SVC classifier to make predictions 
    clf.fit( X_train, y_train )
    
    # make a prediction on the y values
    y_pred = clf.decision_function(X_test)
    
    # calculate the accuracy
    conf_matrix = confusion_matrix(y_test, y_pred > -220)
    
    return conf_matrix # Return your answer

answer_four()


# ### Question 5
# 
# Train a logisitic regression classifier with default parameters using X_train and y_train.
# 
# For the logisitic regression classifier, create a precision recall curve and a roc curve using y_test and the probability estimates for X_test (probability it is fraud).
# 
# Looking at the precision recall curve, what is the recall when the precision is `0.75`?
# 
# Looking at the roc curve, what is the true positive rate when the false positive rate is `0.16`?
# 
# *This function should return a tuple with two floats, i.e. `(recall, true positive rate)`.*

# In[29]:

def answer_five():    
    # Your code here
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import recall_score, precision_score, accuracy_score
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc
    #import matplotlib.pyplot as plt
    
    # create a classifier object
    LogReg = LogisticRegression()
    
    # train a logistic regression classifier to make predictions 
    LogReg.fit(X_train, y_train)
    
    # make a prediction on the y values
    y_pred = LogReg.predict(X_test)
    
    # calculate the recall
    recall = recall_score(y_test, y_pred)
    
    # calculate the accuracy
    precision = precision_score(y_test, y_pred)
    
    # plot the recall curve
    '''
    average_precision = average_precision_score(y_test, y_pred)
    precision_, recall_, _ = precision_recall_curve(y_test, y_pred)
    plt.step(recall_, precision_, color='b', alpha=0.2, where='post')
    plt.fill_between(recall_, precision_, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()
    
    
    # plot the ROC curve
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    '''
    
    
    return (recall, precision) # Return your answer

answer_five()


# ### Question 6
# 
# Perform a grid search over the parameters listed below for a Logisitic Regression classifier, using recall for scoring and the default 3-fold cross validation.
# 
# `'penalty': ['l1', 'l2']`
# 
# `'C':[0.01, 0.1, 1, 10, 100]`
# 
# From `.cv_results_`, create an array of the mean test scores of each parameter combination. i.e.
# 
# |      	| `l1` 	| `l2` 	|
# |:----:	|----	|----	|
# | **`0.01`** 	|    ?	|   ? 	|
# | **`0.1`**  	|    ?	|   ? 	|
# | **`1`**    	|    ?	|   ? 	|
# | **`10`**   	|    ?	|   ? 	|
# | **`100`**   	|    ?	|   ? 	|
# 
# <br>
# 
# *This function should return a 5 by 2 numpy array with 10 floats.* 
# 
# *Note: do not return a DataFrame, just the values denoted by '?' above in a numpy array. You might need to reshape your raw result to meet the format we are looking for.*

# In[38]:

def answer_six():    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    # Your code here
    
    # specify parameters
    tuned_parameters = [{'penalty': ['l1', 'l2'],'C':  [0.01, 0.1, 1, 10, 100]}]
    
    # create a classifier object
    LogReg = LogisticRegression()
    
    # create a grid seach object
    clf = GridSearchCV(LogReg, param_grid = tuned_parameters, scoring = 'recall')
    
    # fit the model
    clf.fit(X_train, y_train)
    
    # predict the y labels
    y_pred = clf.predict(X_test)
    
    # store the results in a 5 X 2 matrix
    score_results = (clf.cv_results_['mean_test_score'].reshape(5,2))
    
    return score_results # Return your answer

answer_six()


# In[11]:

# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    # %matplotlib notebook
    import seaborn as sns
    # import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0);

#GridSearch_Heatmap(answer_six())

