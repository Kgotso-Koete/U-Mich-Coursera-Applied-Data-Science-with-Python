
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 2 - Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

# In[1]:

import nltk
import pandas as pd
import numpy as np


# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*

# In[2]:

def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*

# In[3]:

def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*

# In[4]:

from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*

# In[5]:

def answer_one():
    
    # total number of words
    total = len(moby_tokens)
    
    # number of unique words
    unique = len(set(moby_tokens))
    
    return unique / total # Your answer here

answer_one()


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*

# In[6]:

def answer_two():
    
    # do a count for lower and upper case
    count = moby_tokens.count('whale') + moby_tokens.count('Whale')
    
    # total number of words
    total = len(moby_tokens)
    
    return (count / total) * 100 # Your answer here

answer_two()


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*

# In[7]:

from collections import Counter

def answer_three():
    
    # count frequency of all words
    data = Counter(moby_tokens)
    
    # sort the list
    l_sorted = data.most_common()[0:20]
    
    return l_sorted # Your answer here

answer_three()


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return a sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*

# In[8]:

from nltk.probability import FreqDist

def answer_four():
    
    #create  distribution of word frequencies
    dist = FreqDist(text1)
    
    # vocabulary list
    vocab = dist.keys()
    
    # create a list of frequent words per condidion
    freqwords = [w for w in vocab if len(w) > 5 and dist[w] > 150]
    
    return sorted(freqwords) # Your answer here

answer_four()


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*

# In[9]:

def answer_five():
    
    # longest word
    word = max(moby_tokens, key=len)
    
    # length of longest word
    L = len(word)
    
    return (word, L)# Your answer here

answer_five()


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*

# In[10]:

from nltk.probability import FreqDist

def answer_six():
    
    #create  distribution of word frequencies
    dist = FreqDist(text1)
    
    # vocabulary list
    vocab = dist.keys()
    
    # find the frequency per the conditions
    freqwords = [(dist[w], w)  for w in vocab if w.isalpha() and dist[w] > 2000]
    
    # sort the list by distribution number
    results = sorted(freqwords , key=lambda x: x[0], reverse = True)
    
    return results # Your answer here

answer_six()


# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*

# In[11]:

def answer_seven():
    
    # split entire text into sentences in a list
    sentences = nltk.sent_tokenize(moby_raw)
    
    # total number of sentences
    num_sentences = len(sentences)
    
    # total number of words
    total_tokens = len(moby_tokens)

    return total_tokens / num_sentences  # Your answer here

answer_seven()


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*

# In[12]:

from collections import Counter

def answer_eight():
    
    # tag each token as a part of speach
    pos = nltk.pos_tag(moby_tokens)
    
    # list only the parts of speech for frequency analysis
    list_of_pos = [x[1] for x in pos]
    
    # count frequency of all pos
    data = Counter(list_of_pos)
    
    # sort the list
    l_sorted = data.most_common()[0:5]
    
    return l_sorted # Your answer here

answer_eight()


# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.

# In[34]:

from nltk.corpus import words

correct_spellings = words.words()


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[75]:

def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    
    #specify num of ngrams
    n = 3
    
    results = []

    for entry in entries:

        # highest coeff
        Highest_Coeff = 0
        best_word = ""

        # loop over each word in the dictionary
        for word in correct_spellings:
            
            # if first letters match, then do the following
            if entry[0].lower() == word[0].lower():
                
                # calculate the Jaccarrd coefficient
                Jaccard_Coeff = 1 - nltk.jaccard_distance(set(nltk.ngrams(entry, n)), set(nltk.ngrams(word, n)))
                
                # keep track of highest coefficient and best word
                if Jaccard_Coeff > Highest_Coeff:
                    Highest_Coeff = Jaccard_Coeff
                    best_word = word
               
        # append the best results for each word entry pair to the final list
        results.append((entry, best_word, Jaccard_Coeff))
    
    # store the best results in a tuple and return suggested words
    final = [results[0][1], results[1][1], results[2][1]]

    return final # Your answer here
    
answer_nine()


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[76]:

def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    
    #specify num of ngrams
    n = 4
    
    results = []

    for entry in entries:

        # highest coeff
        Highest_Coeff = 0
        best_word = ""

        # loop over each word in the dictionary
        for word in correct_spellings:
            
            # if first letters match, then do the following
            if entry[0].lower() == word[0].lower():
                
                # calculate the Jaccarrd coefficient
                Jaccard_Coeff = 1 - nltk.jaccard_distance(set(nltk.ngrams(entry, n)), set(nltk.ngrams(word, n)))
                
                # keep track of highest coefficient and best word
                if Jaccard_Coeff > Highest_Coeff:
                    Highest_Coeff = Jaccard_Coeff
                    best_word = word
               
        # append the best results for each word entry pair to the final list
        results.append((entry, best_word, Jaccard_Coeff))
    
    # store the best results in a tuple and return suggested words
    final = [results[0][1], results[1][1], results[2][1]]

    return final # Your answer here
    
answer_ten()


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[92]:

def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    
    results = []

    for entry in entries:

        # highest coeff
        Lowest_Coeff = float("inf")
        best_word = ""

        # loop over each word in the dictionary
        for word in correct_spellings:
            
            # if first letters match, then do the following
            if entry[0].lower() == word[0].lower():
                
                # calculate the Jaccarrd coefficient
                Levenshtein_dist = nltk.edit_distance(entry, word, transpositions=True)
                
                ## keep track of highest coefficient and best word
                if Levenshtein_dist < Lowest_Coeff:
                    Lowest_Coeff = Levenshtein_dist
                    best_word = word
               
        # append the best results for each word entry pair to the final list
        results.append((entry, best_word, Levenshtein_dist))
    
    # store the best results in a tuple and return suggested words
    final = [results[0][1], results[1][1], results[2][1]]

    return final # Your answer here
    
answer_eleven()


# In[ ]:



