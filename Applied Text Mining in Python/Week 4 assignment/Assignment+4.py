
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 4 - Document Similarity & Topic Modelling

# ## Part 1 - Document Similarity
# 
# For the first part of this assignment, you will complete the functions `doc_to_synsets` and `similarity_score` which will be used by `document_path_similarity` to find the path similarity between two documents.
# 
# The following functions are provided:
# * **`convert_tag:`** converts the tag given by `nltk.pos_tag` to a tag used by `wordnet.synsets`. You will need to use this function in `doc_to_synsets`.
# * **`document_path_similarity:`** computes the symmetrical path similarity between two documents by finding the synsets in each document using `doc_to_synsets`, then computing similarities using `similarity_score`.
# 
# You will need to finish writing the following functions:
# * **`doc_to_synsets:`** returns a list of synsets in document. This function should first tokenize and part of speech tag the document using `nltk.word_tokenize` and `nltk.pos_tag`. Then it should find each tokens corresponding synset using `wn.synsets(token, wordnet_tag)`. The first synset match should be used. If there is no match, that token is skipped.
# * **`similarity_score:`** returns the normalized similarity score of a list of synsets (s1) onto a second list of synsets (s2). For each synset in s1, find the synset in s2 with the largest similarity value. Sum all of the largest similarity values together and normalize this value by dividing it by the number of largest similarity values found. Be careful with data types, which should be floats. Missing values should be ignored.
# 
# Once `doc_to_synsets` and `similarity_score` have been completed, submit to the autograder which will run `test_document_path_similarity` to test that these functions are running correctly. 
# 
# *Do not modify the functions `convert_tag`, `document_path_similarity`, and `test_document_path_similarity`.*

# In[15]:

import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd


def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """
    

    # Your Code Here
    
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag, word_tokenize
    
    # convert the document to a list of tokens
    tokens = word_tokenize(doc)
    
    # create a nltk dictionary of tokens and their pos tags
    tag_dict = pos_tag(tokens)
    
    # initialize an empty list of synsets
    synsets = []

    # initialize the lemmatizer before iterating
    # lemmatzr = WordNetLemmatizer()

    # iterate over the token-tag dictionary
    for (token, tag) in tag_dict:

        # get the wordnet tag
        wn_tag = convert_tag(tag)

        # leave word as is if the word net tag was not found
        #if not wn_tag: continue

        # lemmatize each token
        #lemma = lemmatzr.lemmatize(token, pos = wn_tag)
        
        fetched_synsets = wn.synsets(token, pos = wn_tag)

        if len(fetched_synsets) > 0:
            # fetch the synset and append
            synsets.append(fetched_synsets[0]) 
    
    return synsets # Your Answer Here


def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """
    
    
    # Your Code Here
    
    # create a list of similaroty scores for each synset pair
    scores = []
    
    # iterate over each word in s2
    for word_1 in s1:
        
        # create a temporary list to keep scores for each word in s2
        temp = []
        
        # iterate over each word in s1
        for word_2 in s2:
            
            # calcualte the similarity score
            similarity_score = word_1.path_similarity(word_2)
            # add the similarity score if there is one
            if similarity_score != None: temp.append(similarity_score)
                
        scores.append(temp)
            
    # get the biggest scores for each word
    biggest = []
    for scores_ in scores:
        if len(scores_) > 0:
            biggest.append(max(scores_))
    
    
    return sum(biggest) / len(biggest)  # Your Answer Here


def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2


# ### test_document_path_similarity
# 
# Use this function to check if doc_to_synsets and similarity_score are correct.
# 
# *This function should return the similarity score as a float.*

# In[16]:

def test_document_path_similarity():
    doc1 = 'This is a function to test document_path_similarity.'
    doc2 = 'Use this function to see if your code in doc_to_synsets     and similarity_score is correct!'
    return document_path_similarity(doc1, doc2)
test_document_path_similarity()


# <br>
# ___
# `paraphrases` is a DataFrame which contains the following columns: `Quality`, `D1`, and `D2`.
# 
# `Quality` is an indicator variable which indicates if the two documents `D1` and `D2` are paraphrases of one another (1 for paraphrase, 0 for not paraphrase).

# In[17]:

# Use this dataframe for questions most_similar_docs and label_accuracy
paraphrases = pd.read_csv('paraphrases.csv')
paraphrases.head()


# ___
# 
# ### most_similar_docs
# 
# Using `document_path_similarity`, find the pair of documents in paraphrases which has the maximum similarity score.
# 
# *This function should return a tuple `(D1, D2, similarity_score)`*

# In[18]:

def most_similar_docs():
    
    # Your Code Here
    
    best_score = 0
    best_D1 = ""
    best_D2 = ""
    
    # iterate over each row
    for index, row in paraphrases.iterrows():
        
        #extract documents
        current_D1 = row['D1']
        current_D2 = row['D2']
        
        # calculat the document similarity score
        current_score = document_path_similarity(current_D1, current_D2)
        
        # keep track of highest score and best doc combination
        if current_score > best_score:
            best_score = current_score
            best_D1 = current_D1 
            best_D2 = current_D2
    
    return (best_D1, best_D2, best_score) # Your Answer Here


# In[19]:

most_similar_docs()


# ### label_accuracy
# 
# Provide labels for the twenty pairs of documents by computing the similarity for each pair using `document_path_similarity`. Let the classifier rule be that if the score is greater than 0.75, label is paraphrase (1), else label is not paraphrase (0). Report accuracy of the classifier using scikit-learn's accuracy_score.
# 
# *This function should return a float.*

# In[20]:

def label_accuracy():
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_validation import train_test_split

    # Your Code Here
    
    scores = []
    
    # iterate over each row
    for index, row in paraphrases.iterrows():
        
        #extract documents
        current_D1 = row['D1']
        current_D2 = row['D2']
        
        # calculat the document similarity score
        current_score = document_path_similarity(current_D1, current_D2)
        
        # add to the list of scores that will be added to the dataframe
        scores.append(current_score)
    
    # create a new dataframe for the classifier
    df = paraphrases
    df['similarity_score'] = scores
    
    # naive classifier based on similarity scores
    df['paraphrase'] = [1 if x > 0.75 else 0 for x in df['similarity_score']]
    
    # extract the true classsification and naive prediction
    y_true = df.Quality
    y_pred = df.paraphrase
    
    # fetch the accuracy score
    accuracy_score = accuracy_score(y_true, y_pred)
        
    return accuracy_score # Your Answer Here

label_accuracy()


# ## Part 2 - Topic Modelling
# 
# For the second part of this assignment, you will use Gensim's LDA (Latent Dirichlet Allocation) model to model topics in `newsgroup_data`. You will first need to finish the code in the cell below by using gensim.models.ldamodel.LdaModel constructor to estimate LDA model parameters on the corpus, and save to the variable `ldamodel`. Extract 10 topics using `corpus` and `id_map`, and with `passes=25` and `random_state=34`.

# In[21]:

import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer

# Load the list of documents
with open('newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)

# Use CountVectorizor to find three letter tokens, remove stop_words, 
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')
# Fit and transform
X = vect.fit_transform(newsgroup_data)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())


# ### lda_topics
# 
# Using `ldamodel`, find a list of the 10 topics and the most significant 10 words in each topic. This should be structured as a list of 10 tuples where each tuple takes on the form:
# 
# `(9, '0.068*"space" + 0.036*"nasa" + 0.021*"science" + 0.020*"edu" + 0.019*"data" + 0.017*"shuttle" + 0.015*"launch" + 0.015*"available" + 0.014*"center" + 0.014*"sci"')`
# 
# for example.
# 
# *This function should return a list of tuples.*

# In[10]:

# Use the gensim.models.ldamodel.LdaModel constructor to estimate 
# LDA model parameters on the corpus, and save to the variable `ldamodel`

# Your code here:
ldamodel = gensim.models.ldamodel.LdaModel


# In[17]:

def lda_topics():
    
    # Your Code Here
    model = ldamodel(corpus=corpus, id2word=id_map, num_topics = 10, passes=25, random_state=34)
    
    #  10 topics and the most significant 10 words in each topic
    topics = model.print_topics(num_words=10)
    
    return topics # Your Answer Here
lda_topics()


# ### topic_distribution
# 
# For the new document `new_doc`, find the topic distribution. Remember to use vect.transform on the the new doc, and Sparse2Corpus to convert the sparse matrix to gensim corpus.
# 
# *This function should return a list of tuples, where each tuple is `(#topic, probability)`*

# In[22]:

new_doc = ["\n\nIt's my understanding that the freezing will start to occur because of the\ngrowing distance of Pluto and Charon from the Sun, due to it's\nelliptical orbit. It is not due to shadowing effects. \n\n\nPluto can shadow Charon, and vice-versa.\n\nGeorge Krumins\n-- "]


# In[23]:

def topic_distribution():
    
    # Your Code Here 
    
    # Fit and transform
    X_2 = vect.transform(new_doc)

    # Convert sparse matrix to gensim corpus.
    corpus_2 = gensim.matutils.Sparse2Corpus(X_2, documents_columns=False)
    
    # initialise lda model
    ldamodel = gensim.models.ldamodel.LdaModel
    lda = ldamodel(corpus=corpus, id2word=id_map, num_topics = 10,random_state=34)
    
    # get the distribution of topics and probabilities NB(might be worth including passes=25 if results are incorrect)
    result = lda.get_document_topics(corpus_2, minimum_probability=0.01)

    return list(result)[0]

topic_distribution()


# ### topic_names
# 
# From the list of the following given topics, assign topic names to the topics you found. If none of these names best matches the topics you found, create a new 1-3 word "title" for the topic.
# 
# Topics: Health, Science, Automobiles, Politics, Government, Travel, Computers & IT, Sports, Business, Society & Lifestyle, Religion, Education.
# 
# *This function should return a list of 10 strings.*

# In[ ]:

def topic_names():
    
    # Your Code Here
    
    return ['Computers & IT', 'Automobiles', 'Computers & IT', 'Religion', 'Automobiles', 'Sports',
             'Education', 'Religion', 'Computers & IT', 'Science']# Your Answer Here


# In[ ]:

topic_names()

