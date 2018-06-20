
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-social-network-analysis/resources/yPcBs) course resource._
# 
# ---

# # Assignment 1 - Creating and Manipulating Graphs
# 
# Eight employees at a small company were asked to choose 3 movies that they would most enjoy watching for the upcoming company movie night. These choices are stored in the file `Employee_Movie_Choices.txt`.
# 
# A second file, `Employee_Relationships.txt`, has data on the relationships between different coworkers. 
# 
# The relationship score has value of `-100` (Enemies) to `+100` (Best Friends). A value of zero means the two employees haven't interacted or are indifferent.
# 
# Both files are tab delimited.

# In[4]:

import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms import bipartite


# This is the set of employees
employees = set(['Pablo',
                 'Lee',
                 'Georgia',
                 'Vincent',
                 'Andy',
                 'Frida',
                 'Joan',
                 'Claude'])

# This is the set of movies
movies = set(['The Shawshank Redemption',
              'Forrest Gump',
              'The Matrix',
              'Anaconda',
              'The Social Network',
              'The Godfather',
              'Monty Python and the Holy Grail',
              'Snakes on a Plane',
              'Kung Fu Panda',
              'The Dark Knight',
              'Mean Girls'])


# you can use the following function to plot graphs
# make sure to comment it out before submitting to the autograder

def plot_graph(G, weight_name=None):
    '''
    G: a networkx G
    weight_name: name of the attribute for plotting edge weights (if G is weighted)
    '''
    #%matplotlib notebook
    #import matplotlib.pyplot as plt
    
    plt.figure()
    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = None
    
    if weight_name:
        weights = [int(G[u][v][weight_name]) for u,v in edges]
        labels = nx.get_edge_attributes(G,weight_name)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        nx.draw_networkx(G, pos, edges=edges, width=weights);
    else:
        nx.draw_networkx(G, pos, edges=edges);


# ### Question 1
# 
# Using NetworkX, load in the bipartite graph from `Employee_Movie_Choices.txt` and return that graph.
# 
# *This function should return a networkx graph with 19 nodes and 24 edges*

# In[5]:

from networkx.algorithms import bipartite

def answer_one():
        
    # Your Code Here
    
    # load the text file into a dataframe
    df = pd.read_csv('Employee_Movie_Choices.txt', sep='\t')
    
    # initialise the graph
    G=nx.Graph()
    
    # add in the nodes for both parts of the bipartitie graph
    G.add_nodes_from(df['#Employee'].unique(), bipartite=0)
    G.add_nodes_from(df['Movie'].unique(), bipartite=1)
    
    # add the edges between nodes
    edges = list(zip(df['#Employee'], df['Movie']))
    G.add_edges_from(edges)
    
   
    #----------------------------PLOT BIPARTITE GRAPH-----------------------------------------------#
    '''
    # Separate by group
    l, r = nx.bipartite.sets(G)
    pos = {}

    # Update position for node from each group
    pos.update((node, (1, index)) for index, node in enumerate(l))
    pos.update((node, (2, index)) for index, node in enumerate(r))

    nx.draw(G, pos=pos)
    plt.show()
    '''

    return G # Your Answer Here


# In[6]:

def test_answer_one():
    
    G = answer_one()
    
    # check if graph is bipartite
    assert bipartite.is_bipartite(G) == True
    
    # check the number of nodes
    assert len(G.nodes(data = True)) == 19
    
    # check the number of edges
    assert len(G.edges(data = True)) == 24
    
    print("All test cases pass!")
    
test_answer_one()


# ### Question 2
# 
# Using the graph from the previous question, add nodes attributes named `'type'` where movies have the value `'movie'` and employees have the value `'employee'` and return that graph.
# 
# *This function should return a networkx graph with node attributes `{'type': 'movie'}` or `{'type': 'employee'}`*

# In[7]:

def answer_two():
    
    # Your Code Here
    
    # load the graph
    G = answer_one()
    
    # load the text file into a dataframe
    df = pd.read_csv('Employee_Movie_Choices.txt', sep='\t')
    
    # iterate over all rows in the data frame and add attributes to graph
    for idx, row in df.iterrows():
        G.node[row['#Employee']]['type'] = 'employee'
        G.node[row['Movie']]['type'] = 'movie'
    
    return G # Your Answer Here

answer_two()


# ### Question 3
# 
# Find a weighted projection of the graph from `answer_two` which tells us how many movies different pairs of employees have in common.
# 
# *This function should return a weighted projected graph.*

# In[8]:

def answer_three():
        
    # Your Code Here
    
    # load graph
    G = answer_two()
    
    # load the text file into a dataframe
    df = pd.read_csv('Employee_Movie_Choices.txt', sep='\t')
    
    # get set of movies
    X = set(df['#Employee'].unique())
    
    P = bipartite.weighted_projected_graph(G, X)
    
    return P # Your Answer Here

answer_three()


# ### Question 4
# 
# Suppose you'd like to find out if people that have a high relationship score also like the same types of movies.
# 
# Find the Pearson correlation ( using `DataFrame.corr()` ) between employee relationship scores and the number of movies they have in common. If two employees have no movies in common it should be treated as a 0, not a missing value, and should be included in the correlation calculation.
# 
# *This function should return a float.*

# In[44]:

def answer_four():
        
    # Your Code Here
    
    # load graph
    G = answer_three()
    
    # extract employee edge weights from graph into dataframe
    employee_weights = nx.to_pandas_dataframe(G)
    
    # load the text file into a dataframe
    df = pd.read_csv('Employee_Relationships.txt', sep='\t', header = None)
    
    # add column names to the dataframe
    df.columns = ['Employee_1', 'Employee_2','Frienship_score']
    
    # add dummy column for movie weight for movies in common
    df['Movie_score'] = 0
    
    # add employee weights to the friendship score dataframe
    df['Movie_score'] =  employee_weights.lookup(df['Employee_1'], df['Employee_2'])
    
    # calcualte the Pearson correlation
    Pearson_corr = df['Frienship_score'].corr(df['Movie_score'])
            
    return Pearson_corr  # Your Answer Here

answer_four()

