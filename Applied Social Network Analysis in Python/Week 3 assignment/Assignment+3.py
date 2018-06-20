
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.2** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-social-network-analysis/resources/yPcBs) course resource._
# 
# ---

# # Assignment 3
# 
# In this assignment you will explore measures of centrality on two networks, a friendship network in Part 1, and a blog network in Part 2.

# ## Part 1
# 
# Answer questions 1-4 using the network `G1`, a network of friendships at a university department. Each node corresponds to a person, and an edge indicates friendship. 
# 
# *The network has been loaded as networkx graph object `G1`.*

# In[12]:

import networkx as nx
import numpy as np
import pandas as pd

G1 = nx.read_gml('friendships.gml')


# In[13]:

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
        nx.draw_networkx(G, pos, edges=edges, width=weights, arrows = True);
    else:
        nx.draw_networkx(G, pos, edges=edges, arrows = True);


# In[14]:

def test_graph():
    
    # initialize graph
    G = nx.Graph()  
    
    # add edges and nodes
    G.add_edges_from([('B','A'),('A','C'),('B','D'),('D','G'),('G','F'), ('C','E'),('C','D'),('E','D'),('D','G'),('G','E')]) 
    
    # plot the graph
    #plot_graph(G, weight_name=None)
    
    return  nx.degree_centrality(G)


# ### Question 1
# 
# Find the degree centrality, closeness centrality, and normalized betweeness centrality (excluding endpoints) of node 100.
# 
# *This function should return a tuple of floats `(degree_centrality, closeness_centrality, betweenness_centrality)`.*

# In[15]:

def answer_one():
        
    # Your Code Here
    
    # get node 100
    node_100 = G1.nodes(data = True)[101][0] 
    
    # get the degree centrality
    degree_centrality = nx.degree_centrality(G1)[node_100]
    
    # get the closeness_centrality
    closeness_centrality = nx.closeness_centrality(G1)[node_100]
    
    # get the degree centrality
    betweenness_centrality = nx.betweenness_centrality(G1)[node_100]
    
    return (degree_centrality, closeness_centrality, betweenness_centrality)  # Your Answer Here


# In[16]:

answer_one()


# <br>
# #### For Questions 2, 3, and 4, assume that you do not know anything about the structure of the network, except for the all the centrality values of the nodes. That is, use one of the covered centrality measures to rank the nodes and find the most appropriate candidate.
# <br>

# ### Question 2
# 
# Suppose you are employed by an online shopping website and are tasked with selecting one user in network G1 to send an online shopping voucher to. We expect that the user who receives the voucher will send it to their friends in the network.  You want the voucher to reach as many nodes as possible. The voucher can be forwarded to multiple users at the same time, but the travel distance of the voucher is limited to one step, which means if the voucher travels more than one step in this network, it is no longer valid. Apply your knowledge in network centrality to select the best candidate for the voucher. 
# 
# *This function should return an integer, the name of the node.*

# In[26]:

def answer_two():
        
    # Your Code Here
   
    # get the degree centrality
    degree_centrality = nx.degree_centrality(G1)
    
    # sort the result and put the node with highest score at the top
    results_sorted = sorted(degree_centrality.items(), key=lambda x: x[1], reverse = True)
    
    # extract the node with the best score
    best_node = results_sorted[0][0]

    return best_node # Your Answer Here

answer_two()


# ### Question 3
# 
# Now the limit of the voucher’s travel distance has been removed. Because the network is connected, regardless of who you pick, every node in the network will eventually receive the voucher. However, we now want to ensure that the voucher reaches the nodes in the lowest average number of hops.
# 
# How would you change your selection strategy? Write a function to tell us who is the best candidate in the network under this condition.
# 
# *This function should return an integer, the name of the node.*

# In[27]:

def answer_three():
        
    # Your Code Here
    
    # get the closeness_centrality
    closeness_centrality = nx.closeness_centrality(G1)
    
    # sort the result and put the node with highest score at the top
    results_sorted = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse = True)
    
    # extract the node with the best score
    best_node = results_sorted[0][0]

    return best_node # Your Answer Here

answer_three()


# ### Question 4
# 
# Assume the restriction on the voucher’s travel distance is still removed, but now a competitor has developed a strategy to remove a person from the network in order to disrupt the distribution of your company’s voucher. Your competitor is specifically targeting people who are often bridges of information flow between other pairs of people. Identify the single riskiest person to be removed under your competitor’s strategy?
# 
# *This function should return an integer, the name of the node.*

# In[28]:

def answer_four():
        
    # Your Code Here
    
    # get the betweenness_centrality
    betweenness_centrality = nx.betweenness_centrality(G1)
    
    # sort the result and put the node with highest score at the top
    results_sorted = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse = True)
    
    # extract the node with the best score
    best_node = results_sorted[0][0]

    return best_node # Your Answer Here

answer_four()


# ## Part 2
# 
# `G2` is a directed network of political blogs, where nodes correspond to a blog and edges correspond to links between blogs. Use your knowledge of PageRank and HITS to answer Questions 5-9.

# In[29]:

G2 = nx.read_gml('blogs.gml')


# ### Question 5
# 
# Apply the Scaled Page Rank Algorithm to this network. Find the Page Rank of node 'realclearpolitics.com' with damping value 0.85.
# 
# *This function should return a float.*

# In[34]:

def answer_five():
        
    # Your Code Here
    
    # get all page ranks 
    ranks = nx.pagerank(G2, alpha=0.85) 
    
    return ranks['realclearpolitics.com'] # Your Answer Here

answer_five()


# ### Question 6
# 
# Apply the Scaled Page Rank Algorithm to this network with damping value 0.85. Find the 5 nodes with highest Page Rank. 
# 
# *This function should return a list of the top 5 blogs in desending order of Page Rank.*

# In[42]:

def answer_six():
        
    # Your Code Here
    
    # get all page ranks 
    ranks = nx.pagerank(G2, alpha=0.85) 
    
    # sort the result and put the node with highest score at the top
    results_sorted = sorted(ranks.items(), key=lambda x: x[1], reverse = True)
    
    # extract the node with the best score
    best_pages = [x[0] for x in results_sorted[:5]]
    
    return best_pages # Your Answer Here

answer_six()


# ### Question 7
# 
# Apply the HITS Algorithm to the network to find the hub and authority scores of node 'realclearpolitics.com'. 
# 
# *Your result should return a tuple of floats `(hub_score, authority_score)`.*

# In[45]:

def answer_seven():
        
    # Your Code Here
    
    # get all the hub scores and authority scores
    HITS_scores = nx.hits(G2)
    
    # get all the hub scores
    hub_score_list = nx.hits(G2)[0]
    
    # get all the authority scores
    authority_score_list = nx.hits(G2)[1]
    
    return (hub_score_list['realclearpolitics.com'], authority_score_list['realclearpolitics.com']) # Your Answer Here

answer_seven()


# ### Question 8 
# 
# Apply the HITS Algorithm to this network to find the 5 nodes with highest hub scores.
# 
# *This function should return a list of the top 5 blogs in desending order of hub scores.*

# In[48]:

def answer_eight():
        
    # Your Code Here
    
    # get all the hub scores and authority scores
    HITS_scores = nx.hits(G2)
    
    # get all the hub scores
    hub_score_list = nx.hits(G2)[0]
    
    # sort the result and put the node with highest score at the top
    results_sorted = sorted(hub_score_list.items(), key=lambda x: x[1], reverse = True)
    
    # extract the node with the best score
    best_pages = [x[0] for x in results_sorted[:5]]
    
    return best_pages # Your Answer Here

answer_eight()


# ### Question 9 
# 
# Apply the HITS Algorithm to this network to find the 5 nodes with highest authority scores.
# 
# *This function should return a list of the top 5 blogs in desending order of authority scores.*

# In[49]:

def answer_nine():
        
    # Your Code Here
    
    # get all the hub scores and authority scores
    HITS_scores = nx.hits(G2)
    
    # get all the authority scores
    authority_score_list = nx.hits(G2)[1]
    
    # sort the result and put the node with highest score at the top
    results_sorted = sorted(authority_score_list.items(), key=lambda x: x[1], reverse = True)
    
    # extract the node with the best score
    best_pages = [x[0] for x in results_sorted[:5]]
    
    return best_pages # Your Answer Here

answer_nine()

