
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.2** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-social-network-analysis/resources/yPcBs) course resource._
# 
# ---

# # Assignment 2 - Network Connectivity
# 
# In this assignment you will go through the process of importing and analyzing an internal email communication network between employees of a mid-sized manufacturing company. 
# Each node represents an employee and each directed edge between two nodes represents an individual email. The left node represents the sender and the right node represents the recipient.

# In[1]:

import networkx as nx

# This line must be commented out when submitting to the autograder
#!head email_network.txt

# load the text file into a dataframe
# df = pd.read_csv('email_network.txt', sep='\t')


# In[2]:

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
# Using networkx, load up the directed multigraph from `email_network.txt`. Make sure the node names are strings.
# 
# *This function should return a directed multigraph networkx graph.*

# In[3]:

import pandas as pd

def answer_one():
    
    # Your Code Here
    
    # initialize graph
    G = nx.read_edgelist(path="email_network.txt", 
                         create_using=nx.MultiDiGraph(),
                         delimiter='\t', 
                         nodetype=str, 
                         data=[('time', int)])

    return G # Your Answer Here

answer_one()


# ### Question 2
# 
# How many employees and emails are represented in the graph from Question 1?
# 
# *This function should return a tuple (#employees, #emails).*

# In[4]:

def answer_two():
        
    # Your Code Here
    
    # initialise the graph
    G = answer_one()
    
    # count the number of edges
    edge_count = len(G.edges(data = True))
    
    # count the number of nodes
    node_count = len(G.nodes(data = True))
    
    return (node_count , edge_count) # Your Answer Here

answer_two()


# ### Question 3
# 
# * Part 1. Assume that information in this company can only be exchanged through email.
# 
#     When an employee sends an email to another employee, a communication channel has been created, allowing the sender to provide information to the receiver, but not vice versa. 
# 
#     Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?
# 
# 
# * Part 2. Now assume that a communication channel established by an email allows information to be exchanged both ways. 
# 
#     Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?
# 
# 
# *This function should return a tuple of bools (part1, part2).*

# In[5]:

def answer_three():
        
    # Your Code Here
    
    # initialise the graph
    G = answer_one()
    
    # check if the graph is strongly connected to answer part 1
    strongly_connected_status = nx.is_strongly_connected(G)
    
    # check if the graph is weakly connected to answer part 2
    weakly_connected_status = nx.is_weakly_connected(G)
    
    return (strongly_connected_status, weakly_connected_status )  # Your Answer Here

answer_three()


# ### Question 4
# 
# How many nodes are in the largest (in terms of nodes) weakly connected component?
# 
# *This function should return an int.*

# In[6]:

def answer_four():
        
    # Your Code Here
    
    # initialise the graph
    G = answer_one()
    
    # create list of weakly connected components
    components = nx.weakly_connected_components(G)
    
    # sort the list in descending order and find the length
    results = len(sorted(components, reverse = True)[0])
    
    return results # Your Answer Here

answer_four()


# ### Question 5
# 
# How many nodes are in the largest (in terms of nodes) strongly connected component?
# 
# *This function should return an int*

# In[7]:

def answer_five():
        
    # Your Code Here
    
    # initialise the graph
    G = answer_one()
    
    # create list of weakly connected components
    components = nx.strongly_connected_components(G)
    
    # sort the list in descending order and find the length
    components_sorted = sorted(components, key=lambda x: len(x), reverse=True) 
    
    # get the longest list of nodes and find length
    results = len(components_sorted[0])
    
    return results  # Your Answer Here
    
answer_five()


# ### Question 6
# 
# Using the NetworkX function strongly_connected_component_subgraphs, find the subgraph of nodes in a largest strongly connected component. 
# Call this graph G_sc.
# 
# *This function should return a networkx MultiDiGraph named G_sc.*

# In[8]:

def answer_six():
        
    # Your Code Here
    
    # initialise the graph
    G = answer_one()
    
    # largest subgraph of nodes in a largest strongly connected component
    G_sc = max(nx.strongly_connected_component_subgraphs(G), key=len)
    
    return G_sc # Your Answer Here

answer_six()


# ### Question 7
# 
# What is the average distance between nodes in G_sc?
# 
# *This function should return a float.*

# In[9]:

def answer_seven():
        
    # Your Code Here
    
    # initialise the graph
    G = answer_six()
    
    # get average distance
    avg_dist = nx.average_shortest_path_length(G)
    
    return avg_dist # Your Answer Here

answer_seven()


# ### Question 8
# 
# What is the largest possible distance between two employees in G_sc?
# 
# *This function should return an int.*

# In[10]:

def answer_eight():
        
    # Your Code Here
    
    # initialise the graph
    G = answer_six()
    
    # find the diameter
    diamater = nx.diameter(G)
    
    return diamater  # Your Answer Here

answer_eight()


# ### Question 9
# 
# What is the set of nodes in G_sc with eccentricity equal to the diameter?
# 
# *This function should return a set of the node(s).*

# In[11]:

def answer_nine():
       
    # Your Code Here
    
    # initialise the graph
    G = answer_six()
    
    # get the nodes whos eccentricity is equal to the graph diameter
    nodes = set(nx.periphery(G))
    
    return nodes # Your Answer Here

answer_nine()


# ### Question 10
# 
# What is the set of node(s) in G_sc with eccentricity equal to the radius?
# 
# *This function should return a set of the node(s).*

# In[12]:

def answer_ten():
        
    # Your Code Here
    
    # initialise the graph
    G = answer_six()
    
    # get the nodes whos eccentricity is equal to the graph diameter
    nodes = set(nx.center(G))
    
    return nodes # Your Answer Here
    
answer_ten()


# ### Question 11
# 
# Which node in G_sc is connected to the most other nodes by a shortest path of length equal to the diameter of G_sc?
# 
# How many nodes are connected to this node?
# 
# 
# *This function should return a tuple (name of node, number of satisfied connected nodes).*

# In[13]:

def answer_eleven():
        
    # Your Code Here
    
    # initialise the graph
    G = answer_six()
    
    # get the diameter
    diameter = answer_eight()
    
    # get list of shorts paths
    shortes_paths = nx.shortest_path_length(G)
    
    # initialize the shortlist
    short_list = []
    
    # iterate over each key
    for source, value in shortes_paths.items():
        
        # keep a count of number of paths per key that are equal to diameter
        short_paths_from_source = 0
        
        # iterate over each destination and length pair
        for destination, length in value.items():
            # add a count if the destination is  equal to diameter
            if length == diameter: short_paths_from_source += 1
        
        # add the source and total short paths to the shortlist to be sorted below
        short_list.append((source, short_paths_from_source))
        
    # sort the shortlist in descending order based on number of short paths == diameter
    short_list = sorted(short_list, key=lambda x: x[1], reverse = True)
    
    # get the best node at the top of the list
    best_node = short_list[0][0]
    
    # find the neighbors of the center node
    best_node_paths = short_list[0][1]
    
    return (best_node, best_node_paths)  # Your Answer Here

answer_eleven()


# ### Question 12
# 
# Suppose you want to prevent communication from flowing to the node that you found in the previous question from any node in the center of G_sc, what is the smallest number of nodes you would need to remove from the graph (you're not allowed to remove the node from the previous question or the center nodes)? 
# 
# *This function should return an integer.*

# In[36]:

def answer_twelve():
        
    # Your Code Here
    
    # initialise the graph
    G = answer_six()
    
    # get the list of nodes at the center
    center_list = nx.center(G)
    
    # get the best node with the shortest paths
    best_node = answer_eleven()[0]
    
    # find the minimum nodes that would disconnect the graph 
    min_nodes = nx.minimum_node_cut(G,center_list[0], best_node)
    
    return len(min_nodes) # Your Answer Here

answer_twelve()


# ### Question 13
# 
# Construct an undirected graph G_un using G_sc (you can ignore the attributes).
# 
# *This function should return a networkx Graph.*

# In[21]:

def answer_thirteen():
        
    # Your Code Here
    
    # initialise the graph
    G = answer_six()
    
    # initiaise the undirected
    G2 = G.to_undirected()
    
    # remove weighs by converting to graph
    G_un = nx.Graph(G2)
    
    return G_un # Your Answer Here

answer_thirteen()


# ### Question 14
# 
# What is the transitivity and average clustering coefficient of graph G_un?
# 
# *This function should return a tuple (transitivity, avg clustering).*

# In[22]:

def answer_fourteen():
        
    # Your Code Here
    
    # initialise the graph
    G = answer_thirteen()
    
    # get the transitivity
    transitivity = nx.transitivity(G)
    
    # get the avg clustering coefficient
    avg_clustering_coeff = nx.average_clustering(G)

    
    return (transitivity, avg_clustering_coeff)# Your Answer Here

answer_fourteen()

