
import cv2
import numpy as np 
from input_data import load_data
import random
from scipy.sparse import csr_matrix
import scipy.io

class PartitionGraph(object):
    def __init__(self, embeddings, adj, features, labels):
        self.embeddings = embeddings
        self.adj = adj
        self.features = features
        self.labels = labels
        


    def getPartition(self):
        embeddings = self.embeddings
        adj = self.adj
        features = self.features
        labels = self.labels

        sim_matrix = embeddings.dot(np.transpose(embeddings))

        #plot original graph
        node_sizes = []
        for i in range(len(labels)) :
            node_sizes.append( 0.1)

        adj=adj.toarray()

        (r, c) = adj.shape
        sim_matrix = list(sim_matrix)

        G=nx.Graph()

        for i in range(r):
            for j in range(c):
                if adj[i][j]==1:
                    # print(adj[i][j])
                    G.add_edge(i,j,weight=sim_matrix[i][j])
                    sim_matrix[i][j]=sim_matrix[i][j]
                    
                else :
                    sim_matrix[i][j]=0

        partition = community_louvain.best_partition(G)
        print(partition)
       