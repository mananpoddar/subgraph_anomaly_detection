from __future__ import division
from __future__ import print_function
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
from constructor import get_placeholder, get_model, get_optimizer, update
import numpy as np
from input_data import format_data
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import pandas as pd
from input_data import load_data
import networkx as nx
import itertools
import numpy as np
import community as community_louvain
from scipy import sparse
import scipy.sparse as sp
import scipy.io
from threshold import Threshold
from evaluation_metrics import EvaluationMetrics

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

import matplotlib.pyplot as plt
from collections import Counter

import math

class AnomalyDetectionRunner():
    def __init__(self, settings):
        self.data_name = settings['data_name']
        self.iteration = settings['iterations']
        self.model = settings['model']


    def normalise(self, value, low, high):
        return (value-low)/(high-low)

    def runAutoEncoder(self, model_str, feas):
        placeholders = get_placeholder()

        # construct model
        gcn_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])

        # Optimizer
        opt = get_optimizer(model_str, gcn_model, placeholders, feas['num_nodes'], FLAGS.alpha)

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Train model
        for epoch in range(1, self.iteration+1):

            reconstruction_errors, reconstruction_loss, embeddings = update(gcn_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])
            if epoch % 10 == 0:
                print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(reconstruction_loss))

            # if epoch % 100 == 0:
            # y_true = [label[0] for label in feas['labels']]
        sess.close()
        return reconstruction_errors, reconstruction_loss, embeddings

    def getCoarsenedGraph(self,embeddings):
        sim_matrix = embeddings.dot(np.transpose(embeddings))
        adj, features, labels = load_data(self.data_name)

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

        nodes = max(partition.values())+1
        new_adj = np.zeros((nodes,nodes))
        features=features.toarray()
        (a,col) = features.shape
        new_features = list()

        for i in range(r):
            for j in range(c):
                if adj[i][j]==1 and partition[i]!=partition[j]:
                    new_adj[partition[i]][partition[j]]=1

        feature_dict = dict()
        for node, partnum in partition.items():
            if partnum not in feature_dict:
                feature_dict[partnum] = list(features[node])
            else:
                for i in range(len(list(features[node]))):
                    feature_dict[partnum][i] = max(feature_dict[partnum][i],features[node][i]) 
        
        for i in range(nodes):
            new_features.append([])
        for i in range(nodes):
            new_features[i] = list(feature_dict[i])
 
        new_adj=sparse.csr_matrix(new_adj) 
        new_features = sparse.csr_matrix(new_features)       
        new_labels = sparse.csr_matrix(np.zeros(nodes))     

        return partition, new_adj, new_features, new_labels 

    # adj is original graph adj, partition to node is the dictionary, partnum is the partition number 
    def getAdjSubgraph(self, adj, partition_to_node, partnum):
        nodes = partition_to_node[partnum]
        G=nx.Graph()
        adj = adj.toarray()
        for i in nodes:
            for j in nodes:
                if i==j :
                    continue
                if adj[i][j]==1:
                    G.add_edge(i,j)
        return G

    def saveSubGraphs(self, adj, partition_to_node, num_nodes):
        

         for i in range(num_nodes):
            node_sizes = []
            for j in range(len(partition_to_node[i])) :
                node_sizes.append(0.1)

            plt.clf()
            G = self.getAdjSubgraph(adj,partition_to_node, i)
            pos=nx.spring_layout(G)
            nx.draw(G,pos,node_size = node_sizes, node_color='#A0CBE2',edge_color='#BB0000',width=0.1,edge_cmap=plt.cm.Blues,with_labels=False)
            plt.savefig("output/subGraphs/" + str(i) + ".png", dpi=1000, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1)


    def saveGraphs(self, adj, filename):
        adj = adj.toarray();
        (r,c) = adj.shape
        if filename=="original":
            #plot original graph
            node_sizes = []
            num_nodes = r
            for i in range(num_nodes) :
                node_sizes.append(0.1)

            plt.clf()
            G = nx.Graph(adj)
            pos=nx.spring_layout(G)   #G is my graph
            nx.draw(G,pos,node_size = node_sizes, node_color='#A0CBE2',edge_color='#BB0000',width=0.1,edge_cmap=plt.cm.Blues,with_labels=False)
            plt.savefig("output/original.png", dpi=1000, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1)

        if filename=="shrink":
            plt.clf() 
            G1 = nx.Graph(adj)
            nx.draw(G1, with_labels = True) 
            plt.savefig("output/shrink.png")


    def erun(self):
        model_str = self.model
        feas = format_data(self.data_name, None)
        # self.saveGraphs(feas['adj'], "original")
    
        reconstruction_errors, reconstruct_loss, embeddings = self.runAutoEncoder(model_str,feas)
        partition, new_adj, new_features, new_labels = self.getCoarsenedGraph(embeddings)
        new_feas =  format_data(self.data_name, new_adj, new_features, new_labels)

        # self.saveGraphs(new_adj, "shrink")

        partition_to_node = dict()
        for node, partnum in partition.items():
            if partnum not in partition_to_node:
                partition_to_node[partnum]=[]
            partition_to_node[partnum].append(node)

        self.saveSubGraphs(feas['adj'], partition_to_node, new_adj.shape[0])

        reconstruction_errors, reconstruction_loss, embeddings = self.runAutoEncoder(model_str, new_feas)





        for i in range(len(reconstruction_errors)) :
            reconstruction_errors[i] /= len(partition_to_node[i])


        threshold = Threshold(reconstruction_errors, 0)
        optimum_threshold = threshold.optimumThreshold()

        false_count = 0
        num_count = 0
        predicted_subgraphs = []
        for i in range(len(reconstruction_errors)):
            num = reconstruction_errors[i]
            if num > optimum_threshold:
                num_count = num_count + 1
                predicted_subgraphs.append(partition_to_node[i])
        
        data = scipy.io.loadmat("./data/"+self.data_name+"_labels.mat")

        output_subgraphs = data["labels"]

        
        output_subgraphs = output_subgraphs[0]

        evaluation_metrics = EvaluationMetrics(output_subgraphs, predicted_subgraphs, feas['adj'])
        

        sorted_errors = np.argsort(-reconstruction_errors, axis=0)  
        with open('output/{}-ranking.txt'.format(self.data_name), 'w') as f:
            for index in sorted_errors:
                f.write("%s\n" % feas['labels'][0][index])

        df = pd.DataFrame({'AD-GCA':reconstruction_errors})
        df.to_csv('output/{}-scores.csv'.format(self.data_name), index=False, sep=',')
        




