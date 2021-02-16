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

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

import matplotlib.pyplot as plt
from collections import Counter

# import torch
# import sklearn.metrics as metrics
# import scikitplot as skplt

import math
def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    ans = dotprod / (magA * magB)
    return ans

def addNodes(subgraphs) :
        nodes = []
        for subgraph in subgraphs :
            for node in subgraph :
                nodes.append(node)
        return nodes
class AnomalyDetectionRunner():
    def __init__(self, settings):
        self.data_name = settings['data_name']
        self.iteration = settings['iterations']
        self.model = settings['model']


    def normalise(self, value, low, high):
        return (value-low)/(high-low)

    def erun(self):
        model_str = self.model
        # load data
        print(model_str)

        feas = format_data(self.data_name, None)
        print("feature number: {}".format(feas['num_features']))

        # Define placeholders
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
            y_true = [label[0] for label in feas['labels']]


        y_true = [label[0] for label in feas['labels']]

        sess.close()
        
        scipy.io.savemat("./matfiles/"+"reconstruction_errors"+'.mat', mdict={'reconstruction_errors':reconstruction_errors})

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


        print("Similarity MAtrix")
        print(sim_matrix[0])

        # fpr, tpr, threshold = roc_curve(y_true, reconstruction_errors)

        partition = community_louvain.best_partition(G)
        print(partition)

        # Making Reduced Graph

        nodes = max(partition.values())+1
        new_adj = np.zeros((nodes,nodes))
        print(type(features))
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
        feas =  format_data(self.data_name, new_adj, new_features, new_labels)

        print("feature number: {}".format(feas['num_features']))


        # construct model
        gcn_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])

        # Optimizer
        opt = get_optimizer(model_str, gcn_model, placeholders, feas['num_nodes'], FLAGS.alpha)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())


        # Train model
        for epoch in range(1, self.iteration+1):

            reconstruction_errors, reconstruction_loss, embeddings = update(gcn_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])
            if epoch % 10 == 0:
                print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(reconstruction_loss))


        partition_to_node = dict()
        for node, partnum in partition.items():
            if partnum not in partition_to_node:
                partition_to_node[partnum]=[]
            partition_to_node[partnum].append(node)

        for i in range(len(reconstruction_errors)) :
            reconstruction_errors[i] /= len(partition_to_node[i])

        num_count = 0
        mean = 0
        for num in reconstruction_errors:
            # if num > 200000:
            #     num_count = num_count + 1
            mean = mean + num 
        
        mean = (mean/len(reconstruction_errors))

        # mean = 170000 

        # mean = mean + mean/10

        false_count = 0
        predicted_subgraphs = []
        for i in range(len(reconstruction_errors)):
            num = reconstruction_errors[i]
            if num > mean:
                num_count = num_count + 1
                predicted_subgraphs.append(partition_to_node[i])
          
        print("Num count " + str(num_count))

        print(predicted_subgraphs)
        
        data = scipy.io.loadmat("./matfiles/facebook_labels.mat")

        output_subgraphs = data["labels"]
        print("labels")
        print(output_subgraphs)

        # Evaluate metrics using output_subgraphs and predicted_subgraphs

        accuracy = 0
        output_subgraphs = output_subgraphs[0]

        print(output_subgraphs)
        print('--------------------')
        for out in output_subgraphs:
            max_similarity = -1
            predicted_ans = []
            for pre in predicted_subgraphs:
                l1 = list(out)
                l1 = l1[0]
                l2 = list(pre)
                c1 = Counter(l1)
                c2 = Counter(l2)
                # max_similarity = max(max_similarity,counter_cosine_similarity(c1,c2))
                if max_similarity < counter_cosine_similarity(c1,c2) :
                    max_similarity = counter_cosine_similarity(c1,c2)
                    predicted_ans = l2
            print("max similarity ")
            print(max_similarity)
            print(predicted_ans)
            print(list(out)[0])

            
            accuracy += max_similarity

        for pre in predicted_subgraphs:
            max_similarity = 0
            for out in output_subgraphs:
                l1 = list(out)
                l1 = l1[0]
                l2 = list(pre)
                c1 = Counter(l1)
                c2 = Counter(l2)
                max_similarity = max(max_similarity,counter_cosine_similarity(c1,c2))
            accuracy += max_similarity

        accuracy /= (len(output_subgraphs) +len(predicted_subgraphs))

        print(accuracy)

        predicted_nodes = []
        actual_nodes = []
        
     

        
        output_list = []
        for out in output_subgraphs:
            l1 = list(out)
            l1 = l1[0]
            output_list.append(l1)
                
        
        predicted_nodes = addNodes(list(predicted_subgraphs))
        actual_nodes = addNodes(output_list)

        final_predicted_nodes = []
        final_actual_nodes = []
        for i in range(r) :
            if i in predicted_nodes :
                final_predicted_nodes.append(1)
            else : 
                final_predicted_nodes.append(0)

            if i in actual_nodes :
                final_actual_nodes.append(1)
            else :
                final_actual_nodes.append(0)


        
        auc = roc_auc_score(final_actual_nodes, final_predicted_nodes)
        print("final accuracy")
        print(auc)



        sorted_errors = np.argsort(-reconstruction_errors, axis=0)  
        with open('output/{}-ranking.txt'.format(self.data_name), 'w') as f:
            for index in sorted_errors:
                f.write("%s\n" % feas['labels'][0][index])

        df = pd.DataFrame({'AD-GCA':reconstruction_errors})
        df.to_csv('output/{}-scores.csv'.format(self.data_name), index=False, sep=',')
        




