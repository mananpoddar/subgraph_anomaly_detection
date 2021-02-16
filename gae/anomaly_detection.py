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
    return dotprod / (magA * magB)

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

            # auc = roc_auc_score(y_true, reconstruction_errors)

            # print("ruc_auc")
            # print(auc)
            # print("ytrue")
            # print(y_true)


        y_true = [label[0] for label in feas['labels']]

        sess.close()

        # skplt.metrics.plot_roc_curve(y_true, reconstruction_errors)
        # plt.show()

        # similarity matrix construction

        sim_matrix = embeddings.dot(np.transpose(embeddings))
        print("BEfore")
        adj, features, labels = load_data(self.data_name)

        #plot original graph
        node_sizes = []
        for i in range(len(labels)) :
            node_sizes.append( 0.1)

        # plt.clf()
        # G = nx.Graph(adj)
        # pos=nx.spring_layout(G)   #G is my graph
        # nx.draw(G,pos,node_size = node_sizes, node_color='#A0CBE2',edge_color='#BB0000',width=0.1,edge_cmap=plt.cm.Blues,with_labels=False)
        # plt.savefig("original.png", dpi=1000, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1) 


        adj=adj.toarray()

        (r, c) = adj.shape
        sim_matrix = list(sim_matrix)

        G=nx.Graph()

        # high = max(np.array(sim_matrix))
        # low = min(np.array(sim_matrix))

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

        # for node, partnum in partition.items():
        #     partition[node] = partition[node]
        
        

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
                    # print("DICT")
                    # print(feature_dict[partnum][i])
                    # print("FEAT")
                    # print(features[node][i])
                    feature_dict[partnum][i] = max(feature_dict[partnum][i],features[node][i]) 
        
        for i in range(nodes):
            new_features.append([])
        for i in range(nodes):
            new_features[i] = list(feature_dict[i])
 
        new_adj=sparse.csr_matrix(new_adj) 
        new_features = sparse.csr_matrix(new_features)       
        new_labels = sparse.csr_matrix(np.zeros(nodes))
        feas =  format_data(self.data_name, new_adj, new_features, new_labels)

        #plot shrink graph
        # clearing the current plot 
        # plt.clf() 
        # G1 = nx.Graph(new_adj)
        # nx.draw(G1, with_labels = True) 
        # plt.savefig("shrink.png")

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

        print(reconstruction_errors)            


        # print("threshold : " , threshold)
        # print("false positive : ", fpr)
        # print("true positive : ", tpr)

        # plt.title('Receiver Operating Characteristic')
        # plt.plot(fpr, tpr, 'b')
        # plt.legend(loc = 'lower right')
        # plt.plot([0, 1], [0, 1],'r--')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.ylabel('True Positive Rate')
        # plt.xlabel('False Positive Rate')
        # plt.show()



        # plot reconstruction_errors vs y_true
        # print("plotting started..")
        # print("1 count")
        # print(y_true.count(1))
         # Partition to Node mapping
        partition_to_node = dict()
        for node, partnum in partition.items():
            if partnum not in partition_to_node:
                partition_to_node[partnum]=[]
            partition_to_node[partnum].append(node)

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
        
        data = scipy.io.loadmat("facebook_labels.mat")
        output_subgraphs = data["labels"]

        # Evaluate metrics using output_subgraphs and predicted_subgraphs

        accuracy = 0
        output_subgraphs = output_subgraphs[0]

        print(output_subgraphs)
        print('--------------------')
        for out in output_subgraphs:
            max_similarity = 0
            for pre in predicted_subgraphs:
                l1 = list(out)
                l1 = l1[0]
                l2 = list(pre)
                c1 = Counter(l1)
                c2 = Counter(l2)
                max_similarity = max(max_similarity,counter_cosine_similarity(c1,c2))
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

        accuracy /= (len(output_subgraphs)+len(predicted_subgraphs))

        print(accuracy)

        # print("mean " + str(mean))
        # right = 0
        # for i in range(len(reconstruction_errors)):
        #     if reconstruction_errors[i] >= mean and y_true[i]==1:
        #         right = right+1
        #     elif reconstruction_errors[i] < mean and y_true[i]==0:
        #         right = right + 1
        # print("right" + str(right))
        # print("length" + str(len(reconstruction_errors)))
        # print(right/len(reconstruction_errors))



        # # plt.plot(reconstruction_errors,y_true,"ro")
        # # plt.show()

        # print("plotting done..")


        # print("reconstruction_errors")
        # print(reconstruction_errors.shape)
        # # reconstruction_errors = [1, 2, 3, 4, 5]
        # print(reconstruction_errors)
        
        # # normalise reconstruction_errors
        # new_reconstruction_error = []
        # for ele in reconstruction_errors:
        #     new_reconstruction_error.append(100000*( (ele-min(reconstruction_errors))/(max(reconstruction_errors)-min(reconstruction_errors))))
        
        # # denormalise 800
        # denormalised_threshold = (450*( max(reconstruction_errors)-min(reconstruction_errors) )) / 1000  + min(reconstruction_errors)
        # print("deo thre")
        # print(denormalised_threshold)

        # right = 0
        # for i in range(len(reconstruction_errors)):
        #     if reconstruction_errors[i] >= denormalised_threshold and y_true[i]==1:
        #         right = right+1
        #     elif reconstruction_errors[i] < denormalised_threshold and y_true[i]==0:
        #         right = right + 1
        # print("right" + str(right))
        # print("length" + str(len(reconstruction_errors)))
        # print("deo score:")
        # print(right/len(reconstruction_errors))

        # # get false positive/negative and 1 pe 1
        # false_positive = 0
        # false_negative = 0
        # true_positive = 0
        # denormalised_threshold = 170000
        # for i in range(len(reconstruction_errors)):
        #     # false positive
        #     if reconstruction_errors[i] >= denormalised_threshold and y_true[i]==0:
        #         false_positive = false_positive+1
        #     # false negative
        #     elif reconstruction_errors[i] < denormalised_threshold and y_true[i]==1:
        #         false_negative = false_negative + 1
        #     # true positive
        #     elif reconstruction_errors[i] >= denormalised_threshold and y_true[i]==1:
        #         true_positive = true_positive + 1

        # print("false positive "+ str(false_positive) + "false negative "+str(false_negative) + "true postive "+ str(true_positive) )



        # histogram = dict()
        # norm_value = 150000
        # for ele in reconstruction_errors:
        #     ele = ele - norm_value
        #     if ele not in histogram and (ele >=0 and ele <=10000) :
        #         histogram[ele] = 0
        #     if ele >= 0 and ele <= 10000:
        #         histogram[ele] = histogram[ele]+1
        # print("dictionary")
        # print(histogram)
        # # plt.hist(histogram)
        # plt.bar(histogram.keys(), histogram.values(), 1, color='g')

        # plt.show()

        

        
        # adj, features, labels = load_data(self.data_name)
        # print("adjacency")
        # print(adj.shape)
        # # print(adj)

        # # graph G using networkx library
        # G = nx.Graph(adj)

        # # store the subgraphs nodes and their corresponding anomaly metrics
        # subgraph_rank_list = []

        # # iterate over all the possible subdue subgraphs of size 2
        # for sub_nodes in itertools.combinations(G.nodes(),2):
        #     # sub_nodes = [1, 2]
        #     subg = G.subgraph(sub_nodes)
        #     # check if the subgraph is connected
        #     if nx.is_connected(subg):
        #         subgraph_nodes = subg.nodes
        #         sum = 0
        #         for nodes in subgraph_nodes:
        #             # take the sum of all the node errors present in the subgraph
        #             sum = sum + reconstruction_errors[nodes]
        #         subgraph_rank_list.append((sum,subgraph_nodes))
        # subgraph_rank_list.sort()

        # # print the elements
        # for element in subgraph_rank_list:
        #     print("elements")
        #     print(element[0])
        #     print(element[1])


                

        sorted_errors = np.argsort(-reconstruction_errors, axis=0)  
        with open('output/{}-ranking.txt'.format(self.data_name), 'w') as f:
            for index in sorted_errors:
                f.write("%s\n" % feas['labels'][0][index])

        df = pd.DataFrame({'AD-GCA':reconstruction_errors})
        df.to_csv('output/{}-scores.csv'.format(self.data_name), index=False, sep=',')
        




