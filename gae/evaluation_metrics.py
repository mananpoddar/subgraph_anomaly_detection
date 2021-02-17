# import cv2
import numpy as np 
# from input_data import load_data
import random
from scipy.sparse import csr_matrix
import scipy.io
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


class EvaluationMetrics(object):
    def __init__(self, data_name, data_predicted, data_labels):
        self.data_name = data_name
        data = scipy.io.loadmat("matfiles/{}.mat".format(data_name))
        self.adj = sp.lil_matrix(data["A"])
        self.adj = self.adj.toarray()
        data = scipy.io.loadmat("matfiles/{}.mat".format(data_labels))
        self.output_subgraphs = data["labels"][0]
        data = scipy.io.loadmat("matfiles/{}.mat".format(data_predicted))
        self.predicted_subgraphs = data["predicted_subgraph"][0]
        print(self.predicted_subgraphs)
        print(self.output_subgraphs)
        (self.edges, self.predicted_edges, self.output_edges) = self.get_edge_list()
        # print(self.edges)
        # print(self.predicted_edges)
        # print(self.output_edges)
        output_edge_labels = self.get_edge_labels(self.edges, self.output_edges)
        predicted_edge_labels = self.get_edge_labels(self.edges, self.predicted_edges) 
        auc = roc_auc_score(output_edge_labels, predicted_edge_labels)
        print(auc)

        predicted_nodes = []
        actual_nodes = []
        
        output_list = []
        predicted_list = []
        for out in self.output_subgraphs:
            l1 = list(out)
            l1 = l1[0]
            output_list.append(l1)
        for pre in self.predicted_subgraphs:
        	l1 = list(pre)
        	l1 = l1[0]
        	predicted_list.append(l1)
         
        predicted_nodes = self.addNodes(predicted_list)
        actual_nodes = self.addNodes(output_list)

        final_predicted_nodes = []
        final_actual_nodes = []

        (r, c) = self.adj.shape
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

    def get_edge_list(self):
    	edge_list=[]
    	predicted_edges=[]
    	output_edges=[]
    	for out in self.output_subgraphs:
    		lis = list(out)[0]
    		for i in lis:
    			for j in lis:
    				if (i != j) and ((i,j) not in edge_list) and self.adj[i][j]==1:
    					edge_list.append((i,j))
    					output_edges.append((i,j))

    	for pre in self.predicted_subgraphs:
    		lis = list(pre)[0]
    		for i in lis:
    			for j in lis:
    				if (i != j) and ((i,j) not in edge_list) and self.adj[i][j]==1:
    					edge_list.append((i,j))   
    					predicted_edges.append((i,j))

    	return (edge_list, predicted_edges, output_edges) 

    def get_edge_labels(self, par_list, comp_list):
    	labels=[]
    	for edge in par_list:
    		if edge in comp_list:
    			labels.append(1)
    		else:
    			labels.append(0)
    	return labels

    def addNodes(self, subgraphs) :
        nodes = []
        for subgraph in subgraphs :
            for node in subgraph :
                nodes.append(node)
        return nodes


obj = EvaluationMetrics('facebook_anomaly','predicted_subgraph','facebook_labels')