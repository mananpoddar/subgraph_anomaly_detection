# import cv2
import numpy as np 
# from input_data import load_data
import random
from scipy.sparse import csr_matrix
import scipy.io
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


class EvaluationMetrics(object):
    def __init__(self, output_subgraphs, predicted_subgraphs, adj):
        self.predicted_subgraphs = predicted_subgraphs
        self.output_subgraphs = output_subgraphs
        self.adj = adj.toarray()
        (self.edges, self.predicted_edges, self.output_edges) = self.get_edge_list()
   
        output_edge_labels = self.get_edge_labels(self.edges, self.output_edges)
        predicted_edge_labels = self.get_edge_labels(self.edges, self.predicted_edges) 
        auc = roc_auc_score(output_edge_labels, predicted_edge_labels)
        tpc = 0
        fpc = 0
        for i in range(len(output_edge_labels)):
            if predicted_edge_labels[i]:
                if output_edge_labels[i]:
                    tpc+=1
                else:
                    fpc+=1
        print("True Positive Count"+str(tpc))
        print("False Positive Count"+str(fpc))
        print("edge accuracy")
        print(auc)
        fpr, tpr, thresholds = roc_curve(output_edge_labels, predicted_edge_labels)
        print("Edge Values")
        print("TPR"+str(tpr))
        print("FPR"+str(fpr))
        print('------------------------------------------------------------------------------')
        print("node accuracy")
        self.tpce = tpc
        self.fpce = fpc
        self.tpcn = self.fpcn = 0
        self.actual_node_count = 0
        self.node_accuracy = self.getNodeAccuracy()
        self.edge_accuracy = auc
        print(self.node_accuracy)


    def cosine_similarity(self):
        a = 1
    #     def counter_cosine_similarity(c1, c2):
    # terms = set(c1).union(c2)
    # dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    # magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    # magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    # ans = dotprod / (magA * magB)
    # return ans

        # print(output_subgraphs)
        # print('--------------------')
        # for out in output_subgraphs:
        #     max_similarity = -1
        #     predicted_ans = []
        #     for pre in predicted_subgraphs:
        #         l1 = list(out)
        #         l1 = l1[0]
        #         l2 = list(pre)
        #         c1 = Counter(l1)
        #         c2 = Counter(l2)
        #         # max_similarity = max(max_similarity,counter_cosine_similarity(c1,c2))
        #         if max_similarity < counter_cosine_similarity(c1,c2) :
        #             max_similarity = counter_cosine_similarity(c1,c2)
        #             predicted_ans = l2
        #     print("max similarity ")
        #     print(max_similarity)
        #     print(predicted_ans)
        #     print(list(out)[0])

            
        #     accuracy += max_similarity

        # for pre in predicted_subgraphs:
        #     max_similarity = 0
        #     for out in output_subgraphs:
        #         l1 = list(out)
        #         l1 = l1[0]
        #         l2 = list(pre)
        #         c1 = Counter(l1)
        #         c2 = Counter(l2)
        #         max_similarity = max(max_similarity,counter_cosine_similarity(c1,c2))
        #     accuracy += max_similarity

        # accuracy /= (len(output_subgraphs) +len(predicted_subgraphs))

        # print(accuracy)



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
    		lis = pre
    		for i in lis:
    			for j in lis:
    				if (i != j) and ((i,j) not in edge_list) and self.adj[i][j]==1:
    					edge_list.append((i,j))
    					predicted_edges.append((i,j))
    				elif (i != j) and ((i,j) not in predicted_edges) and self.adj[i][j]==1:
    					predicted_edges.append((i,j))
    	edge_list = []
    	(r,c) = self.adj.shape
    	for i in range(r):
    		for j in range(c) :
    			if self.adj[i][j]:
    				edge_list.append((i,j))
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

    def getNodeAccuracy(self):
        predicted_nodes = []
        actual_nodes = []
        
        output_list = []
        predicted_list = []
        for out in self.output_subgraphs:
            l1 = list(out)
            l1 = l1[0]
            output_list.append(l1)
        for pre in self.predicted_subgraphs:
        	l1 = pre
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
                self.actual_node_count+=1
            else :
                final_actual_nodes.append(0)

        # final_actual_nodes = scipy.io.loadmat("./data/"+"Amazon_anomaly_node_labels.mat")
        # final_actual_nodes = final_actual_nodes['labels'][0]
        # print("actual")
        # for ele in final_actual_nodes:
        #     print(ele)
      
        # print(final_actual_nodes)
        # print("predicted")
        # print(final_predicted_nodes)
        tpc = 0
        fpc = 0
        for i in range(len(final_actual_nodes)):
            if final_predicted_nodes[i]:
                if final_actual_nodes[i]:
                    tpc+=1
                else:
                    fpc+=1
        print("True Positive Count"+str(tpc))
        print("False Positive Count"+str(fpc))

        self.tpcn = tpc
        self.fpcn = fpc


        auc = roc_auc_score(final_actual_nodes, final_predicted_nodes)
        fpr, tpr, thresholds = roc_curve(final_actual_nodes, final_predicted_nodes)
        print("Node Values")
        print("TPR"+str(tpr))
        print("FPR"+str(fpr))
        return auc
