
import cv2
import numpy as np 
from input_data import load_data
import random
from scipy.sparse import csr_matrix
import scipy.io

class IntroduceAnomaly(object):
    def __init__(self, data_name):
        self.data_name = data_name
        adj, features, labels = load_data(self.data_name)
        self.adj = adj.toarray()
        self.features = features    
        self.labels = labels
        num_nodes = len(self.adj)
        print("num nodes")
        print(num_nodes)
        self.nodes=[i for i in range(num_nodes)]
        self.anomalous_nodes=random.choices(self.nodes,k=300)
        self.anomalous_subgraph = []

        self.addStructuralAnomaly()
        self.addFeatureAnomaly()
        self.saveMat()


    def addStructuralAnomaly(self):
        nodes = self.anomalous_nodes
        adj = self.adj
        for i in range(4):
            clique=nodes[i*10:i*10+9]
            self.anomalous_subgraph.append(clique)
            for node1 in clique:
                for node2 in clique:
                    if node1!=node2:
                    
                        adj[node1][node2] = 1
                        adj[node2][node1] = 1
        self.adj = adj
    

    def addFeatureAnomaly(self):
        nodes = self.anomalous_nodes[0:150]
        node_list =[]
        for i in range(len(self.adj)):
            if i not in nodes :
                node_list.append(i)

        A = csr_matrix.todense(self.features)
        for k in range(4) :
            i = random.choice(node_list)
            count = 0
            count = count + 1
            if count > 10 :
                break

            size_subgraph = 15
            neighbour_nodes = [i]
            selected_nodes = []

            for j in range(size_subgraph):
                if len(neighbour_nodes) == 0 :
                    break
                element = neighbour_nodes[ int( random.uniform(0,len(neighbour_nodes) - 1) ) ]  

                selected_nodes.append(element)

                if element in node_list:
                    node_list.remove(element)

                for column in range(len(self.adj)):
                    if self.adj[element][column] and column not in selected_nodes :
                        neighbour_nodes.append(column)

                neighbour_nodes.remove(element)

            random_set=random.choices(node_list,k=50)
           
    
            for anomalous_node in selected_nodes:
                node_J=-1
                node_dist=0
               
                for d in random_set:
                    dist=np.linalg.norm( np.array(A[anomalous_node]) - np.array(A[d]) )
                    if dist>node_dist:
                        node_dist=dist
                        node_J=d
                A[i]=A[node_J]

            self.features = A
            self.anomalous_subgraph.append(selected_nodes)


    def saveMat(self):
        temp=[0 for i in range(len(self.adj))]
        scipy.io.savemat("./data/"+ self.data_name +'_anomaly.mat', mdict={'X':self.features,'A':self.adj,'gnd':temp})
        scipy.io.savemat("./data/"+ self.data_name +'_anomaly_subgraph_labels.mat', mdict={'labels':self.anomalous_subgraph})
        anomalous_nodes = [0] * len(self.adj)
        if(self.data_name != "facebook"):
            for subgraph in self.anomalous_subgraph :
                for node in subgraph :
                    anomalous_nodes[node] = 1
            
            for i in range(len(self.labels)):
                if self.labels[i] :
                    anomalous_nodes[i] = 1
            
            scipy.io.savemat("./data/"+ self.data_name +'_anomaly_node_labels.mat', mdict={'labels':anomalous_nodes})

 
# obj = IntroduceAnomaly("facebook")
obj = IntroduceAnomaly("Amazon")



        
