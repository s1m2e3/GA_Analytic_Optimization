#Sam, modified by Kyle Norland
#Modified 10/10/22

import numpy as np
import cvxpy as cvx
import random
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime



class Smart_Crossover:

    def __init__(self,learners_info):
        
        self.learned_info=learners_info
        
        #Generate the ensemble.
        self.ensemble = self.ensemble_learned_info()
        
        print("Ensembled learned info")
        #NOTE: Ensemble unlearned will blow up.
        print(self.ensemble)
        #Graph it
        self.graph = nx.DiGraph()
        
        #Prepare
        rew_matrix,source_state,sink_state = self.prep()
        self.source_state = list(self.graph.nodes)[source_state]
        self.sink_state = list(self.graph.nodes)[sink_state]
        
        self.solution=[]
        condition=True
        cycle = []
        self.cycle=[]
        self.n_iter = 0
        self.value=0
        while condition:
            self.solution,cycle = self.optimize(rew_matrix,source_state,sink_state,cycle)
            self.n_iter+=1
            if len(cycle)==0:
                condition=False
        
        
    def ensemble_learned_info(self):
        #Collect globally known data.
        total_info={}
        edges={}
        unknown_exist=[]
        for learner in self.learned_info:
            #Edges has visited states?
            edges[learner]=list(self.learned_info[learner]["state-reward"])
            print("Edges for learner")
            print(edges[learner])
            
            for state in self.learned_info[learner]["state-reward"]:
                if state not in list(total_info):
                    total_info[state]=self.learned_info[learner]["state-reward"][state]
            #Is there a check for if another state has learned connections that are not known to another?        
            for unknowns in self.learned_info[learner]["not_known"]:
                if unknowns not in unknown_exist:
                    unknown_exist.append(unknowns)
                    
        total_info={"state-reward":total_info,"edges":edges,"not_known":unknown_exist}
        return total_info
        
        
    def color_nodes(self,g):
        nodes_with_reward=[]
        nodes_without_reward=[]
        color = []
        print(g.nodes)
        for node in g.nodes:
            print(node)
            if self.ensemble["state-reward"][node]>0:
                nodes_with_reward.append(node)
                color.append("red")
            else:
                nodes_without_reward.append(node)
                color.append("blue")
        
        
        return color
    def color_edges(self,g):
        
        edges_color = []
        for edge in g.edges:
            if edge in self.ensemble["not_known"]:
                edges_color.append('gray')
            else:
                edges_color.append('black')
        return edges_color
        
        
    def prep(self):

        sns.set(style='darkgrid')

        dict_states = list(self.ensemble["edges"].values())
        new=[]
        for i in dict_states:
            for j in i:
                new.append(j)
        
        array_states=[*set(new)]
        
        #Randomly generated end of edges?
        array_edges=[(i[j],i[j+1]) for i in dict_states for j in range(len(i)-1)]
        
        print("Array edges")
        print(array_edges)
        self.graph.add_nodes_from(array_states)
        self.graph.add_edges_from(array_edges)
        
        #add rewards to nodes in graph
        for node in self.graph.nodes:
            self.graph.nodes[node]["reward"]=self.ensemble["state-reward"][node]
        

        #plot learned graph        
        #plt.figure(figsize=(20,10))
        #color = self.color_nodes(self.graph)
        #nx.draw(self.graph,node_color=color,with_labels=True)
        #plt.savefig("first_graph.png")
        
        

        #randomize source
        choose_learner = random.choice(list(self.learned_info))
        source_state = list(self.learned_info[choose_learner]["state-reward"])[0]      
        #randomize sink
        #!!! Will sometimes cause infinite loop
        sink_state = source_state
        while sink_state == source_state:
            choose_learner = random.choice(list(self.learned_info))
            sink_state = list(self.learned_info[choose_learner]["state-reward"])[-1]
        
        #color[list(self.graph.nodes).index(source_state)]="green"
        #color[list(self.graph.nodes).index(sink_state)]="brown"
        
        #Make the adjacency matrix out of the previously added edges.
        adj_matrix = np.array(nx.attr_matrix(self.graph,rc_order=self.graph.nodes))
        rew_matrix = np.zeros(adj_matrix.shape)
        for row_index in range(len(adj_matrix)):
            for col_index in range(len(adj_matrix)): 
                if adj_matrix[row_index][col_index]==1:
                    rew_matrix[row_index][col_index]=self.ensemble["state-reward"][array_states[col_index]]            
                

        self.graph.add_edges_from(self.ensemble['not_known'])
        
        #If altered, make it -2 reward for nonexistent edges?
        altered = np.array(nx.attr_matrix(self.graph,rc_order=self.graph.nodes))
        for row_index in range(len(altered)):
            for col_index in range(len(altered)): 
                if altered[row_index][col_index]==1 and rew_matrix[row_index][col_index]==0 :
                    rew_matrix[row_index][col_index]=-2

        #plt.figure()
        
        #edges_color = self.color_edges(self.graph)
        #nx.draw(self.graph,node_color=color,with_labels=True,edge_color=edges_color)
        #plt.savefig("new_graph.png")

        return rew_matrix, array_states.index(source_state), array_states.index(sink_state)

    def optimize(self,rew_matrix,source_state,sink_state,cycle):
        
        #Extra computational time from making multiple adjacency matrices?
        adj_matrix = np.array(nx.attr_matrix(self.graph,rc_order=self.graph.nodes))
        
        #create variables
        x=cvx.Variable(adj_matrix.shape)
        #create constraints, x is binary
        constraints = [x<=1,x>=0]
        
        #Could this be set to <= 0 for better performance?
        for row_index in range(len(adj_matrix)):
            for col_index in range(len(adj_matrix)): 
                if adj_matrix[row_index][col_index]==0:
                    #known to not exist edges are set to 0s
                    constraints += [x[row_index][col_index]==0]
                
                #constraint of only allowing directed edges
                #!!!!!!!!!!!!!
                #TODO: Fix, this one doesn't make sense.
                constraints += [x[row_index][col_index]+x[col_index][row_index]<=1]
       

        #Sum of inflow is the sum of outflow 
        vector = np.zeros(len(adj_matrix))
        vector[sink_state]=-1
        vector[source_state]=1
        constraints += [cvx.sum(x,axis=0)-cvx.sum(x,axis=1)==vector]
        
        #force only one  entry and only one exit 
        constraints += [cvx.sum(x,axis=0)<=1]
        constraints += [cvx.sum(x,axis=1)<=1]

        #Add constraints for found cycles:
        #Q: Can cycles be avoided with some constraints?
        
        for cycle_i in self.cycle: 
            index_nodes_cycle = [list(self.graph.nodes).index(node) for node in cycle_i]
            edges_cycle = [(index_nodes_cycle[i],index_nodes_cycle[i+1]) for i in range(len(index_nodes_cycle)-1)]
            edges_cycle.append((index_nodes_cycle[-1],index_nodes_cycle[0]))
            exp = 0
            for pair in edges_cycle:
                exp+=x[pair[0]][pair[1]]
            num=len(cycle_i)
            constraints += [exp<= num-1]

        #create obj_function 
        problem=cvx.Problem(cvx.Maximize(cvx.sum(cvx.multiply(rew_matrix,x))),constraints)
        
        #solve
        #for writing model//works when unimodular
        #problem.solve("CPLEX",cplex_filename="model.lp")
        #for solving MIP
        problem.solve("GLPK_MI")
        
        if problem.status == "infeasible":
            for cons in constraints:
                print(cons.dual_value)
                return
        else:
            self.value = problem.value
            new_pairs=[]
            for row_index in range(len(x.value)):
                for col_index in range(len(x.value)):
                    if x.value[row_index][col_index]==1:
                        new_pairs.append((list(self.graph.nodes)[row_index],list(self.graph.nodes)[col_index]))
            g=nx.DiGraph()
            g.add_edges_from(new_pairs)
            print("solution_edges",g.edges)
            
            #find cycles:
            cycles=sorted(nx.simple_cycles(g))
            print("found cycles",cycles)
            
            if len(cycles)>0:
                for cycle in cycles:
                    print(cycle)
                    
                    if cycle not in self.cycle:
                        self.cycle.append(cycle)
                        print("added cycle")
                print("found cycles:",self.cycle)
                return list(g.edges),cycles
            
            else: 
                '''
                nodes = g.nodes
                new_colors = []
                color = self.color_nodes(g)
                
                color[list(g.nodes).index(list(self.graph.nodes)[source_state])]="green"
                color[list(g.nodes).index(list(self.graph.nodes)[sink_state])]="brown"
                edges_color = self.color_edges(g)
                for node in nodes:
                    new_list=(list(self.graph.nodes))
                    new_colors.append(color[new_list.index(node)])

                '''
                plt.figure(figsize=(20,10))
                nx.draw(g,with_labels=True)
                plt.show()
                print("source state",list(self.graph.nodes)[source_state])
                print("sink state",list(self.graph.nodes)[sink_state])
                return list(g.edges),cycles


            


