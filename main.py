import numpy as np
import cvxpy as cvx
import random
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime
import sys
    
import smart_crossover 
import os



def create_sols(n_states,n_learners):

    number_states=n_states
    states=np.arange(1,number_states+1,1)
    rewards=np.random.randint(low=0,high=5,size=len(states))
    total_info = dict(zip(states,rewards))
    learned_info={}
    for i in range(n_learners):
        learned_info[i]={}
        number_states_visited=np.random.randint(low=2,high=n_states)
        visited_states=random.choices(states,k=number_states_visited)
        edges_exist = [(i,i+1) for i in range(len(visited_states)-1)]
        total_edges = [(i,j) for i in visited_states for j in visited_states]
        edges_unknown = []
        for edge in total_edges:
            if edge[0]==edge[1]:
                pass
            elif edge not in edges_exist:
                edges_unknown.append(edge)
        number_of_non_existing = np.random.randint(low=1,high=len(edges_unknown))
        edges_unknown = random.choices(edges_unknown,k=number_of_non_existing)
        rewards_obtained=[total_info[i] for i in visited_states]
        learned_info[i]["state-reward"]=dict(zip(visited_states,rewards_obtained))
        learned_info[i]["not_known"]=edges_unknown
    return learned_info

if __name__== "__main__":

    
    n_states=np.random.randint(low=20,high=25)
    n_learners=np.random.randint(low=2,high=n_states//3)
    learned_info=create_sols(n_states,n_learners)
    now = datetime.now().time()
    print(now)
    for learner in learned_info:
        print("value found by learner "+str(learner), sum(list(learned_info[learner]['state-reward'].values())))
    
    crossover=smart_crossover.Smart_Crossover(learned_info)
    after = datetime.now().time()
    print("time without plots")
    print("number of states: ",n_states)
    print("n_iter:",crossover.n_iter)
    print("solution:",crossover.solution)
    print("value:",crossover.value)
    print(after)