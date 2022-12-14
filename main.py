#Sam, modified by Kyle Norland
#Modified 10/10/22

import numpy as np
import cvxpy as cvx
import random
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime
import sys
    
import smart_crossover as smart_crossover
import os



def create_sols(n_states,n_learners):

    number_states=n_states
    #States and rewards are
    states=np.arange(1,number_states+1,1)
    rewards=np.random.randint(low=0,high=5,size=len(states))
    
    #Dictionary with states as key, rewards as value.
    total_info = dict(zip(states,rewards))
    
    #Learner dictionary with index for key.
    learned_info={}
    for i in range(n_learners):
        learned_info[i]={}
        
        #Randomly generated number of states visited with random number and then generated)
        number_states_visited=np.random.randint(low=2,high=n_states)
        
        #Array of visited state numbers
        visited_states=random.choices(states,k=number_states_visited)
        
        #Edges exist from the known states to the next one?
        edges_exist = [(i,i+1) for i in range(len(visited_states)-1)]
        
        #All possible edges between visited states.
        total_edges = [(i,j) for i in visited_states for j in visited_states]
        edges_unknown = []
        
        for edge in total_edges:
            #Ignore if i and j are the same.
            if edge[0]==edge[1]:
                pass
            #Append to edges unknown if not known to exist.
            elif edge not in edges_exist:
                edges_unknown.append(edge)
        
        #Generate some non-existing edges?
        number_of_non_existing = np.random.randint(low=1,high=len(edges_unknown))
        
        #Select some random edges unknown.
        edges_unknown = random.choices(edges_unknown,k=number_of_non_existing)
        
        #Get rewards for states visited.
        rewards_obtained=[total_info[i] for i in visited_states]
        
        #Visited state reward knowledge
        learned_info[i]["state-reward"]=dict(zip(visited_states,rewards_obtained))
        
        #Edges unknown
        learned_info[i]["not_known"]=edges_unknown
        
        #Return sets of state-reward and not-known
    return learned_info

if __name__== "__main__":

    #Generate a random number of states and learners
    n_states=np.random.randint(low=20,high=25)
    #n_learners=np.random.randint(low=2,high=n_states//3)
    n_learners=np.random.randint(low=2,high=3)
    
    #Create assumed learned info (Unecessary once replaced.
    learned_info=create_sols(n_states,n_learners)
    #print("Learned info")
    #for learner in learned_info:
        #print(learner)
    
    #Start timer
    now = datetime.now().time()
    print(now)
    
    for learner in learned_info:
        print("value found by learner "+str(learner), sum(list(learned_info[learner]['state-reward'].values())))
    
    #Apply the crossover to the dictionary of agents.
    #print("Learned info")
    #print(learned_info[0])
    crossover=smart_crossover.Smart_Crossover(learned_info)
    
    #End timer
    after = datetime.now().time()
    
    print("time without plots")
    print("Source state", crossover.source_state)
    print("Sink state", crossover.sink_state)
    print("number of states: ",n_states)
    print("n_iter:",crossover.n_iter)
    print("solution:",crossover.solution)
    print("value:",crossover.value)
    print(after)