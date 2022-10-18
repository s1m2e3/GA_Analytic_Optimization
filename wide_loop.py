#Kyle Norland, using code from Sam
#10/10/22


#----------------------------------------------------
#----------------IMPORTS-----------------------------
#----------------------------------------------------
#Ignore future warnings
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

#General Imports
import numpy as np
import copy

import random

from datetime import datetime
import sys
import os

#Visualization
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

#Solver (also requires cvopt)
import cvxpy as cvx

#Graph Handling
import networkx as nx

#Custom Crossover Library
import smart_crossover 

#Support for RL environment.
import gym
import vector_grid_goal #Custom environment
#-------------------------------------------
#--------------Functions--------------------
#-------------------------------------------
def test_individual(individual):
    #n_iterations
    pass

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
    
    #----------------------------------------------------------------
    #--------------------------Globals-------------------------------
    #----------------------------------------------------------------
    n_episodes = 1
    ga_frequency = 1   #How often the GA algorithm runs. May want to add in a parameter concerning the age of each model.
    
    #Seed randoms
    random.seed(1234)
    np.random.seed(1234)
    #env = gym.make('FrozenLake-v1')    #If you want to try with frozen lake.
    #env.seed(0)
    
    #---------------------------------------------------------------------
    #--------------------Initialize Environment---------------------------
    #---------------------------------------------------------------------
    grid_dims = (5,5)
    player_location = (0,0)
    goal_location = (4,4)
    
    map = np.array([[0,1,1,1,0],
                    [0,0,1,0,0],
                    [0,0,0,0,0],
                    [0,1,0,0,0],
                    [0,0,0,0,0]])
    
    #Deterministic switch for action environment.
    env = vector_grid_goal.CustomEnv(grid_dims=grid_dims, player_location=player_location, goal_location=goal_location, map=map)
    env.action_space.np_random.seed(1)
    
    n_observations = env.observation_space.n
    n_actions = env.action_space.n
    
    #---------------------------------------------------------------------
    #--------------------Initialize Population---------------------------
    #--------------------------------------------------------------------
    #Generate population
    model_dict = {}
    for i in range(2):
        model_dict[i] = {}
        #model_dict[i]['n_episodes'] = 20000      #number of episode we will run
        model_dict[i]['max_iter_episode'] = 10  #maximum of iteration per episode
        model_dict[i]['exploration_proba'] = 1   #initialize the exploration probability to 1
        model_dict[i]['exploration_decreasing_decay'] = 0.001    #exploartion decreasing decay for exponential decreasing
        model_dict[i]['min_exploration_proba'] = 0.01    # minimum of exploration proba
        model_dict[i]['gamma'] = 0.99            #discounted factor
        model_dict[i]['lr'] = 0.1                 #learning rate
        model_dict[i]['Q_table'] = np.zeros((n_observations,n_actions))
        model_dict[i]['total_rewards_episode'] = list()
        model_dict[i]['rewards_per_episode'] = []
        model_dict[i]['plotted_rewards'] = []
        
        #Keeping track for crossover model.
        model_dict[i]['sa_info'] = [list([{} for x in range(n_actions)]) for x in range(n_observations)]
        model_dict[i]['transition_bank'] = []
        '''
        model_dict[i]['static_transitions'] = []
        model_dict[i]['known_rewards'] = np.full(n_observations,-1)
        model_dict[i]['known_transitions'] = np.full((n_observations, n_observations), 0)
        model_dict[i]['known_not_transitions'] = np.full((n_observations, n_observations), 0)
        '''
    #---------------------------------------------------------
    #------------------Initialize Animation-------------------
    #---------------------------------------------------------
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)

    def animate(i):
        ax1.clear()
        for key in model_dict.keys():
            ax1.plot(model_dict[key]['plotted_rewards'])
        #for reward_set in plotted_rewards:
            #ax1.plot(reward_set)
        plt.draw()
        plt.pause(0.001)

    plt.ion()
    plt.show()
    
    

    #Core and added actions
    #Left -0
    #Down = 1
    #Right = 2
    #Up = 3

    #----------------------------------------------------------------
    #-------------------Main Training Loop---------------------------
    #----------------------------------------------------------------
    #For number of episodes
    for e in range(n_episodes):
        
        #For each model
        for key in model_dict.keys():
            model = model_dict[key]     #copy by reference
            #we initialize the first state of the episode
            model['current_state'] = env.reset()
            model['done'] = False
            
            #sum the rewards that the agent gets from the environment
            model['total_episode_reward'] = 0
            
            #-------------------------------------------------------
            #--------------Individual Training/Evaluation-----------
            #-------------------------------------------------------
            
            for i in range(model['max_iter_episode']): 
                
                #Epsilon random action decision
                if np.random.uniform(0,1) < model['exploration_proba']:
                    #model['action'] = env.action_space.sample()
                    model['action'] = env.action_space.sample()
                    
                else:
                    model['action'] = np.argmax(model['Q_table'][model['current_state'],:])
                
                #Run action
                model['next_state'], model['reward'], model['done'], _ = env.step(model['action'])
                
                #---------------Record Transitions-----------------
                nr = {}
                nr['from_state'] = model['current_state']
                nr['action'] = model['action']
                nr['to_state'] = int(model['next_state'])
                nr['reward'] = model['reward']
                nr['done'] = model['done']
                
                #print("The model is: ", model)
                model['transition_bank'].append(copy.deepcopy(nr))
                
                
                
                #-----------------------------------------------
                #-----------Q-Table update----------------------
                #-----------------------------------------------                
                # We update our Q-table using the Q-learning iteration
                model['Q_table'][model['current_state'], model['action']] = (1-model['lr']) * model['Q_table'][model['current_state'], model['action']] + model['lr']*(model['reward'] + model['gamma']*max(model['Q_table'][model['next_state'],:]))
                
                #Update reward
                model['total_episode_reward'] = model['total_episode_reward'] + model['reward']
                
                # If the episode is finished, we leave the for loop
                if model['done']:
                    break
                    
                #Update state record
                model['current_state'] = model['next_state']
            
            #Update parameters that adjust on episode.
            #We update the exploration proba using exponential decay formula 
            model['exploration_proba'] = max(model['min_exploration_proba'], np.exp(-model['exploration_decreasing_decay']*e))
            model['rewards_per_episode'].append(model['total_episode_reward'])
            
            #Update/visualize parameters
            #Prepare partial outputs
            if e % 1000 == 0 and e != 0:
                print("Avg Reward at", e, "for key:", key, "is", np.mean(model['rewards_per_episode'][e-1000:e]))
                model['plotted_rewards'].append(np.mean(model['rewards_per_episode'][e-1000:e]))
                
                #Reanimate when new performance numbers are in
                animate(5)
                
            if e == n_episodes-1:
                print("Done")
                
            #Update the models memory tables
            #for record in model['transition_bank']:
                #print(record)
                
            #Update failed transitions
            #print()
            #print("Static transitions")
            
            #Convert to sa_info
            #print("sa_info")
            #print(model['sa_info'])
            for record in model['transition_bank']:
                i1 = record['from_state']
                i2 = record['action']
                #print(i1, i2)
                #print(model['sa_info'][i1][i2])
                #print(model['sa_info'][record['from_state'][record['action']])
                model['sa_info'][i1][i2] = {'to_state': int(record['to_state']), 'reward': record['reward'], 'done': record['done']}
                #print("hi")
                #{'to_state': record['to_state'], 'reward': record['reward'], 'done': record['done']}
            #print('sa_info')
            #print(model['sa_info'])
            
            sa_table = []
            for j in range(len(model['sa_info'])):
                row = []
                for k in range(len(model['sa_info'][0])):
                    if len(model['sa_info'][j][k]) > 0:
                        row.append(int(model['sa_info'][j][k]['to_state']))
                    else:
                        row.append('*')
                sa_table.append(row)
            
            '''
            print('sa_table')
            for entry in sa_table:
                print(entry)
            '''
            
            '''
            for record in model['transition_bank']:
                if record['from_state'] == record['to_state']:
                    #print(record)
                    model['static_transitions'].append(record)
            
            for record in model['transition_bank']:
                #Update reward table
                model['known_rewards'][record['to_state']] = record['reward']
            
                #Update known transitions to include states from the memories.
                model['known_transitions'][int(record['from_state'])][int(record['to_state'])] = 1
                
            #Update transitions that did not work. (Note: Only works if deterministic)
            #print("Figuring non transitions")
            #Set all first then reset nons.
            model['known_not_transitions'][model['known_transitions'] != 1] = 1
            for i in range(len(model['known_transitions'])):
                if sum(model['known_transitions'][i]) < n_actions:
                    model['known_not_transitions'][i] = 0
                            
            #Clear out transition bank from the episode:
            '''
            print("Transition bank")
            for transition in model['transition_bank']:
                print(int(transition['from_state']), int(transition['to_state']))
            model['transition_bank'] = []

        #-------------------------------------------------------------
        #--------------------GA Procedure-----------------------------
        #-------------------------------------------------------------
        #If time to run the ga
        if e % ga_frequency == 0: # and e != 0:
            
            mutprb = 0.05   #Mutation probability
            cxprb = 1    #Crossover probability
            
            #Different ways of selecting which individuals to crossover. But can do crossover based on sequential.
            #Currently replacing both parents with children.
            #Dictionary is i indexed
            for i in range(1, len(model_dict)):
                #Apply crossover if less than crossover probability
                if random.random() < cxprb:
                    #Format models correctly and enter into the crossover function.
                    parent_1 = model_dict[i]
                    parent_2 = model_dict[i-1]
                    
                    #Initialize learned info structure
                    learned_info = {}
                    
                    #Convert them to algorithm compatible forms
                    for j, model in enumerate([parent_1, parent_2]):
                        learned_info[j] = {}
                        learned_info[j]['state-reward'] = {}
                        learned_info[j]['known_states'] = []
                        learned_info[j]['not_known'] = []
                        learned_info[j]['edges_exist'] = []
                        
                        for k in range(0,n_observations):
                            existing_edges = []
                            not_known_tos = [True] * n_observations
                            known_edges_counter = 0
                            
                            for m in range(0, n_actions):
                                record = model['sa_info'][k][m]
                                if len(record) > 0:
                                    known_edges_counter += 1
                                    #Update reward record
                                    learned_info[j]['state-reward'][record['to_state']] = record['reward']
                                    
                                    #Update known states
                                    learned_info[j]['known_states'].append(k)
                                    learned_info[j]['known_states'].append(record['to_state'])
                                    
                                    
                                    #Update edges exist
                                    existing_edges.append((k,record['to_state']))
                                    
                                    not_known_tos[record['to_state']] = False

                                    learned_info[j]['edges_exist'].append((k,record['to_state']))
                        
            
            
            
                        #Deal with not known edges    
                        #Remove duplicate edges            
                        learned_info[j]['edges_exist'] = list(dict.fromkeys(learned_info[j]['edges_exist'])) 
                        
                        learned_info[j]['known_states'] = list(set(learned_info[j]['known_states']))
                        
                        learned_info[j]['not_known'] = [(from_state, to_state) for from_state in learned_info[j]['known_states'] for to_state in learned_info[j]['known_states'] if (from_state, to_state) not in learned_info[j]['edges_exist']]

                        '''
                        #Add not_known edges
                        if known_edges_counter >= n_actions:
                            pass
                        else:
                            print("Not known edges")
                            
                            print([(k,to_state) for to_state in np.array(range(n_observations))[np.array(not_known_tos)]])
                            learned_info[j]['not_known'].extend([(k,to_state) for to_state in np.array(range(n_observations))[not_known_tos]])
                        '''       

                        
                        
                        print("Input", j)
                        print(learned_info[j]['known_states'])
                        print(learned_info[j]['edges_exist'])
                        print(learned_info[j]['not_known'])
                        print()
                        print()
                        
                        
                                        
                                        
                    
                    #print("Learned info")
                    #print(learned_info[0])
                    
                    #To avoid looping more
                    #break
                    
                    #Check crossover code for direct inserting known edges.
                    start = datetime.now()
                    #edges_exist
                    #player_location
                    #goal_location
                    
                    print("Pre learner")
                    print(learned_info)
                    crossover=smart_crossover.Smart_Crossover(learned_info)
                    
                    print("Source state", crossover.source_state)
                    print("Sink state", crossover.sink_state)
                    
                    print("Solution time: ", datetime.now() - start)
                    
                    #Crossover stats
                    print("time without plots")
                    #print("number of states: ",n_states)
                    print("n_iter:",crossover.n_iter)
                    print("solution:",crossover.solution)
                    print("value:",crossover.value)
                    

    #-----------------------------------------------------------------------
    #------------------------End of Experiment Outputs----------------------
    #-----------------------------------------------------------------------
    '''
    #Print out the outputs
    for key in model_dict.keys():
        model = model_dict[key]     #copy by reference
        print('Known rewards')
        print(model['known_rewards'])
        
        print('Known transitions')
        #print(model['known_transitions'])
        for line in model['known_transitions']:
            print(line)
        
        print('Known not transitions')
        print(model['known_not_transitions'])       
    '''
