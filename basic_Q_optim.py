#Author: Kyle Norland, using code from Sam
#Date: 12-20-22
#Purpose: Get back to very basic Q-Learner


#----------------------------------------------------
#----------------IMPORTS-----------------------------
#----------------------------------------------------
#Ignore future warnings
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

#General Imports
import numpy as np
rng = np.random.default_rng(12345)
np.set_printoptions(suppress = True)
import copy
import json
from sklearn.preprocessing import normalize
import random
import time

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

#Visualizer
import output_processor



#-------------------------
#--Default Run Settings---
#-------------------------
run = {}
#run['env'] = 'FrozenLake-v1'
run['env'] = 'vector_grid_goal'
run['n_episodes'] = 5000
run['ga_frequency'] = 100
run['crossover_chance'] = 1
run['mutation_prob'] = 0.2 
run['python_seed'] = 12345
run['np_seed'] = 12345
run['env_seed'] = 12345
run['grid_dims'] = (5,5)
run['player_location'] = (0,0)
run['goal_location'] = (4,4)
run['map'] = np.array([[0,0,0,0,0],
                        [0,1,0,0,0],
                        [0,0,0,1,0],
                        [0,1,0,0,1],
                        [0,0,0,0,0]])
run['max_iter_episode'] = 10      #maximum of iteration per episode
run['exploration_proba'] = 1    #initialize the exploration probability to 1
run['exploration_decreasing_decay'] = 0.0005       #exploration decreasing decay for exponential decreasing
run['min_exploration_proba'] = 0.01
run['gamma'] = 0.90            #discounted factor
run['lr'] = 0.1                 #learning rate 
run['pop_size'] = 1

run['output_dict'] = {}

#Env Config
run['env_config'] = {}
#run['env_config']['env_name'] = 'FrozenLake-v1'
run['env_config']['env_name'] = run['env']
#run['env_config']['env_name']


#Make environment
env = vector_grid_goal.CustomEnv(grid_dims=run['grid_dims'], player_location=run['player_location'], goal_location=run['goal_location'], map=run['map'])


#Output Path
run['output_path'] = 'GA_output'
if not os.path.exists(run['output_path']):
    os.makedirs(run['output_path'], exist_ok = True)    

#Generate or Load Experiment
folder_mode = False
generate_mode = True

experiment = {'runs': []}

#---------------Set Seeds--------------------
#Start Timer
print("Run")
print(run)
run['run_start_time'] = time.time()

#Set random seeds
#random.seed(run['python_seed'])
#np.random.seed(run['np_seed'])

#Env seed for action space
#env.action_space.np_random.seed(run['env_seed'])        

#Grab observation and action space for initializations
n_observations = env.observation_space.n
n_actions = env.action_space.n


#-----------------------------------
#----------Make Q_learner-----------
#-----------------------------------
model_dict = {}
for i in range(run['pop_size']):
    model_dict[i] = {}
    #Copy some of the run parameters for each member of the population.
    for entry in ['max_iter_episode', 'exploration_proba', 'exploration_decreasing_decay', 'min_exploration_proba', 'gamma', 'lr']:
        model_dict[i][entry] = run[entry]
    '''
    model_dict[i]['max_iter_episode'] = 10  #maximum of iteration per episode
    model_dict[i]['exploration_proba'] = 0.5   #initialize the exploration probability to 1
    model_dict[i]['exploration_decreasing_decay'] = 0.001    #exploartion decreasing decay for exponential decreasing
    model_dict[i]['min_exploration_proba'] = 0.01    # minimum of exploration proba
    model_dict[i]['gamma'] = 0.99            #discounted factor
    model_dict[i]['lr'] = 0.1                 #learning rate
    '''
    #Initialize Q table
    model_dict[i]['Q_table'] = np.zeros((n_observations,n_actions))
    #model_dict[i]['total_rewards_episode'] = list()

    #Extra Information Saved
    model_dict[i]['sa_info'] = [list([{} for x in range(n_actions)]) for x in range(n_observations)]
    model_dict[i]['transition_bank'] = []
    
    #Last run information
    model_dict[i]['prev_run'] = []
    
    #Outputs
    model_dict[i]['rewards_per_episode'] = []
    model_dict[i]['plotted_rewards'] = []
            
#Just grab the first one as the model
model = model_dict[0]

#---------------------------------------------
#--------------Training Loop------------------
#---------------------------------------------

for e in range(run['n_episodes']):
    #we initialize the first state of the episode
    model['current_state'] = env.reset()
    model['done'] = False
    
    first_state=True
    
    #sum the rewards that the agent gets from the environment
    model['total_episode_reward'] = 0
    
    #Run the episode
    
    #print("The model is: ", model)
    model['prev_run'] = []
    model['prev_first'] = 0
    model['prev_last'] = 0
    
    
    
    for i in range(model['max_iter_episode']): 
        #--------------------------------------
        #---------Choose Action----------------
        #---------------------------------------
        #Epsilon random action decision
        if np.random.uniform(0,1) < model['exploration_proba']:
            #model['action'] = env.action_space.sample()
            model['action'] = env.action_space.sample()
            
        else:
            model['action'] = np.argmax(model['Q_table'][model['current_state'],:])
        
        #--------------Run action---------------------
        #print('S', model['current_state'], ' A', model['action'])
        model['next_state'], model['reward'], model['done'], _ = env.step(model['action'])
        
        #--------------------------------------------------
        #---------------Record Transition-----------------
        #--------------------------------------------------
        nr = {}
        nr['from_state'] = int(model['current_state'])
        nr['action'] = model['action']
        nr['to_state'] = int(model['next_state'])
        nr['reward'] = model['reward']
        nr['done'] = model['done']
        
        #print("The model is: ", model)
        model['transition_bank'].append(copy.deepcopy(nr))
        
        #Also record the state,action pair
        
        model['prev_run'].append((int(model['current_state']), model['action']))
        if first_state: 
            model['prev_first'] = int(model['current_state'])
            first_state = False
            
        if i == model['max_iter_episode'] - 1:
            model['prev_last'] = int(model['next_state'])
        
        
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
            
        #Update state
        model['current_state'] = model['next_state']
    
    #-----------------------------------------------------------------
    #----------------End Of Episode Actions---------------------------
    #-----------------------------------------------------------------
    #Decay Exploration Probability
    model['exploration_proba'] = max(model['min_exploration_proba'], np.exp(-model['exploration_decreasing_decay']*e))
    
    #Update total rewards
    model['rewards_per_episode'].append(model['total_episode_reward'])
    
    #-----------------------------------------------------------
    #----------------------Convert Transitions into SA knowledge Table--------------------
    #------------------------------------------------------------
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
    
    #Print transitions from episode
    #if e % 100 == 0:
    
    #print("Transitions")
    #print(model['transition_bank'])
        
    #-----------Clear transition bank after information has been added-------------------------
    model['transition_bank'] = []
    #print("Size of transition_bank: ", len(model['transition_bank']))
    


    #-------------------------------------------------------------
    #----------Print Population Performance every # of episodes---
    #-------------------------------------------------------------
    if e % 50 == 0:
        print("-"*40)
        print("Episode:", e, "Rewards: ")        
        for key, value in model_dict.items():
            model = value
            #Print reward
            print(key, ":", model['total_episode_reward'])
            #Print Q-Table
            print("Q-Table:")
            print(model['Q_table'])
            
            #Print the greedy path through
            max_indices = []
            for row in model['Q_table']:
                #Get max index
                max_indices.append(row.argmax())
            
            direction_mapping = {0:'L', 1:'U', 2:'R', 3:'D'}
            max_indices = [direction_mapping[x] for x in max_indices]
            max_indices = np.array(max_indices)
            print("reformed")
            print(max_indices.reshape(run['grid_dims'][0], run['grid_dims'][1]))
            
            
        print("-"*40)       
    
    #--------------------If on a solving episode-----------------
    if e % run['ga_frequency'] == 0 and e!= 0 and random.random() < run['crossover_chance']:
        #-----------Convert info-----------
        #Initialize learned info structure
        learned_info = {}
        
        #Convert the known data to algorithm compatible forms
        for j, model in enumerate([model]):
            learned_info[j] = {}
            learned_info[j]['state-reward'] = {}
            learned_info[j]['known_states'] = []
            learned_info[j]['not_known'] = []
            learned_info[j]['edges_exist'] = []
            
            #Manually add in the starting spot as zero.
            start_index = int((run['player_location'][1] * run['grid_dims'][0]) + run['player_location'][0])
            print("Start index: ", start_index)
            learned_info[j]['state-reward'][start_index] = 0
            
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
                        
                        #Add knownledge of the previous run
                        learned_info[j]['prev_run'] = model['prev_run']
                        learned_info[j]['prev_first'] = model['prev_first']
                        learned_info[j]['prev_last'] = model['prev_last']

            #Deal with not known edges    
            #Remove duplicate edges            
            learned_info[j]['edges_exist'] = list(dict.fromkeys(learned_info[j]['edges_exist'])) 
            
            learned_info[j]['known_states'] = list(set(learned_info[j]['known_states']))
            
            learned_info[j]['not_known'] = [(from_state, to_state) for from_state in learned_info[j]['known_states'] for to_state in learned_info[j]['known_states'] if (from_state, to_state) not in learned_info[j]['edges_exist']]
        

        #----------------------------------
        #----------Run Solver--------------
        #----------------------------------
        #Start timer to time solver
        start = datetime.now()
        
        #Run the crossover
        crossover=smart_crossover.Smart_Crossover(learned_info)
        
        print("Solution time: ", datetime.now() - start)

        print("Solution:",crossover.solution)
        #print("Objective Value:",crossover.value)
        
        #--------------------------------------------------
        #--------------Update the Individual---------------
        #--------------------------------------------------
        
        #Currently replacing in place.
        print("-"*10, "Making new individuals", "-"*10)
        solution_path = crossover.solution
        print("Solution path", solution_path)
        #Get the ensemble information
        ensemble = crossover.ensemble
        #print("Edges", ensemble['edges'])
        
        
        #For known edges, get the known transition
        sa_path = []
        for edge in solution_path:
            if edge in ensemble['edges']:
                new_sa_pair = (-1,-1)
                print("Edge", edge, "known")
                #Search the model
                for index, sa_pair in enumerate(model['sa_info'][edge[0]]):
                    if len(sa_pair) > 0:
                        #print("SA Pair", sa_pair)
                        if sa_pair['to_state'] == edge[1]:
                            #print("Found matching sa pair", sa_pair)
                            new_sa_pair = (edge[0], index)
                            break
                sa_path.append(new_sa_pair)           
                        
            else:
                print("Edge", edge, "unknown")
                sa_path.append((edge[0], env.action_space.sample()))
                break
        
        print("SA_Path")
        print(sa_path)
        
        #For each individual, set the path sa pairs to 1.5 times the highest Q value.
        parent = model
        #Calculate the highest Q value
        max_Q = max(max(x) for x in parent['Q_table'])
        #print("Max Q: ", max_Q)
        print("Pre Q table")
        print(parent['Q_table'])
        
        #Direction_mapping
        direction_mapping = {0:'L', 1:'U', 2:'R', 3:'D'}
                            
        print("Pre-Updated")
        #Print the greedy path through
        max_indices = []
        for row in parent['Q_table']:
            #Get max index
            max_indices.append(row.argmax())
        max_indices = [direction_mapping[x] for x in max_indices]
        max_indices = np.array(max_indices)
        print(max_indices.reshape(run['grid_dims'][0], run['grid_dims'][1]))
        
        
        for sa_pair in sa_path:
            parent['Q_table'][sa_pair[0], sa_pair[1]] = max(max_Q * 1.2, 1)
        
        #Normalize matrix
        normalize(parent['Q_table'], axis=1, norm='l1')
        normalize(parent['Q_table'], axis=0, norm='l1')
        
        print("Post Q table")
        print(parent['Q_table'])
        print()
        
        #Show the updated path
        print("Updated")
        #Print the greedy path through
        max_indices = []
        for row in parent['Q_table']:
            #Get max index
            max_indices.append(row.argmax())
        max_indices = [direction_mapping[x] for x in max_indices]
        max_indices = np.array(max_indices)
        print(max_indices.reshape(run['grid_dims'][0], run['grid_dims'][1]))


#--------------------------------------------------------
#------------------Final Outputs-------------------------
#--------------------------------------------------------

json_output = {'model_rewards': []}

#Print out the outputs
for key in model_dict.keys():
    model = model_dict[key]     #copy by reference
    
    #Save rewards to json
    json_output['model_rewards'].append(model['rewards_per_episode'])
    
    #Graph rewards
    print("Rewards", model['rewards_per_episode'])
    running_avg = []
    window = 10
    for point in range(window, len(model['rewards_per_episode'])):
        running_avg.append(np.mean(model['rewards_per_episode'][point-window: point]))
            
    plt.plot(running_avg)

plt.title("Rewards per individual")
plt.show()
#plt.savefig(os.path.join(output_folder,"Rewards.png"))

'''
#Save json to file
with open(os.path.join(output_folder, "crossover.json"), 'w') as f:
    json.dump(json_output, f)
'''
#-------------------------------------------------------
#---------------Print final models----------------------
#-------------------------------------------------------
print()
print("Current Models")
for model_num, model in model_dict.items():
    print("Model: ", model_num)
    print(model['Q_table'])
