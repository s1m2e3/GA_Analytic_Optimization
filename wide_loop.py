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
np.set_printoptions(suppress = True)
import copy
import json
from sklearn.preprocessing import normalize
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

#------------------------------------------
#-------------Globals----------------------
#------------------------------------------
input_folder = "none"
output_folder = "output_stage_1"


#-------------------------------------------
#--------------Functions--------------------
#-------------------------------------------
def evaluate_Q_individual(individual, e):
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
        pass
        #print("Done")
        
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

def convert_to_learned_info(parent_1, parent_2):
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
    
    return learned_info

def create_new_Q_individuals(parent_1, parent_2, crossover):
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
            #Search the parents for the correct action
            for parent in [parent_1, parent_2]:
                for index, sa_pair in enumerate(parent['sa_info'][edge[0]]):
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
    for parent in [parent_1, parent_2]:
        #Calculate the highest Q value
        max_Q = max(max(x) for x in parent['Q_table'])
        #print("Max Q: ", max_Q)
        #print("Pre Q table")
        #print(parent['Q_table'])
        for sa_pair in sa_path:
            parent['Q_table'][sa_pair[0], sa_pair[1]] = max(max_Q * 1.2, 1)
        
        #Normalize matrix
        normalize(parent['Q_table'], axis=1, norm='l1')
        normalize(parent['Q_table'], axis=0, norm='l1')
        
        #print("Post Q table")
        #print(parent['Q_table'])
        print()

        
#-------------------------------------------
#------------------Main---------------------
#-------------------------------------------

if __name__== "__main__":
    
    #----------------------------------------------------------------
    #--------------------------Global Parameters-------------------------------
    #----------------------------------------------------------------
    n_episodes = 2001
    ga_frequency = 200   #How often the GA algorithm runs. May want to add in a parameter concerning the age of each model.
    crossover_chance = 1
    
    #Seed randoms
    random.seed(1234)
    np.random.seed(1234)

    
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
    env = vector_grid_goal.CustomEnv(grid_dims=grid_dims, player_location=player_location, goal_location=goal_location, map=map)
    
    #Make action space deterministic
    env.action_space.seed(2)

    n_observations = env.observation_space.n
    n_actions = env.action_space.n
    
    #Core and added actions
    #Left: 0
    #Down = 1
    #Right = 2
    #Up = 3
    
    #---------------------------------------------------------------------
    #--------------------Initialize Population---------------------------
    #--------------------------------------------------------------------
    #Generate population
    model_dict = {}
    for i in range(2):
        model_dict[i] = {}
        #model_dict[i]['n_episodes'] = 20000      #number of episode we will run
        model_dict[i]['max_iter_episode'] = 10  #maximum of iteration per episode
        model_dict[i]['exploration_proba'] = 0.5   #initialize the exploration probability to 1
        model_dict[i]['exploration_decreasing_decay'] = 0.001    #exploartion decreasing decay for exponential decreasing
        model_dict[i]['min_exploration_proba'] = 0.01    # minimum of exploration proba
        model_dict[i]['gamma'] = 0.99            #discounted factor
        model_dict[i]['lr'] = 0.1                 #learning rate
        model_dict[i]['Q_table'] = np.zeros((n_observations,n_actions))
        #model_dict[i]['total_rewards_episode'] = list()
        model_dict[i]['rewards_per_episode'] = []
        model_dict[i]['plotted_rewards'] = []
        
        #Keeping track for crossover model.
        model_dict[i]['sa_info'] = [list([{} for x in range(n_actions)]) for x in range(n_observations)]
        model_dict[i]['transition_bank'] = []

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

    #----------------------------------------------------------------
    #-------------------Training Loop--------------------------------
    #----------------------------------------------------------------
    #For number of episodes
    for e in range(n_episodes):
        
        #For each model
        for key in model_dict.keys():
            model = model_dict[key]     #copy by reference
            
            #Evaluate the individual
            evaluate_Q_individual(model, e)
            
        #Print the rewards gathered by each agent in the population
        print("-"*40)
        print("Episode:", e, "Rewards: ")        
        for key in model_dict.keys():
            model = model_dict[key]
            #Print reward
            print(key, ":", model['total_episode_reward'])
        print("-"*40)            
            
        #-------------------------------------------------------------
        #--------------------GA Procedure-----------------------------
        #-------------------------------------------------------------
        #If time to run the ga
        if e % ga_frequency == 0: # and e != 0:
            
            mutprb = 0.05   #Mutation probability
            cxprb = crossover_chance    #Crossover probability
            
            #Different ways of selecting which individuals to crossover. But can do crossover based on sequential.
            #Currently replacing both parents with children.
            #Dictionary is i indexed
            for i in range(1, len(model_dict)):
                #Apply crossover if less than crossover probability
                if random.random() < cxprb:
                    #Format models correctly and enter into the crossover function.
                    parent_1 = model_dict[i]
                    parent_2 = model_dict[i-1]
                    
                    #Prep for Solver
                    learned_info = convert_to_learned_info(parent_1, parent_2)
                    
                    #---------------------------------
                    #----------Run Solver-------------
                    #---------------------------------
                    #Start timer to time solver
                    start = datetime.now()
                    
                    #print("Pre learner")
                    #print(learned_info)
                    crossover=smart_crossover.Smart_Crossover(learned_info)
                    #print("Source state", crossover.source_state)
                    #print("Sink state", crossover.sink_state)
                    
                    print("Solution time: ", datetime.now() - start)
                    
                    #print("number of states: ",n_states)
                    #print("n_iter:",crossover.n_iter)
                    print("Solution:",crossover.solution)
                    print("Objective Value:",crossover.value)
                    
                    #Create new individuals
                    create_new_Q_individuals(parent_1, parent_2, crossover)

                                
        #---------------------------------------------------------------
        #------------------------New Generation-------------------------
        #---------------------------------------------------------------
        '''
        print("Current Models")
        for model_num, model in model_dict.items():
            print("Model: ", model_num)
            for row_number, row in enumerate(model['Q_table']):
                print(row_number, ":", np.argmax(row))
            #print(model['Q_table'])
        '''         

    #-----------------------------------------------------------------------
    #------------------------End of Experiment Outputs----------------------
    #-----------------------------------------------------------------------
    #Save model rewards as json
    json_output = {'model_rewards': []}
    
    #Print out the outputs
    for key in model_dict.keys():
        model = model_dict[key]     #copy by reference
        
        #Save rewards to json
        json_output['model_rewards'].append(model['rewards_per_episode'])
        
        #Graph rewards
        print("Rewards", model['rewards_per_episode'])
        plt.plot(model['rewards_per_episode'])
    
    plt.title("Rewards per individual")
    #plt.show()
    plt.savefig(os.path.join(output_folder,"Rewards.png"))
    
    #Save json to file
    with open(os.path.join(output_folder, "crossover.json"), 'w') as f:
        json.dump(json_output, f)
    
    #-------------------------------------------------------
    #---------------Print final models----------------------
    #-------------------------------------------------------
    print()
    print("Current Models")
    for model_num, model in model_dict.items():
        print("Model: ", model_num)
        print(model['Q_table'])
