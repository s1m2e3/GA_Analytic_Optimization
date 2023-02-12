#Author: Kyle Norland, using code from Sam
#Date: 12-20-22
#Purpose: Get back to very basic Q-Learner


#----------------------------------------------------
#----------------IMPORTS-----------------------------
#----------------------------------------------------
import warnings #Ignore future warnings
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
import numpy as np
import graph_plotter    #Custom plotting library
import matplotlib.pyplot as plt
from matplotlib import interactive
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import time
import pygame
import sys
from pygame.locals import *
import math
import seaborn as sns
import cvxpy as cvx #Solver (also requires cvopt)
import networkx as nx #Graph Handling
import smart_crossover #Custom Crossover Library
import gym #Support for RL environment.
import vector_grid_goal #Custom environment
import output_processor #Visualizer
from datetime import datetime
import sys
import os

#--------------------------------------
#-------------Functions----------------
#--------------------------------------
def update_plot(input_array):
    global surf
    
    #Generate data
    rows = len(input_array)
    columns = len(input_array[0])
    X = np.arange(0, columns, 1)
    Y = np.arange(0, rows, 1)
    X, Y = np.meshgrid(X, Y)
    
    #Remove existing surface
    surf.remove()
    
    #Adjust the z lim
    ax.set_zlim(-1.01, np.amax(input_array))
    #Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
                           
    print("Showing plot")
    plt.draw()
    fig.canvas.flush_events()
    #time.sleep(1)

def q_to_path(env, Q_table, test_length=20):
    #Finds the greedy path through a q_table by running env
    
    #Restart environment
    current_state = env.reset()
    done = False
    path = []
    actions = []
    
    for i in range(test_length):
        action = np.argmax(Q_table[current_state,:])
        next_state, reward, done, _ = env.step(action)
    
        #Keep track of actions and states
        actions.append(action)
        path.append((int(current_state), int(next_state)))
        
        current_state = next_state
        
        #Break if done
        if done:
            break
            
    return path, actions

#-------------------------
#--Default Run Settings---
#-------------------------
run = {}
#run['env'] = 'FrozenLake-v1'
run['env'] = 'vector_grid_goal'
run['n_episodes'] = 4000
sns.set(style='darkgrid')
run['ga_frequency'] = 100
run['crossover_chance'] = 1
run['mutation_prob'] = 0.2 
run['python_seed'] = 12345
run['np_seed'] = 12345
run['env_seed'] = 12345

'''
run['map'] = np.array([[0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0],
                        [0,0,0,1,0,0,0],
                        [0,0,0,0,1,0,0],
                        [0,1,0,0,0,0,0],
                        [0,0,0,1,0,0,0],
                        [0,0,1,0,0,0,0]])
'''
'''
run['map'] = np.array([[0,0,0,0,0],
                        [0,1,0,0,0],
                        [0,0,0,1,0],
                        [0,0,0,0,1],
                        [0,0,1,0,0]])
'''
'''
run['map'] = np.array([[0,0,0,0,0],
                        [0,0,0,0,0],
                        [0,0,0,0,0],
                        [0,0,0,0,1],
                        [0,0,0,0,0]])
'''                        
run['map'] = np.array([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,1,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,1,0,0,0],
                        [0,1,0,0,0,0,0],
                        [0,0,0,0,0,0,1],
                        [0,0,0,0,0,0,0]])
                       
run['grid_dims'] = (len(run['map'][0]),len(run['map'][1]))
run['player_location'] = (0,0)
run['goal_location'] = (run['grid_dims'][0] - 1, run['grid_dims'][1] - 1)  
print(run['grid_dims'] ,  run['goal_location'])                  
                        
run['max_iter_episode'] = 30      #maximum of iteration per episode
run['exploration_proba'] = 1    #initialize the exploration probability to 1
run['exploration_decreasing_decay'] = 0.001       #exploration decreasing decay for exponential decreasing
run['min_exploration_proba'] = 0.001
run['gamma'] = 0.90            #discounted factor 0.9
run['lr'] = 0.1                 #learning rate 0.1
run['pop_size'] = 1
run['q_visuals'] = False
run['q_vis_frequency'] = 10
run['graph_visuals'] = True
run['screen_viz_frequency'] = 50

run['update_path_max_q'] = False
run['update_path_increment'] = False
run['policy_mixing'] = True
run['policy_uses'] = 500

run['path_q_increment'] = 10
run['random_endpoints'] = False
run['reward_endpoints'] = True


#-Visuals--
run['pause_on_visual'] = False

run["mip_flag"] = True
run["cycle_flag"] = False

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

#--------------------------------------------
#-------------INITIALIZE---------------------
#--------------------------------------------
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

#----------------------------------------------
#----------Make Q_learner/Population-----------
#----------------------------------------------
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
    
    #Mixing Policy Slots
    model_dict[i]['mixing_policy_list'] = []

    #Extra Information Saved
    model_dict[i]['sa_info'] = [list([{} for x in range(n_actions)]) for x in range(n_observations)]
    model_dict[i]['reward_array'] = {}
    model_dict[i]['transition_bank'] = []
    
    #Last run information
    model_dict[i]['prev_run'] = []
    
    #Outputs
    model_dict[i]['rewards_per_episode'] = []
    model_dict[i]['plotted_rewards'] = []
            
#Just grab the first one as the model
model = model_dict[0]

#---------------Visualizations-----------------------------
#Initialize the visualizations
if run['q_visuals']:
    #Put initial matplotlib setting
    interactive(True)

    #Generate initial plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    rows = len(model['Q_table'])
    columns = len(model['Q_table'][0])
    X = np.arange(0, columns, 1)
    Y = np.arange(0, rows, 1)
    X, Y = np.meshgrid(X, Y)
    Z = model['Q_table']
    #Z = np.random.random((5,6))

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(-1.01, 5.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.draw()
    fig.canvas.flush_events()

if run['graph_visuals']:
    #Set up pygame
    screen = graph_plotter.initialize_plot()

#---------------------------------------------
#--------------Training Loop------------------
#---------------------------------------------

for e in range(run['n_episodes']):
    #print("--------------Episode:", str(e), "-----------------")
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
            
        #-----------Policy Mixing----------------
        if run['policy_mixing']:
            #print("Mixing Policy")
            #Clean policy list of policies with 0 uses remaining.
            model['mixing_policy_list'] = [x for x in model['mixing_policy_list'] if x['uses_remaining'] > 0]
            
            #Check policy list for applicable policies and generate an action.
            for entry in model['mixing_policy_list']:  
                if int(model['current_state']) in entry['policy']:
                    #print("using mixed policy")
                    model['action'] = entry['policy'][int(model['current_state'])]
                    #Decrement the policy uses.
                    entry['uses_remaining'] -= 1
                    #print('uses_remaining', entry['uses_remaining'])
        
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
        #print("Pre-Value:", model['Q_table'][model['current_state'], model['action']])
        #old one: model['Q_table'][model['current_state'], model['action']] = (1-model['lr']) * model['Q_table'][model['current_state'], model['action']] + model['lr']*(model['reward'] + model['gamma']*max(model['Q_table'][model['next_state'],:]))
        
        
        model['Q_table'][model['current_state'], model['action']] = model['Q_table'][model['current_state'], model['action']] + model['lr']*(model['reward'] + model['gamma']*max(model['Q_table'][model['next_state'],:]) - model['Q_table'][model['current_state'], model['action']])
        #print("Post_value:", model['Q_table'][model['current_state'], model['action']])
        
        #Update reward
        model['total_episode_reward'] = model['total_episode_reward'] + model['reward']
        
        #Print Q Table
        #print("Updated Q table")
        #print(model['Q_table'])
        
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
        #Print run record
        #print("Run record")
        #print(record)
        
        i1 = record['from_state']
        i2 = record['action']
        #print(i1, i2)
        #print(model['sa_info'][i1][i2])
        #print(model['sa_info'][record['from_state'][record['action']])
        
        #----Reward Array-------
        if record['to_state'] in model['reward_array']:
            if record['reward'] > model['reward_array'][record['to_state']]:
                model['reward_array'][record['to_state']] = record['reward']
        else:
            model['reward_array'][record['to_state']] = record['reward']
        
        
        #----------Update sa_info rewards-------------
        if len(model['sa_info'][i1][i2]) > 0:
            #If it exists, set reward to higher one to avoid resetting reward to zero. (quick fix)
            best_reward = max(model['sa_info'][i1][i2]['reward'], record['reward'])
        else:
            best_reward = record['reward']
        
        model['sa_info'][i1][i2] = {'to_state': int(record['to_state']), 'reward': best_reward, 'done': record['done']}
        #print("updated model for", i1," ", i2, " is ", model['sa_info'][i1][i2])
        
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
    
    #Reset path so that it is guaranteed to exist on draw
    #sa_path = []

    
    #----------------------------------------------
    #---------------Print Q Table------------------
    #-----------------------------------------------
    #print("Updated Q_table")
    #print(model['Q_table'])
    
    #-------------------------------------------------------------
    #----------Print Population Performance every # of episodes---
    #-------------------------------------------------------------
    if e % run['screen_viz_frequency'] == 0: 
        print_population_performance = True
    else:
        print_population_performance = False
    
    if e % run['ga_frequency'] == 0 and e!= 0 and random.random() < run['crossover_chance']:
        solving_episode = True
    else:
        solving_episode = False
    
    if print_population_performance:
        print("-"*40)
        print("Episode:", e, "Rewards: ")        
        for key, value in model_dict.items():
            model = value
            #Print reward
            print(key, ":", model['total_episode_reward'])
            print("Epsilon: ", model['exploration_proba'])
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
    if solving_episode:
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
            
            #Manually update the state-reward
            learned_info[j]['state-reward'] = model['reward_array']
            print("Updated state reward", model['reward_array'])
            print("Known_states",learned_info[j]['known_states'])
            
            for k in range(0,n_observations):
                existing_edges = []
                not_known_tos = [True] * n_observations
                known_edges_counter = 0
                
                for m in range(0, n_actions):
                    record = model['sa_info'][k][m]
                    if len(record) > 0:
                        known_edges_counter += 1
                        #Update reward record
                        
                        '''
                        try:
                            if learned_info[j]['state-reward'][record['to_state']] != record['reward']:
                                #print("Reward for ", record['to_state'], "changing from ", learned_info[j]['state-reward'][record['to_state']], "to ", record['reward'])
                        except:
                            #print("Reward for ", record['to_state'], "changing to ", record['reward'])
                            pass
                        '''
                        
                        
                        #learned_info[j]['state-reward'][record['to_state']] = record['reward']
                        
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
        print("Starting crossover")
        crossover=smart_crossover.Smart_Crossover(learned_info, run)
        
        
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
        
        #---------------------------------------------------
        #---------------Alternate Update Strategies---------
        #---------------------------------------------------
        
        
        if run['policy_mixing']:
            #Clear past models (For now)
            model['mixing_policy_list'] = []
            #Generate new policy out of path.
            print("Generating new policy")
            new_entry = {'policy': {}, 'uses_remaining': run['policy_uses']}
            for entry in sa_path:
                #print(entry)
                #Add the path entries as a partial policy
                new_entry['policy'][entry[0]] = entry[1]
            
            #Add the new entry to the policy mixing list
            model['mixing_policy_list'].append(new_entry)
        
        if run['update_path_increment'] or run['update_path_max_q']:
            #For each individual, set the path sa pairs to 1.5 times the highest Q value.
            parent = model
            #Calculate the highest Q value
            max_Q = max(max(x) for x in parent['Q_table'])
            
            #print("Max Q: ", max_Q)
            #print("Pre Q table")
            #print(parent['Q_table'])
            
            #Direction_mapping
            direction_mapping = {0:'L', 1:'U', 2:'R', 3:'D'}
                                
            #print("Pre-Updated")
            #Print the greedy path through
            max_indices = []
            for row in parent['Q_table']:
                #Get max index
                max_indices.append(row.argmax())
            max_indices = [direction_mapping[x] for x in max_indices]
            max_indices = np.array(max_indices)
            #print(max_indices.reshape(run['grid_dims'][0], run['grid_dims'][1]))
            
            print("Pre")
            print(parent['Q_table'])
            for sa_pair in sa_path:
                if run['update_path_increment']:
                    print("Incrementing")
                    
                    parent['Q_table'][sa_pair[0], sa_pair[1]] += run['path_q_increment']

                    
                if run['update_path_max_q']:
                    print("Maxing")
                    #Max replace.
                    parent['Q_table'][sa_pair[0], sa_pair[1]] = max(max_Q * 1.2, 1)
            

            print("Post")
            print(parent['Q_table'])
        
        #Normalize matrix
        #normalize(parent['Q_table'], axis=1, norm='l1')
        #normalize(parent['Q_table'], axis=0, norm='l1')
        
        #print("Post Q table")
        #print(parent['Q_table'])
        #print()
        
        #Show the updated path
        #print("Updated")
        #Print the greedy path through
        max_indices = []
        for row in model['Q_table']:
            #Get max index
            max_indices.append(row.argmax())
        max_indices = [direction_mapping[x] for x in max_indices]
        max_indices = np.array(max_indices)
        #print(max_indices.reshape(run['grid_dims'][0], run['grid_dims'][1]))

    #----------------If either, update graphics-----------------
    if print_population_performance or solving_episode:
        print("pop, solving", print_population_performance, solving_episode)
        #time.sleep(2)
    
        if run['q_visuals']:
            #Update and visualize
            #time.sleep(1)
            print("updating model")
            update_plot(model['Q_table'])
            print(model['Q_table'])
            
        if run['graph_visuals']:
            #Update pygame display
            #Gather the necessary information
            known_edges = []
            explored = []
            known_rewards = []
            
            try:
                print("Learned info")
                known_edges = learned_info[0]['edges_exist']
                explored = learned_info[0]['known_states']
                known_rewards = []
                for key, value in learned_info[0]['state-reward'].items():
                    if value > 0:
                        known_rewards.append(key)
                print('solution_path', solution_path)
                print("known_rewards", known_rewards)
            except:
                print("No learned info yet")
                solution_path = []
            
            
            #explored = [0,1,5,8,10]
            #known_rewards = [0,5,9,12]
            #known_edges = [(8,9),(10,5), (9,6),  (5,9), (2,12)]
            #print("model q table")
            #print(model['Q_table'])
            greedy_path, actions = q_to_path(env, model['Q_table'], test_length=20)
            
            #Plot the graph and wait
            graph_plotter.draw_plot(screen, run['grid_dims'][0], run['grid_dims'][0], explored, known_rewards, known_edges, greedy_path=greedy_path, optim_path=solution_path)
        
        #Wait after either
        if run['pause_on_visual']:
            input("Press enter")    

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
    #print("Rewards", model['rewards_per_episode'])
    running_avg = []
    window = 50
    for point in range(window, len(model['rewards_per_episode'])):
        running_avg.append(np.mean(model['rewards_per_episode'][point-window: point]))
            
    plt.plot(running_avg)

plt.title("Rewards per individual")
plt.savefig(os.path.join('output_images','rewards_' + str(run['crossover_chance']) + '.png'))
plt.show()


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

#Quit out of pygame
pygame.quit()

