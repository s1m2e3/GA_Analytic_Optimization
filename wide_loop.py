#Kyle Norland, using code from Sam
#10/10/22, modified 12/1/22


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


#--------------------------------------------------
#---------------------Functions--------------------
#--------------------------------------------------
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

#-------------------State Discretization------------------
def get_discrete_state(self, state, bins, obsSpaceSize):
    #https://github.com/JackFurby/CartPole-v0
    stateIndex = []
    for i in range(obsSpaceSize):
        stateIndex.append(np.digitize(state[i], bins[i]) - 1) # -1 will turn bin into index
    return tuple(stateIndex)

def init_environment(env_config):
    env_name = env_config['env_name']
    
    if env_name == 'FrozenLake-v1':
        env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
        #env = gym.make("FrozenLake-v1", is_slippery=False)
    
    if env_name == 'vector_grid_goal':
        grid_dims = (5,5)
        player_location = (0,0)
        goal_location = (4,4)
        custom_map = np.array([[0,0,0,0,0],
                                [0,1,0,0,0],
                                [0,0,0,1,0],
                                [0,0,0,0,1],
                                [0,0,1,0,0]])
                                
    
        env = vector_grid_goal.CustomEnv(grid_dims=grid_dims, player_location=player_location, goal_location=goal_location, map=custom_map)
    
    
    return env

#GA Evaluations and Custom for Experiment
def evaluate_Q_individual(individual, e):
    #we initialize the first state of the episode
    model['current_state'] = env.reset()
    model['done'] = False
    
    #sum the rewards that the agent gets from the environment
    model['total_episode_reward'] = 0
    
    #-------------------------------------------------------
    #--------------Individual Training/Evaluation-----------
    #-------------------------------------------------------
    #print("max iters:", model['max_iter_episode'])
    for i in range(model['max_iter_episode']): 
        
        #Epsilon random action decision
        if np.random.uniform(0,1) < model['exploration_proba']:
            #model['action'] = env.action_space.sample()
            model['action'] = env.action_space.sample()
            
        else:
            model['action'] = np.argmax(model['Q_table'][model['current_state'],:])
        
        #Run action
        model['next_state'], model['reward'], model['done'], _ = env.step(model['action'])
        
        #--------------------------------------------------
        #---------------Record Transitions-----------------
        #--------------------------------------------------
        nr = {}
        nr['from_state'] = int(model['current_state'])
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
    

    #------------------------End Of Episode---------------------------
    #Convert model states to int
    #model['current_state'] = int(model['current_state'])
    #model['next_state'] = int(model['next_state'])
    
    #Decay Exploration Probability
    model['exploration_proba'] = max(model['min_exploration_proba'], np.exp(-model['exploration_decreasing_decay']*e))
    
    #Update total rewards
    model['rewards_per_episode'].append(model['total_episode_reward'])
    
    #----------------------Convert Transitions into SA knowledge Table--------------------
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
    
    #-----------Clear transition bank after information has been added-------------------------
    model['transition_bank'] = []
    #print("Size of transition_bank: ", len(model['transition_bank']))
    
    '''
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


def gen_exp_ga_frequency(default_run):
    #Experiment Objective: See effect of ga_frequency
    #Experiment Designer: Kyle Norland
    #Designed 11/22
    
    ga_frequency_settings = [500, 20000]#[50, 100, 250, 500, 1000, 20000] #[5,10,25,50,100,200,300,400,500]
    random_seeds = rng.integers(low=0, high=9999, size=5)
        
    new_experiment = {'runs':[]}
    new_experiment['generation_time']= time.time()
    new_experiment['variables'] = ['ga_frequency']
    new_experiment['experiment_name'] = "changing_ga_frequency"

    color_list = ['green', 'blue', 'red', 'yellow', 'orange', 'brown']
    color_counter = 0
    
    for ga_frequency in ga_frequency_settings:
        for seed in random_seeds:
            new_run = copy.deepcopy(default_run)
            new_run['ga_frequency'] = ga_frequency
            new_run['np_seed'] = seed
            new_run['env_seed'] = seed
            new_run['python_seed'] = seed
            new_run['color'] = color_list[color_counter]
            new_run['label'] = "".join([i + ': ' + str(new_run[i]) + ' ' for i in new_experiment['variables']])
            print('Settings: ', new_run['label'])
            
            #Add run to experiment
            new_experiment['runs'].append(copy.deepcopy(new_run))
        
        color_counter += 1   
    print("Returning new experiment")
    return new_experiment              

def gen_episode_length(default_run):
    #Experiment Objective: See effect of episode length
    #Experiment Designer: Kyle Norland
    #Designed 12-9-22
    
    ga_frequency_settings = [500, 20000]#[50, 100, 250, 500, 1000, 20000] #[5,10,25,50,100,200,300,400,500]
    episode_lengths = [5, 10, 20]
    random_seeds = rng.integers(low=0, high=9999, size=2)
        
    new_experiment = {'runs':[]}
    new_experiment['generation_time']= time.time()
    new_experiment['variables'] = ['ga_frequency', 'max_iter_episode']
    new_experiment['experiment_name'] = "changing_episode_length"

    color_list = ['green', 'blue', 'red', 'yellow', 'orange', 'brown']
    color_counter = 0
    
    for ga_frequency in ga_frequency_settings:
        for episode_length in episode_lengths:
            for seed in random_seeds:
                new_run = copy.deepcopy(default_run)
                new_run['ga_frequency'] = ga_frequency
                new_run['max_iter_episode'] = episode_length
                new_run['np_seed'] = seed
                new_run['env_seed'] = seed
                new_run['python_seed'] = seed
                new_run['color'] = color_list[color_counter]
                print(new_run['color'])
                new_run['label'] = "".join([i + ': ' + str(new_run[i]) + ' ' for i in new_experiment['variables']])
                print('Settings: ', new_run['label'])
                
                #Add run to experiment
                new_experiment['runs'].append(copy.deepcopy(new_run))
        
            color_counter += 1
    print("Returning new experiment")
    return new_experiment     
        
#-------------------------------------------
#------------------Main---------------------
#-------------------------------------------

if __name__== "__main__":
    
    #------------------------------------------
    #--------------Generate Experiment---------
    #------------------------------------------    
    #Default Run Settings
    default_run = {}
    #default_run['env'] = 'FrozenLake-v1'
    default_run['env'] = 'vector_grid_goal'
    default_run['n_episodes'] = 4000
    default_run['ga_frequency'] = 200
    default_run['crossover_chance'] = 1
    default_run['mutation_prob'] = 0.2 
    default_run['python_seed'] = 1234
    default_run['np_seed'] = 1234
    default_run['env_seed'] = 1234
    default_run['grid_dims'] = (5,5)
    default_run['player_location'] = (0,0)
    default_run['goal_location'] = (4,4)
    default_run['map'] = np.array([[0,1,1,1,0],
                                    [0,0,1,0,0],
                                    [0,0,0,0,0],
                                    [0,1,0,0,0],
                                    [0,0,0,0,0]])
    default_run['max_iter_episode'] = 10      #maximum of iteration per episode
    default_run['exploration_proba'] = 1    #initialize the exploration probability to 1
    default_run['exploration_decreasing_decay'] = 0.0001       #exploration decreasing decay for exponential decreasing
    default_run['min_exploration_proba'] = 0.01
    default_run['gamma'] = 0.99            #discounted factor
    default_run['lr'] = 0.1                 #learning rate 
    default_run['pop_size'] = 2
    
    default_run['output_dict'] = {}
    
    #Env Config
    default_run['env_config'] = {}
    #default_run['env_config']['env_name'] = 'FrozenLake-v1'
    default_run['env_config']['env_name'] = default_run['env']
    #default_run['env_config']['env_name']
    
    #Output Path
    default_run['output_path'] = 'GA_output'
    if not os.path.exists(default_run['output_path']):
        os.makedirs(default_run['output_path'], exist_ok = True)    
    
    #Generate or Load Experiment
    folder_mode = False
    generate_mode = True
    
    experiment = {'runs': []}
    
    if generate_mode:
        #Generate experiment
        #experiment = copy.deepcopy(gen_epsilon_exist(default_run))
        #experiment = copy.deepcopy(gen_exp_ga_frequency(default_run))
        experiment = copy.deepcopy(gen_episode_length(default_run))
        #experiment = copy.deepcopy(epsilon_switching(default_run))
        #Save experiment
        experiment_name = str(experiment['generation_time']) + '.json'
        with open(os.path.join('saved_experiments', experiment_name), 'w') as f:
            json.dump(experiment, f, cls=MyEncoder)
        
    #----------------------------------------------------
    #---------------Run the Experiment-------------------
    #----------------------------------------------------
    for run in experiment['runs']:
        #Start Timer
        print("Run")
        print(run)
        run['run_start_time'] = time.time()

        #Set random seeds
        random.seed(run['python_seed'])
        np.random.seed(run['np_seed'])
        
        #Initialize environment
        env = init_environment(run['env_config'])
        
        #Env seed for action space
        env.action_space.np_random.seed(run['env_seed'])        

        #Grab observation and action space for initializations
        n_observations = env.observation_space.n
        n_actions = env.action_space.n
        


        #--------------------------------------------------------------------
        #--------------------Initialize Genetic Algorithm--------------------
        #--------------------------------------------------------------------
        #Generate population
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
            
            #Outputs
            model_dict[i]['rewards_per_episode'] = []
            model_dict[i]['plotted_rewards'] = []

        
        #----------------------------------------------------------------
        #-------------------Training Loop--------------------------------
        #----------------------------------------------------------------
        #For number of episodes
        for e in range(run['n_episodes']):
            
            #-----------Evaluate Each Individual------------------
            for key, value in model_dict.items():
                model = value     #copy by reference
                
                #Evaluate the individual
                evaluate_Q_individual(model, e)
                
            #----------Print Population Performance every # of episodes---
            if e % 100 == 0:
                print("-"*40)
                print("Episode:", e, "Rewards: ")        
                for key, value in model_dict.items():
                    model = value
                    #Print reward
                    print(key, ":", model['total_episode_reward'])
                print("-"*40)            
                
            #-------------------------------------------------------------
            #--------------------GA Procedure-----------------------------
            #-------------------------------------------------------------
            #If time to run the ga
            if e % run['ga_frequency'] == 0 and e != 0:
                
                mutprb = run['mutation_prob']       #Mutation probability
                
                #Full crossover without elites (TODO: Alternate crossover structure)
                for i in range(1, len(model_dict)):
                    #Apply crossover if less than crossover probability
                    if random.random() < run['crossover_chance']:
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
                        
                        #Run the crossover
                        crossover=smart_crossover.Smart_Crossover(learned_info)
                        
                        print("Solution time: ", datetime.now() - start)

                        print("Solution:",crossover.solution)
                        #print("Objective Value:",crossover.value)
                        
                        #Update individuals
                        create_new_Q_individuals(parent_1, parent_2, crossover)


        #-----------------------------------------------------------------------
        #------------------------End of Run Outputs----------------------
        #-----------------------------------------------------------------------
        #Save details to run.
        #Save model rewards as json
        run['models'] = copy.deepcopy(model_dict)
        
    #--------------------------------------------------------
    #--------------End of Experiment Processing--------------
    #--------------------------------------------------------
    
    #Save
    #save_name = str(round(time.time())) + "_json_output.json"
    
    out_folder_name = str(experiment['generation_time']) + '_experiment'
    os.makedirs(os.path.join('results', out_folder_name), exist_ok=True)
    
    save_name = "json_output.json"
    with open(os.path.join('results', out_folder_name, save_name ), 'w') as f:
        json.dump(experiment, f, cls=MyEncoder)    
    
    #Save a text file with the changed variables
    save_name = '+'.join(experiment['variables'])
    file_path = os.path.join('results', out_folder_name, save_name)
    with open(file_path, 'w') as f:
        pass

    print("Processing Visuals")
    output_processor.process_folders(latest=True)
    
    
    
    
    
    
    
        
    '''
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
    '''
