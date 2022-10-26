import numpy as np
import gym
import vector_grid_goal #Custom environment
import random
from datetime import datetime
import copy

#Matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time



#Initialize gym stuff
#env = gym.make("FrozenLake-v1")
n_episodes = 100

#Seed randoms and environment
#Now seeded from time.
random.seed(1234)
np.random.seed(1234)
#env = gym.make('FrozenLake-v1')
#env.seed(0)

#Start timer
start_time = datetime.now()

#Initialize Environment
grid_dims = (5,5)
player_location = (0,0)
goal_location = (4,4)

env = vector_grid_goal.CustomEnv(grid_dims=grid_dims, player_location=player_location, goal_location=goal_location, map='random')
n_observations = env.observation_space.n
n_actions = env.action_space.n



model_dict = {}
for i in range(1):
    model_dict[i] = {}
    #model_dict[i]['n_episodes'] = 20000      #number of episode we will run
    model_dict[i]['max_iter_episode'] = 100  #maximum of iteration per episode
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
    #model_dict[i]['transition_bank'] = []
    model_dict[i]['transition_dict'] = {}
    model_dict[i]['reward_dict'] = {}
    #model_dict[i]['static_transitions'] = []
    #model_dict[i]['known_rewards'] = np.full(n_observations,-1)
    #model_dict[i]['known_transitions'] = np.full((n_observations, n_observations), 0)
    #model_dict[i]['known_not_transitions'] = np.full((n_observations, n_observations), 0)
    
    
'''
#Initialize the Q-table to 0
Q_table = np.zeros((n_observations,n_actions))
print(Q_table)

#Parameters
n_episodes = 20000      #number of episode we will run
max_iter_episode = 100  #maximum of iteration per episode
exploration_proba = 1   #initialize the exploration probability to 1
exploration_decreasing_decay = 0.001    #exploartion decreasing decay for exponential decreasing
min_exploration_proba = 0.01    # minimum of exploration proba
gamma = 0.99    #discounted factor
lr = 0.1    #learning rate

#Output Structures
total_rewards_episode = list()
rewards_per_episode = []
plotted_rewards = [[]]
'''



#Matplotlib globals
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

#we iterate over episodes
for e in range(n_episodes):
    for key in model_dict.keys():
        model = model_dict[key]     #copy by reference
        #we initialize the first state of the episode
        model['current_state'] = env.reset()
        model['done'] = False
        
        #sum the rewards that the agent gets from the environment
        model['total_episode_reward'] = 0
        
        for i in range(model['max_iter_episode']): 
            #Epsilon random
            if np.random.uniform(0,1) < model['exploration_proba']:
                model['action'] = env.action_space.sample()
            else:
                model['action'] = np.argmax(model['Q_table'][model['current_state'],:])
            
            #Run action
            model['next_state'], model['reward'], model['done'], _ = env.step(model['action'])
            
            #Add to transition bank
            '''
            nr = {}
            nr['from_state'] = model['current_state']
            nr['action'] = model['action']
            nr['to_state'] = model['next_state']
            nr['reward'] = model['reward']
            nr['done'] = model['done']
            '''
            
            #print("The model is: ", model)
            #model['transition_bank'].append(copy.deepcopy(nr))
            
            #----------------------------------------------
            #----------------Update Dicts-------------------
            #----------------------------------------------
            
            #Next state dict
            #If current state not in dict.
            if model['current_state'] not in model['transition_dict']:
                model['transition_dict'][model['current_state']] = {model['action']: model['next_state']}
            #Starting state exists
            else:
                model['transition_dict'][model['current_state']][model['action']] = model['next_state']
                
            #Reward dict
            #If current state not in dict.
            if model['current_state'] not in model['transition_dict']:
                model['transition_dict'][model['current_state']] = {model['action']: model['reward']}
            #Starting state exists
            else:
                model['transition_dict'][model['current_state']][model['action']] = model['reward']   
                
                
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
        
        #We update the exploration proba using exponential decay formula 
        model['exploration_proba'] = max(model['min_exploration_proba'], np.exp(-model['exploration_decreasing_decay']*e))
        model['rewards_per_episode'].append(model['total_episode_reward'])
        
        #Update/visualize parameters
        #Prepare partial outputs
        if e % 1000 == 0 and e != 0:
            print("Avg Reward at", e, "is", np.mean(model['rewards_per_episode'][e-1000:e]))
            model['plotted_rewards'].append(np.mean(model['rewards_per_episode'][e-1000:e]))
            animate(5)
        if e == n_episodes-1:
            print("Done")
            
        #Update the models memory tables
        #for record in model['transition_bank']:
            #print(record)
            
        #Update failed transitions
        #print()
        #print("Static transitions")
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
        
        model['transition_bank'] = []
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







        
#Final outputs
#print(Q_table)
'''
print("Mean reward per thousand episodes")
for i in range(10):
    print((i+1)*1000,": mean episode reward: ", np.mean(rewards_per_episode[1000*i:1000*(i+1)]))
'''
    
    
    
    
    
    
    
    
    
    
    
    