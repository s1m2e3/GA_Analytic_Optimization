#Author: Kyle Norland, Sam
#Date: 9-29-22
#Purpose: Implement GA for grid world with modified crossover.


#-------------------Imports-----------------------
#General
import numpy as np
import os
import time
import random
import time
import json
from copy import deepcopy
from datetime import datetime

#OpenAI Gym
import gym
from gym import wrappers
import vector_grid_goal #Custom environment

#Matplotlib
'''
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

from moviepy.editor import ImageSequenceClip
'''

#deap
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


#--------------------------------------------------
#---------------------Functions--------------------
#--------------------------------------------------

def get_discrete_state(self, state, bins, obsSpaceSize):
    #https://github.com/JackFurby/CartPole-v0
    stateIndex = []
    for i in range(obsSpaceSize):
        stateIndex.append(np.digitize(state[i], bins[i]) - 1) # -1 will turn bin into index
    return tuple(stateIndex)
   
def cxTwoPointCopy(ind1, ind2):
    """Executes a two-point crossover on the input :term:`sequence`
    individuals. The two individuals are modified in place and both keep
    their original length.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.randint` function from the Python
    base :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

    print(ind1)
    print(type(ind1))
    return ind1, ind2


def cxTwoDim(ind1, ind2):
    #Grabs a 2d section and swaps it between two implied grid individuals
    global grid_dims
    
    #Convert the strings to matrices
    indiv_1 = np.array(list(ind1)).reshape(grid_dims[0], grid_dims[1])
    indiv_2 = np.array(list(ind2)).reshape(grid_dims[0], grid_dims[1])

    cxp_1 = [np.random.randint(0,grid_dims[0]), np.random.randint(0,grid_dims[1])]
    cxp_2 = [np.random.randint(0,grid_dims[0]), np.random.randint(0,grid_dims[1])]

    #Ensure that first point is upper right
    #print(cxp_1, cxp_2)
    if cxp_1[0] > cxp_2[0]:
        cxp_1[0],  cxp_2[0] = cxp_2[0],  cxp_1[0]

    if cxp_1[1] > cxp_2[1]:
        cxp_1[1],  cxp_2[1] = cxp_2[1],  cxp_1[1]
        
    '''
    print(indiv_1)
    print()
    print(indiv_2)
    print(cxp_1, cxp_2)
    print()


    print("Crossing Over")
    #Do the crossover
    print(indiv_1[cxp_1[0]:cxp_2[0], cxp_1[1]:cxp_2[1]])
    print()
    print(indiv_2[cxp_1[0]:cxp_2[0], cxp_1[1]:cxp_2[1]])
    print()
    '''

    temp_1 = deepcopy(indiv_1[cxp_1[0]:cxp_2[0], cxp_1[1]:cxp_2[1]])
    temp_2 = deepcopy(indiv_2[cxp_1[0]:cxp_2[0], cxp_1[1]:cxp_2[1]])
    
    indiv_1[cxp_1[0]:cxp_2[0], cxp_1[1]:cxp_2[1]] = temp_2
    indiv_2[cxp_1[0]:cxp_2[0], cxp_1[1]:cxp_2[1]] = temp_1
    
    indiv_1 = indiv_1.flatten()
    indiv_2 = indiv_2.flatten()
    
    ind1[:] = indiv_1[:]
    ind2[:] = indiv_2[:]

    return ind1, ind2


def eval_individual(ind):
    global env
    episode_len = 100
    render=False
    total_reward = 0
    
    if render: print("New episode")
    
    obs = env.reset()
    for t in range(episode_len):
        if render:
            env.render()
            print("y")
            time.sleep(0.05)
        action = ind[obs]
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward, 


#----------------------------------------------------
#-----------------MAIN-------------------------------
#----------------------------------------------------

if __name__ == "__main__":
    #Seed randoms and environment
    #Now seeded from time.
    random.seed(1234)
    np.random.seed(1234)
    #env = gym.make('FrozenLake-v1')
    #env.seed(0)
    
    #Start timer
    start_time = datetime.now()
    
    #Initialize Environment
    grid_dims = (10,10)
    player_location = (0,0)
    goal_location = (9,9)
    

    env = vector_grid_goal.CustomEnv(grid_dims=grid_dims, player_location=player_location, goal_location=goal_location, map='random')
    n_observations = env.observation_space.n
    n_actions = env.action_space.n



    genome_size = env.observation_space.n

    #Options for each gene
    gene_options = list(range(0, env.action_space.n))
    
    print("Genome Size: ", genome_size)
    print("Gene options: ", gene_options)


    #Deap initialization
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    #Structure Initializers
    toolbox.register("attr_int", random.randint, 0, env.action_space.n-1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, env.observation_space.n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    #Define operations
    toolbox.register("evaluate", eval_individual)
    
    
    #Modified crossover
    #toolbox.register("mate", tools.cxTwoPoint)
    #toolbox.register("mate", cxTwoPointCopy)
    toolbox.register("mate", cxTwoDim)
    
    
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=env.action_space.n-1, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    #Create the population and run the evolution
    pop = toolbox.population(n=200)
    
    
    #Register stats and other records
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    #Run the algorithm using a pre-set training algorithm.
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, 
                                   stats=stats, halloffame=hof, verbose=True)

    #Stop training time timer:
    print("Training time:", datetime.now() - start_time)
    
    #Get best agent and try it
    #print(pop)
    #print(stats)
    #print(hof)
    
    
    #Run eval
    print("Running with best")
    #print(hof[0])
    best = pop[0]
    print(pop[0])
    #Visualize it:
    print("Visualized")
    pre_flip = np.array(list(pop[0])).reshape(grid_dims[0], grid_dims[1])
    print(np.flip(pre_flip, 0))
    
    episode_len = 100
    total_reward = 0
    
    obs = env.reset()
    for t in range(episode_len):
        env.render()
        time.sleep(0.1)
        action = best[obs]
        print("Action: ", action)
        obs, reward, done, _ = env.step(action) #Random stepping?
        total_reward += reward
        if done:
            break
    print("Total reward: ", total_reward)  
    
    '''   
    #Initialize structures
    #Check for too large of an observation space
    env_type="Discrete"
    try:
        obs_size = len(env.observation_space.high)
        if len(env.observation_space.high) > 6:
            print("Observation space too large with current binning")  
        env_type = "Box"
    except:
        pass
    
    
        
    num_bins = 10
    size = ([num_bins * self.obs_space_size + [self.act_size])
    self.Q = np.random.uniform(low=-1, high=1, size=size)
    #self.Q = np.zeros((size)) 
    '''
