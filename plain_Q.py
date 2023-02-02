import numpy as np
import gym
import vector_grid_goal #Custom environment
<<<<<<< HEAD
<<<<<<< HEAD
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


#--------Functions--------------
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

=======
import matplotlib.pyplot as plt


>>>>>>> 7b78d1458ddfb1f2a19265523ce75b37e6b0f8ac
=======
import matplotlib.pyplot as plt


>>>>>>> 7b78d1458ddfb1f2a19265523ce75b37e6b0f8ac
#Make environment
run = {}
run['grid_dims'] = (5,5)
run['player_location'] = (0,0)
run['goal_location'] = (4,4)
run['map'] = np.array([[0,0,0,0,0],
                        [0,1,0,0,0],
                        [0,0,0,1,0],
                        [0,0,0,0,1],
                        [0,0,1,0,0]])
env = vector_grid_goal.CustomEnv(grid_dims=run['grid_dims'], player_location=run['player_location'], goal_location=run['goal_location'], map=run['map'])


<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 7b78d1458ddfb1f2a19265523ce75b37e6b0f8ac
=======

>>>>>>> 7b78d1458ddfb1f2a19265523ce75b37e6b0f8ac
n_observations = env.observation_space.n
n_actions = env.action_space.n

#Initialize the Q-table to 0
Q_table = np.zeros((n_observations,n_actions))
print(Q_table)

<<<<<<< HEAD
<<<<<<< HEAD
#Parameters
=======

>>>>>>> 7b78d1458ddfb1f2a19265523ce75b37e6b0f8ac
=======

>>>>>>> 7b78d1458ddfb1f2a19265523ce75b37e6b0f8ac
n_episodes = 10000
max_iter_episode = 20
exploration_proba = 1
exploration_decreasing_decay = 0.0005
min_exploration_proba = 0.01
gamma = 0.90 #Changed from 0.99
lr = 0.1
<<<<<<< HEAD
<<<<<<< HEAD
q_visuals = True
q_vis_frequency = 10
graph_visuals = True
=======
>>>>>>> 7b78d1458ddfb1f2a19265523ce75b37e6b0f8ac
=======
>>>>>>> 7b78d1458ddfb1f2a19265523ce75b37e6b0f8ac

total_rewards_episode = list()
rewards_per_episode = []

<<<<<<< HEAD
<<<<<<< HEAD
#Initialize the visualizations
if q_visuals:
    #Put initial matplotlib setting
    interactive(True)

    #Generate initial plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    rows = len(Q_table)
    columns = len(Q_table[0])
    X = np.arange(0, columns, 1)
    Y = np.arange(0, rows, 1)
    X, Y = np.meshgrid(X, Y)
    Z = Q_table
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

if graph_visuals:
    #Set up pygame
    screen = graph_plotter.initialize_plot()
    
    '''
    pygame.init()# Setting your screen size with a tuple of the screen width and screen height
    screen_width = 800
    screen_height = 600
    aspect_ratio = screen_width/screen_height
    screen = pygame.display.set_mode((800,600))# Setting a random caption title for your pygame graphical window.
    pygame.display.set_caption("pygame test")
    '''
    
    #Draw initial plot
    explored = [0,1,5,8,10]
    known_rewards = [0,5,9,12]
    known_edges = [(8,9),(10,5), (9,6),  (5,9), (2,12)]
    graph_plotter.draw_plot(screen, run['grid_dims'][0], run['grid_dims'][0], explored, known_rewards, known_edges)


=======
>>>>>>> 7b78d1458ddfb1f2a19265523ce75b37e6b0f8ac
=======
>>>>>>> 7b78d1458ddfb1f2a19265523ce75b37e6b0f8ac

#Core and added actions
#Left -0
#Down = 1
#Right = 2
#Up = 3

#we iterate over episodes
for e in range(n_episodes):
    #we initialize the first state of the episode
    current_state = env.reset()
    done = False
    
    #sum the rewards that the agent gets from the environment
    total_episode_reward = 0
    
    for i in range(max_iter_episode): 
        # we sample a float from a uniform distribution over 0 and 1
        # if the sampled flaot is less than the exploration proba
        #     the agent selects arandom action
        # else
        #     he exploits his knowledge using the bellman equation 
        
        if np.random.uniform(0,1) < exploration_proba:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[current_state,:])
        
        # The environment runs the chosen action and returns
        # the next state, a reward and true if the epiosed is ended.
        next_state, reward, done, _ = env.step(action)
        
        # We update our Q-table using the Q-learning iteration
        Q_table[current_state, action] = (1-lr) * Q_table[current_state, action] +lr*(reward + gamma*max(Q_table[next_state,:]))
        total_episode_reward = total_episode_reward + reward
        # If the episode is finished, we leave the for loop
        if done:
            break
        current_state = next_state
    #We update the exploration proba using exponential decay formula 
    exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
    rewards_per_episode.append(total_episode_reward)
    
<<<<<<< HEAD
<<<<<<< HEAD
    #If at frequency, update the q_table
    if e % q_vis_frequency == 0 and e!= 0:
        #Update and visualize
        #time.sleep(1)
        update_plot(Q_table)
        print(Q_table)
        
        #Update pygame display
        explored = [0,1,5,8,10]
        known_rewards = [0,5,9,12]
        known_edges = [(8,9),(10,5), (9,6),  (5,9), (2,12)]
        greedy_path, actions = q_to_path(env, Q_table, test_length=20)
        graph_plotter.draw_plot(screen, run['grid_dims'][0], run['grid_dims'][0], explored, known_rewards, known_edges, greedy_path=greedy_path)
        input("Press enter")
=======
>>>>>>> 7b78d1458ddfb1f2a19265523ce75b37e6b0f8ac
=======
>>>>>>> 7b78d1458ddfb1f2a19265523ce75b37e6b0f8ac

#Final outputs
print(Q_table)

print("Mean reward per thousand episodes")
for i in range(10):
    print((i+1)*1000,": mean espiode reward: ", np.mean(rewards_per_episode[1000*i:1000*(i+1)]))

running_avg = []
window = 10
for point in range(window, len(rewards_per_episode)):
    running_avg.append(np.mean(rewards_per_episode[point-window: point]))
    
#plt.plot(rewards_per_episode)
#print(rewards_per_episode[:10])
#print(running_avg[:10])

plt.plot(running_avg)
<<<<<<< HEAD
<<<<<<< HEAD
plt.show()

#Quit out of pygame
pygame.quit()
#sys.exit()
=======
plt.show()
>>>>>>> 7b78d1458ddfb1f2a19265523ce75b37e6b0f8ac
=======
plt.show()
>>>>>>> 7b78d1458ddfb1f2a19265523ce75b37e6b0f8ac
