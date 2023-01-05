import numpy as np
import gym
import vector_grid_goal #Custom environment
import matplotlib.pyplot as plt


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



n_observations = env.observation_space.n
n_actions = env.action_space.n

#Initialize the Q-table to 0
Q_table = np.zeros((n_observations,n_actions))
print(Q_table)


n_episodes = 10000
max_iter_episode = 20
exploration_proba = 1
exploration_decreasing_decay = 0.0005
min_exploration_proba = 0.01
gamma = 0.90 #Changed from 0.99
lr = 0.1

total_rewards_episode = list()
rewards_per_episode = []


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
plt.show()