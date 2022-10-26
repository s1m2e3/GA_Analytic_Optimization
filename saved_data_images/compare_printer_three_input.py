#Kyle Norland
#10-23-22
#Compare performance with and without the crossover.


#Imports
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

#call fancy seaborn style
sns.set()

#Load in the json files
regular_json = "regular_json.json"
crossover_cycle_elim = "crossover_new_cycle_elimination.json"
crossover_no_cycle_elim = "crossover_new_no_cycle_elimination.json"

with open(regular_json) as f:
    regular_dict = json.load(f) 

with open(crossover_cycle_elim) as f:
    crossover_cycle_elim_dict = json.load(f)

with open(crossover_no_cycle_elim) as f:
    crossover_no_cycle_elim_dict = json.load(f)


#---------------------------------------------------
#----------Model Reward Printing Full Detail--------
#---------------------------------------------------
plt.figure()
for reward_series in regular_dict["model_rewards"]:
    plt.plot(reward_series, color='red', marker='o', label='Baseline',alpha=0.7)
    
for reward_series in crossover_cycle_elim_dict["model_rewards"]:
    plt.plot(reward_series, color='green', marker='o', label='Modified Crossover with Cycle Elimination',alpha=0.3)

for reward_series in crossover_no_cycle_elim_dict["model_rewards"]:
    plt.plot(reward_series, color='darkgreen', marker='o', label='Modified Crossover without Cycle Elimination',alpha=0.3)
plt.savefig("Individuals Comparison Plot.png")
#---------------------------------------
#----------Model Reward Printing--------
#---------------------------------------
plt.figure()

#Kernel for averaging
kernel_size = 100
kernel = np.ones(kernel_size) / kernel_size

reward_series = []
for i in range(len(regular_dict["model_rewards"][0])):
    i_series = [x[i] for x in regular_dict['model_rewards']]
    average = sum(i_series) / len(i_series)
    reward_series.append(average)
reward_series = np.convolve(reward_series, kernel, mode='same')
plt.plot(reward_series[:-kernel_size], color='red', linewidth=2, label='Baseline (No Analytic Crossover)')


reward_series = []
for i in range(len(crossover_cycle_elim_dict["model_rewards"][0])):
    i_series = [x[i] for x in crossover_cycle_elim_dict['model_rewards']]
    average = sum(i_series) / len(i_series)
    reward_series.append(average)
reward_series = np.convolve(reward_series, kernel, mode='same')
plt.plot(reward_series[:-kernel_size], color='#15B01A', linewidth=2, label='Crossover with Cycle Elimination')

reward_series = []
for i in range(len(crossover_no_cycle_elim_dict["model_rewards"][0])):
    i_series = [x[i] for x in crossover_no_cycle_elim_dict['model_rewards']]
    average = sum(i_series) / len(i_series)
    reward_series.append(average)
reward_series = np.convolve(reward_series, kernel, mode='same')
plt.plot(reward_series[:-kernel_size], color='orange', linewidth=2, label='Crossover without Cycle Elimination') 


#Other characteristics of plot
plt.title("Method Comparison: Episode vs Reward")
plt.legend(loc='upper left')
plt.xlabel("Episode")
plt.ylabel("Population Average Reward")
plt.savefig("Comparison Plot.png") 
