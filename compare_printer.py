#Kyle Norland
#10-23-22
#Compare performance with and without the crossover.


#Imports
import matplotlib.pyplot as plt
import seaborn as sns
import json

#call fancy seaborn style
sns.set()

#Load in the json files
regular_json = "regular_json.json"
crossover_json = "crossover_json.json"

with open(regular_json) as f:
    regular_dict = json.load(f) 

with open(crossover_json) as f:
    crossover_dict = json.load(f)
    


#Plot them
    
#Plot each
plt.figure()
for reward_series in regular_dict["model_rewards"]:
    plt.plot(reward_series, color='red', marker='o', label='Baseline',alpha=0.7)
    
for reward_series in crossover_dict["model_rewards"]:
    plt.plot(reward_series, color='green', marker='o', label='With Modified Crossover',alpha=0.3)
plt.savefig("Individuals Comparison Plot.png")
    
#Plot average
plt.figure()
reward_series = []
for i in range(len(regular_dict["model_rewards"][0])):
    i_series = [x[i] for x in regular_dict['model_rewards']]
    average = sum(i_series) / len(i_series)
    reward_series.append(average)
plt.plot(reward_series, color='red', marker='o', label='Baseline',alpha=0.9)

reward_series = []
for i in range(len(crossover_dict["model_rewards"][0])):
    i_series = [x[i] for x in crossover_dict['model_rewards']]
    average = sum(i_series) / len(i_series)
    reward_series.append(average)
plt.title('Found rewards over iterations')
plt.xlabel("Episode number ")
plt.ylabel("Average rewards per episode")
plt.plot(reward_series, color='green', marker='o', label='With Modified Crossover',alpha=0.5)   
  
plt.savefig("Comparison Plot.png") 
