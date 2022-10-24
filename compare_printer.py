#Kyle Norland
#10-23-22
#Compare performance with and without the crossover.


#Imports
import matplotlib.pyplot as plt
import json



#Load in the json files
regular_json = "regular_json.json"
crossover_json = "crossover_json.json"

with open(regular_json) as f:
    regular_dict = json.load(f) 

with open(crossover_json) as f:
    crossover_dict = json.load(f)
    


#Plot them
'''
#Plot each
for reward_series in regular_dict["model_rewards"]:
    plt.plot(reward_series, color='red', marker='o', label='Baseline')
    
for reward_series in crossover_dict["model_rewards"]:
    plt.plot(reward_series, color='green', marker='o', label='With Modified Crossover')
'''
#Plot average
reward_series = []
for i in range(len(regular_dict["model_rewards"][0])):
    i_series = [x[i] for x in regular_dict['model_rewards']]
    average = sum(i_series) / len(i_series)
    reward_series.append(average)
plt.plot(reward_series, color='red', marker='o', label='Baseline')


reward_series = []
for i in range(len(crossover_dict["model_rewards"][0])):
    i_series = [x[i] for x in crossover_dict['model_rewards']]
    average = sum(i_series) / len(i_series)
    reward_series.append(average)
plt.plot(reward_series, color='green', marker='o', label='With Modified Crossover')   


  
plt.savefig("Comparison Plot.png") 
