#Kyle Norland
#10-23-22

#Imports
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np

def process_folders(latest=True):
    #call fancy seaborn style
    sns.set()

    #Global Controls
    only_most_recent = latest

    input_folder = 'results'


    #----------------Determine which folders to process----------------
    folders = [x for x in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, x))]

    process_list = []

    if only_most_recent:
        recent_folder = None
        max_time = 0
        
        for folder in folders:
            parts = folder.split('_')
            exp_time = parts[0]
            exp_name = parts[1]
            print(exp_name, exp_time)
            if float(exp_time) > max_time:
                recent_folder = folder
                max_time = float(exp_time)
        if recent_folder is not None:
            process_list.append(folder)
            
    else:
        process_list = folders

    #--------------------------------------
    #--------Process Each Folder-----------
    #--------------------------------------
    for folder in process_list:
        json_file = os.path.join(input_folder, folder, 'json_output.json')
        with open(json_file) as f:
            exp_dict = json.load(f) 
        
        #Set up the figure   
        fig = plt.figure()
        ax = plt.subplot(111)
        

        color_array = ['r', 'b', 'g', 'c', 'm', 'y', 'orange', 'yellow', 'gray']
        label_array = []

        for run in exp_dict['runs']:
            if run['ga_frequency'] in label_array:
                color = color_array[label_array.index(run['ga_frequency'])]
            else:
                label_array.append(run['ga_frequency'])
                color = color_array[label_array.index(run['ga_frequency'])]
                
            for k,v in run['models'].items():
                #print(color)
                #print(run['ga_frequency'])
                #print(v['rewards_per_episode'])
                #print(len(v['rewards_per_episode']))
                #Average the outputs
                window = 50
                averaged = [np.mean(v['rewards_per_episode'][x-window:x]) for x in range(window, len(v['rewards_per_episode']))]
                #plt.plot(v['rewards_per_episode'], c=color, label=run['ga_frequency'])
                ax.plot(averaged, c=run['color'], label=run['label'])

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.4,
                     box.width, box.height * 0.6])
                     
        altered_string = ""
        for i in range(0, len(exp_dict['variables'])): altered_string += exp_dict['variables'][i]

        plt.title(exp_dict['experiment_name'] +  ": " + altered_string)
        #plt.legend()
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, fancybox=True, shadow=True)

        plt.show()
        plt.savefig(os.path.join(input_folder, folder, 'graph.png'))

if __name__ == '__main__':
    process_folders(latest=True)

'''
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
'''