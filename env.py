from simulation import *
import gym
import numpy as np

class MachineShopEnvironment(gym.Env):
    
    def __init__(self,simulation):
        
        super(MachineShopEnvironment,self).__init__()
        self.simulation = simulation
        
        # Define a 2-D observation space
        self.observation_shape = (self.simulation.n_processors)
        self.observation_space = gym.spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape)*self.simulation.n_jobs,
                                            dtype = np.float16)
    
        
        # Define an action space ranging from 0 to 4
        self.action_space = gym.spaces.Discrete(self.simulation.n_processors)
        
        def reset(self):
            self.simulation.reset()
            
        def step(self,action):

            '''
            Check if action is valid
            '''
            assert self.action_space.contains(action)

            self.simulation.assign(action)
            if self.simulation.check_sim():
                self.simulation.run_each()
            
            states,rewards = self.simulation.get_info()
            done =  self.simulation.simulation_check()

            return states,rewards,done

        def observe(self):
            
             states,_ = self.simulation.get_info()