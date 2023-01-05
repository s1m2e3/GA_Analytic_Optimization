import simpy
import numpy as np
import scipy
import random

def time_per_machine():
    return random.expovariate(1)

class Job:
    
    def __init__(self,id,job_type,processor=0,):
        
        self.processor = processor
        self.job_type = job_type
        self.id = id
    
class Processor:
    
    def __init__(self,env,id,proc_type,resource):
        
        self.env = env
        self.id = id
        self.type = proc_type
        self.queue = []
        self.queue_length = len(self.queue) 
        self.resource = resource
        self.process = self.env.process(self.work(self.queue,self.resource))
        self.total_tasks = 0
    
    
    def assign(self,job):
        
        self.queue.append(job)
            
        if job.processor == 0:
            job.processor = self.id
    
    def work(self,queue,resource):

        try:
            for job in queue:
                queue = queue[1:]
                starts = self.env.now
                done_in = time_per_machine()
                print('%f seconds for job: %i at machine %i' % (done_in,job.id, self.id))
                print("number of jobs in queue: %i at machine %i" % (self.queue_length,self.id))
                with resource.request() as req:
                    yield req
                    yield self.env.timeout(done_in)
                    finished = self.env.now
                    print('%7.4f job: %i finishes at machine %i' % (finished,job.id, self.id))
                    self.total_tasks += 1
                    self.queue_length -= 1
                    print("remaining jobs in queue: %i at machine %i" % (self.queue_length,self.id))

        except simpy.Interrupt:
                    print("except")
                    


class MachineShopSimulator:

    def __init__(self,n_jobs,n_processors):
        
        self.env = simpy.Environment()
        self.n_jobs = n_jobs
        self.n_processors = n_processors
        self.start = False
        self.processors = [Processor(self.env,i, 1,simpy.Resource(self.env, capacity=1)) for i in range(self.n_processors)]
        self.jobs = [Job(i, 1) for i in range(self.n_jobs)]
        
    def assign(self,action):
        self.processors[action[0]].assign(self.jobs[action[1]])
        self.processors[action[0]].queue_length += 1
    
    def simulation_check(self):
        accumulated = 0
        for processor in self.processors:
            accumulated +=processor.total_tasks
        return accumulated == self.n_jobs
    
    def run(self):
        finished  = False
        while not finished:
            #self.env.run()
            self.env.step()
            states,rewards=self.get_info()
            print(states,rewards)
            finished = self.simulation_check()

    def run_each(self):
        finished  = False
        if not finished:
            #self.env.run()
            self.env.step()
            finished = self.simulation_check()
    
    def get_info(self):
        states = [processor.queue_length for processor in self.processors]
        rewards = 0
        total_tasks = [processor.total_tasks for processor in self.processors]
        states= states + total_tasks
        return states,rewards

    def check_sim(self):
        queue = [processor.queue_length for processor in self.processors]
        total_tasks = [processor.total_tasks for processor in self.processors]
        check = False
        if min(queue)>=1 or sum(total_tasks)>self.n_jobs-self.n_processors:
            check=True
        #print(check)
        return check
    
    def reset(self):
        self.__init__(self.n_jobs,self.n_processors) 

## create jobs,machines and simulation
            
n_jobs = 20
n_machines = 3
jobs = np.arange(n_jobs)
machines = np.arange(n_machines)
actions = [(random.choice(machines),job_i) for job_i in jobs]
simulation = MachineShopSimulator(n_jobs,n_machines)


for action in actions:
    simulation.assign(action)
    if simulation.check_sim():
        simulation.run_each()
        states,rewards=simulation.get_info()
        print(states,rewards)
    else:
        print("assigned not run")
# if simulation.check_sim():
#     simulation.run()
#simulation.run()


        


        
        
        
         
                    