import simpy
import numpy as np
import scipy
import random



def time_per_machine():
    return random.expovariate(1)

class job:
    
    def __init__(self,id,job_type,processor=0,):
        
        self.processor = processor
        self.job_type = job_type
        self.id = id
    
class processor:
    
    def __init__(self,env,id,proc_type,resource):
        
        self.env = env
        self.id = id
        self.type = proc_type
        self.queue =[]
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
                #print('%f seconds for job: %i at machine %i' % (done_in,job.id, self.id))
                with resource.request() as req:
                    yield req
                    yield self.env.timeout(done_in)
                    finished = self.env.now
                    #print('%7.4f job: %i finishes at machine %i' % (finished,job.id, self.id))
                    self.total_tasks += 1
        
        except simpy.Interrupt:
                    print("except")
                    


class simulator:

    def __init__(self,n_jobs,n_processors):
        
        self.env = simpy.Environment()
        self.n_jobs = n_jobs
        self.n_processors = n_processors
        self.processors = [processor(self.env,i, 1,simpy.Resource(self.env, capacity=1)) for i in range(self.n_processors)]
        self.jobs = [job(i, 1) for i in range(self.n_jobs)]

            
    def assign(self,action):
        self.processors[action[0]].assign(self.jobs[action[1]])
    
    def simulation_check(self):
        accumulated = 0
        for processor in self.processors:
            accumulated +=processor.total_tasks
        return accumulated == self.n_jobs
    
    def run(self):
        finished  = False
        while not finished:
            self.env.step()

            finished = self.simulation_check()
        

## create jobs,machines and simulation
            
n_jobs = 20
n_machines = 3
jobs = np.arange(n_jobs)
machines = np.arange(n_machines)
actions = [(random.choice(machines),job_i) for job_i in jobs]
simulation = simulator(n_jobs,n_machines)
for action in actions:
    simulation.assign(action)
simulation.run()


        


        
        
        
         
                    