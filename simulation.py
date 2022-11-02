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
        self.job = 0
        self.queue =[]
        self.resource = resource
        self.broke = False
        self.process = self.env.process(self.work())
    
    
    def assign(self,job):
        
        if self.job ==0:
            self.job=job

        else:
            self.queue.append(job)

        if job.processor == 0:
            job.processor = self.id
    
    def work(self):

        while True:
            done_in = time_per_machine()
            print(done_in)
            while done_in:
                try:
                    # Working on the part
                    start = self.env.now
                    yield self.env.timeout(done_in)
                    done_in = 0 # Set to 0 to exit while loop.
                    print("machine with id :",self.id)
                    print("finished job :",self.job.id)
                    if len(self.queue)>0:
                        self.job=self.queue[0]
                    
                    
                except simpy.Interrupt:
                    print("except")
                    done_in -= self.env.now - start  # How much time left?

'''
Create simulator of machine shop. Receives schedule, in a sparse matrix.
Number of rows for schedule is the number of processors(machines).
Number of columns is the number of jobs, x_ij=1 means job j  is assigned to machine i 
'''

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
        check = False
        tasks = []
        for processor in self.processors:
            if processor.job==0:
                tasks.append(0)
            else:
                tasks.append(1)
        if min(tasks)!=0:
            check=True
        return check
    
    def run(self):
        
        if self.simulation_check():
            self.env.step()
           # print(self.env.active_process)


## create jobs,machines and simulation
            
n_jobs = 20
n_machines = 3
jobs = np.arange(n_jobs)
machines = np.arange(n_machines)
actions = [(random.choice(machines),job_i) for job_i in jobs]
print(actions)
## create tuple pairs
simulation = simulator(n_jobs,n_machines)
for action in actions:
    simulation.assign(action)
    simulation.run()


        


        
        
        
         
                    