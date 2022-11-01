import simpy
import numpy as np
import scipy

def create_schedule(processors):
    
    max_task = 0
    for processor in processors:
        if len(processor.jobs)>max_task:
            max_task = len(processor.jobs) 
    schedule = np.zeros((max_task,len(processors)))
    for processor in processors:
        jobs = processor.jobs
        processor_vector = np.zeros(max_task)
        for job in jobs:
            processor_vector[job.id]=1
        schedule[processor.id,:]=processor_vector    

    return scipy.sparse.csr_matrix(schedule)


class job:
    
    def __init__(self,processor,id,job_type):
        
        self.processor = processor
        self.job_type = job_type
        self.id = id
    
class processor:
    
    def __init__(self,id,proc_type,resource):
        
        self.id = id
        self.type = proc_type
        self.jobs = []
        self.broke = False
    
    def assign(self,job):

        processor.jobs.append(job)
    
    def process(self,env):


    def processor_break(self,env):



class simulator:

    def __init__(self,n_jobs,n_processors):
        
        self.env = simpy.Environment()
        
         