#replication join for scams.csv and transaction dataset
#1) output scam category with aggregated value from Transaction dataset
#2) output scam category-status with aggregated value from Transaction dataset
#3) output scam status with aggregated value from Transaction dataset

from mrjob.job import MRJob
from mrjob.step import MRStep
import re


class partd(MRJob):
    #scam_list=[]
    scam_table = {}
    #list1=[]
    
    #first step performs replication join of the scam table and the transaction data
    def mapper_join_init(self):
        #generate scam_table with format: scam address:[scamid, status, category]
        #j = 0
        #self.scam_list.clear()
        with open("scams.csv") as f:
            for line in f:
                fields = line.split(',')
                if len(fields)==4:
                    
                    scamid= fields[0]
                    status = fields[1]
                    category = fields[2]
                    addr = fields[3]
                    if '\n' in addr:
                        join_key = re.sub("[^A-Za-z0-9]","",addr)
                    else:
                        join_key = addr
                    
                    list1 = [scamid, status, category]
                    self.scam_table[join_key]= list1
                    
    
    def mapper_repl_join(self, _, line):
        #join the scam_table data with transactions data for the same addresses    
        try:
            #i = 0
            fields = line.split(',')
            if len(fields)==7:          
                to_addr = fields[2]
                if to_addr in self.scam_table:
                    value = float(fields[3])
                    status = self.scam_table[to_addr][1]
                    category = self.scam_table[to_addr][2]
                    yield(status, value)
                    yield(category, value)
                    yield(category+'-'+status, value)
        except:
            pass
    
    def mapper_agg(self,key,value):
        yield(key, value)
    
    def reducer_sum(self, key, values):
        total = sum(values)
        yield(key, total) 
 


    def steps(self):
        return [MRStep(mapper_init=self.mapper_join_init,
        mapper=self.mapper_repl_join),
        MRStep(mapper=self.mapper_agg,
        reducer=self.reducer_sum)
        ]

if __name__ == '__main__':
    partd.run()