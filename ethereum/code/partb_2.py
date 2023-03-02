#this file is to find the top 10 address/value pairs (in the order of value) for addresses in both Transaction and Contrct Dataset
#1st step: address and its aggregated value 
#2nd step: find the top 10 address/value pairs based on sorting to the values 

from mrjob.job import MRJob
from mrjob.step import MRStep


class partB(MRJob):
    def steps(self):
        return [
            MRStep(mapper=self.mapper_join,
                   reducer=self.reducer_join),
            MRStep(mapper=self.mapper_top10,
                   combiner=self.combiner_top10,
                   reducer=self.reducer_top10)
        ]
    
    
    def mapper_join(self, _, line):
        try:
            fields = line.split(',')
            if len(fields)== 7: # transaction data
                join_key = fields[2] #to_address
                tran_value = float(fields[3])
                yield(join_key, (tran_value,1))
            
            elif len(fields)== 5: #contracts data
                join_key = fields[0] #address
                yield(join_key, (1,2))
        except:
            pass


    def reducer_join(self, key, values):
        total = 0
        smart = False
        for value in values:
            if value[1]==1: #transaction data
                total += value[0]
            if value[1]==2: #contracts data
                smart = True
        if total>0 and smart:
            yield(key,total)
        
    def mapper_top10(self,key,value):
        yield("top",(key,value))
    
    def combiner_top10(self, _, values):
        sorted_values = sorted(values, reverse=True, key=lambda x:x[1])
        i=0
        for value in sorted_values:
            yield("top",value)
            i +=1
            if i>=10:
                break
    
    def reducer_top10(self, _, values):
        sorted_values = sorted(values, reverse=True, key=lambda x:x[1])
        i=0
        for value in sorted_values:
            yield("{}-{}".format(value[0],value[1]),None)
            i+=1
            if i>=10:
                break

if __name__ == '__main__':
    partB.run()
