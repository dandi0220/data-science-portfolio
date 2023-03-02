#the first mrjob: aggregate transaction and contract data with the common address, 
# and output each address' aggreated (value, gas, no. of transactions)
#the second mrjob: rank the top 10 in order of value and output in the format <address>-<value>-<avg gas>"

from mrjob.job import MRJob
from mrjob.step import MRStep


class partd(MRJob):
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
                gas = float(fields[4])
                join_value = (tran_value, gas, 1) #1 is used for no. of transaction later on
                yield(join_key, (join_value,1)) # 1 is the indicator for transaction dataset
            
            elif len(fields)== 5: #contracts data
                join_key = fields[0] #address
                yield(join_key, (1,2)) #2 is the indicator for contract dataset
        except:
            pass


    def reducer_join(self, key, join_values):
        tran_count = 0
        tran_value = 0
        gas = 0
        smart = False
        for join_value in join_values:

            if join_value[1]==1: #transaction data
                tran_value += join_value[0][0]
                gas += join_value[0][1]
                tran_count += join_value[0][2]

            if join_value[1]==2: #contracts data
                smart = True
        if tran_value>0 and smart:
            yield(key,(tran_value, gas, tran_count))
        
    def mapper_top10(self,addr,value):
        yield("top", (addr,value)) #value is the aggregated (tran_value, gas, tran_count)
    
    def combiner_top10(self, _, values):
        sorted_values = sorted(values, reverse=True, key=lambda x:x[1][0]) #sort by tran_value
        i=0
        for value in sorted_values:
            yield("top",value) #value: (addr,(tran_value, gas, tran_count))
            i +=1
            if i>=10:
                break
    
    def reducer_top10(self, _, values):
        sorted_values = sorted(values, reverse=True, key=lambda x:x[1][0]) #sort by tran_value
        i=0
        for value in sorted_values:
            avg_gas = value[1][1]/value[1][2]
            yield("{}-{}-{}".format(value[0], value[1][0], avg_gas),None) #output top 10 address-tran_value-avg_gas
            i+=1
            if i>=10:
                break

if __name__ == '__main__':
    partd.run()
