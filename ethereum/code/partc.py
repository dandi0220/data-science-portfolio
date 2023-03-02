#this code is based on block dataset
#1st step: output miner with its aggregated size
#2nd step: find the top 10 miner
from mrjob.job import MRJob
from mrjob.step import MRStep


class partc(MRJob):
    def steps(self):
        return [
            MRStep(mapper=self.mapper_1,
                   combiner=self.combiner_1,
                   reducer=self.reducer_1),
            MRStep(mapper=self.mapper_top10,
                   combiner=self.combiner_top10,
                   reducer=self.reducer_top10)
        ]


    def mapper_1(self, _, line):
        try:
            fields = line.split(',')
            if len(fields)== 9: #block data
                miner = fields[2]
                size = int(fields[4])
                yield(miner,size)
        except:
            pass

    def combiner_1(self, key, values):
        total = sum(values)
        yield(key,total)

    def reducer_1(self, key, values):
        total = sum(values)
        yield(key,total)
    
    def mapper_top10(self,key,value):
        yield("top",(key,value))

    def combiner_top10(self,key,values):
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
    partc.run()
