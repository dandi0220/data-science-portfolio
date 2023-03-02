#this part of code is calculating the monthly total number of transactions based on the transaction dataset
 
from mrjob.job import MRJob
from datetime import datetime

class partA(MRJob):

    def mapper(self, _, line):
        try:
            fields = line.split(',')
            if len(fields)== 7: #transaction dataset
                dt = datetime.fromtimestamp(int(fields[6])) #format the time to yyyy/mm
                yrmth =dt.strftime("%Y/%m")
                yield(yrmth,1)

        except:
            pass

    def combiner(self, key, count):
        yield(key,sum(count))

    def reducer(self, key, count):
        yield(key,sum(count))


if __name__ == '__main__':
    partA.run()
