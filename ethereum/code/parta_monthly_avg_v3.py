#this code is for calculating the monthly average value of transactions based on the transaction dataset


from mrjob.job import MRJob
from datetime import datetime

class partA(MRJob):

    def mapper(self, _, line):
        try:
            fields = line.split(',')
            if len(fields)== 7:
                dt = datetime.fromtimestamp(int(fields[6])) #format the time
                yrmth = dt.strftime("%Y/%m") #format time to yy/mm
                tran_value = int(fields[3])
                yield(yrmth, (tran_value,1))
        except:
            pass

    def combiner(self, key, values):
        count = 0 #sum of no. of transactions
        total = 0 #sum of total transaction values
        for value in values:
            count += value[1] #sum of no. of transactions
            total += value[0] #sum of total transaction values
        yield(key,[total,count])

    def reducer(self, key, values):
        count = 0
        total = 0
        for value in values:
            count += value[1]
            total += value[0]
        yield(key,[total/count]) #average of value of transactions


if __name__ == '__main__':
    partA.run()
