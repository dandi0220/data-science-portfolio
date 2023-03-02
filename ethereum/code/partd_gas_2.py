#output average gas and gas price for 1)monthly basis and 2)over the whole time

from mrjob.job import MRJob
from datetime import datetime

class partd(MRJob):

    def mapper(self, _, line):
        try:
            fields = line.split(',')
            if len(fields)== 7:
                dt = datetime.fromtimestamp(int(fields[6])) #format the time
                yrmth = dt.strftime("%Y/%m") #format time to yy/mm
                gas = float(fields[4])
                gas_price = float(fields[5])
                yield(yrmth, (gas, gas_price, 1))
                yield('avg over the whole period',(gas, gas_price, 1)) #for the avg of gas and gas price for the whole period of data
        except:
            pass


    def reducer(self, key, values):
        count = 0
        gas = 0
        gas_price = 0
        for value in values:
            
            gas += value[0] #sum of gas
            gas_price += value[1]
            count += value[2] #sum of transaction count
        yield(key,([gas/count],[gas_price/count])) #average of gas and gas price for monthly and whole period 


if __name__ == '__main__':
    partd.run()
