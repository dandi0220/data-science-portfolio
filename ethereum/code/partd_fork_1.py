from mrjob.job import MRJob
from datetime import datetime
#observe period: one week before and after the DAO fork event on 20/07/2016
#output daily avg value of transactions, daily total no. of transactions and daily avg gas price of transactions

class partd(MRJob):

    def mapper(self, _, line):
        try:
            fields = line.split(',')
            if len(fields)== 7:
                dt = datetime.fromtimestamp(int(fields[6])) #format the time
                yr = dt.strftime("%Y") #return year
                mth = dt.strftime("%m") #return month
                day = dt.strftime("%d") #return day
                tran_value = float(fields[3])
                gas_price = float(fields[5])

                if yr == '2016' and mth == '07' and 13<int(day)<27:
                    yield([day+mth+yr], (tran_value,1,gas_price))
        except:
            pass

    def combiner(self, date, values):
        count = 0
        total_value = 0
        total_gasprice = 0

        for value in values:
            count += value[1] #sum of daily total no. of transactions
            total_value += value[0] #sum of daily total transaction values
            total_gasprice += value[2] #sum of daily total gas price

        yield(date,(total_value, count, total_gasprice))

    def reducer(self, date, values):
        count = 0
        total_value = 0
        total_gasprice = 0

        for value in values:
            count += value[1] #sum of daily total no. of transactions
            total_value += value[0] #sum of daily total transaction values
            total_gasprice += value[2] #sum of daily total gas price

        yield(date,(total_value/count, count, total_gasprice/count)) #emit daily avg value, daily count of transactions, daily avg gas price


if __name__ == '__main__':
    partd.run()
