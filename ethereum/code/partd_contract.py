from mrjob.job import MRJob
#join transaction dataset and contract datset for the common address
#output the aggreated value, no. of transactions and no. of from_addresses


class partd(MRJob):
    def mapper(self, _, line):
        try:
            fields = line.split(',')

            if len(fields)== 5: #contracts data
                join_key = fields[0] #address
                yield(join_key, (1,1))
            
            elif len(fields)== 7: # transaction data
                join_key = fields[2] #to_address
                join_value = (float(fields[3]), 1, fields[1]) #emit value,1,from_address
                yield(join_key, (join_value,2))
        except:
            pass


    def reducer(self, key, join_values):
        smart=False #indicator for judging if the address is in the contract dataset
        value=0
        transactions=0
        from_addr=[]
        for join_value in join_values:
            if join_value[1]==1: #contract indicator
                smart = True
            if join_value[1]== 2: #transaction indicator
                value += join_value[0][0]
                transactions += join_value[0][1]
                from_addr.append(join_value[0][2])
        if smart and value>0:
            yield(key,(value, transactions, len(from_addr)))


if __name__=='__main__':
    partd.run()