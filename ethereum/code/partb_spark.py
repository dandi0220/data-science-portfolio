import pyspark

sc = pyspark.SparkContext()

def good_line_trans(line):
    try:
        fields = line.split(',')
        if len(fields) !=7: #check if transaction data
            return False
        float(fields[3])  #transaction value
        return True

    except:
        return False
    
def good_line_contracts(line):
    try:
        fields = line.split(',')
        if len(fields)!=5: #check if contract data
            return False
        
        return True
    except:
        return False

contr_lines = sc.textFile("/data/ethereum/contracts")
clean_contr_lines = contr_lines.filter(good_line_contracts)
addr = clean_contr_lines.map(lambda l:(l.split(',')[0],0))

trans_lines = sc.textFile("/data/ethereum/transactions")
clean_trans_lines = trans_lines.filter(good_line_trans)
features = clean_trans_lines.map(lambda l:(l.split(',')[2], float(l.split(',')[3]))).reduceByKey(lambda a,b: a+b)

join_table = addr.join(features)
top10 = join_table.takeOrdered(10, key= lambda x: -x[1][1])

for item in top10:
    print("{}-{}".format(item[0], item[1][1]))
