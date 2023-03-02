#this code is to output the scam id, status, category and addresses in a csv table

import json
import csv 
with open('scams.json') as json_file:
    data = json.load(json_file)

result = data["result"]  

keys = []
#dict1={}
list1 = []
list2 = []
for key in result:
 keys.append(key)
for key in keys:
  #print('key',key)
  addr=data["result"][key]["addresses"]
  id = data["result"][key]["id"]
  status = data["result"][key]["status"]
  category = data["result"][key]["category"]
  list1=[id, status, category, addr]
  list2.append(list1)
  #dict1[id] = addr
  #print('addr',addr)
with open('scams.csv','w', newline='') as f:
  writer = csv.writer(f)
  for item in list2:
    if len(item[2]) ==1:
      writer.writerow([item[0],item[1], item[2], item[3][0]])
    if len(item[1])>1:
      for i in range(len(item[3])):
        writer.writerow([item[0],item[1], item[2], item[3][i]])
