import numpy as np
import pandas as pd
import csv


data = np.array(pd.read_csv('scores1.csv')['scores']).reshape(-1,25)


sum25=[]
max25=[]
min25=[]
temp=[]

print (len(data)/25)
for i in range(1989):
    temp.append(sum(data[i]))
    temp.append(max(data[i]))
    temp.append(min(data[i]))

temp=np.reshape(temp,(-1,3))


OUTPUT_PATH='sum_max_min_25.csv'
print('Writing results to {}'.format(OUTPUT_PATH))
with open (OUTPUT_PATH,'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    writer.writerow(['sum','max','min'])#in all 64 emojis
    for i, row in enumerate(temp):
        try:
            writer.writerow(row)
        except:
            print("Exception at row {}!".format(i))