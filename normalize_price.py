import pandas as pd
import numpy as np
import csv


prices=np.asarray(pd.read_csv('DJIA_table.csv')['Close'])[::-1]
print (prices[0])

return0=(11431.43-11656.07)/11656.07
return1=(11734.32-11431.43)/11431.43

returns=[]
returns.append(return0)
returns.append(return1)

for i in range(len(prices)-2):
    temp=(prices[i+1]-prices[i])/prices[i]
    returns.append(temp)


maxr=max(returns[0:1611])
minr=min(returns[0:1611])
ndr=[]
for i in range(len(returns)):
    temp=(2*returns[i]-minr-maxr)/(maxr-minr)
    ndr.append(temp)

print (len(returns))
print (len(ndr))
returns=np.reshape(np.asarray(returns),(-1,1))
ndr=np.reshape(np.asarray(ndr),(-1,1))


OUTPUT_PATH='returns.csv'

print('Writing results to {}'.format(OUTPUT_PATH))
with open (OUTPUT_PATH,'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    writer.writerow(['returns'])#in all 64 emojis
    for i,row in enumerate(returns):
        try:
            writer.writerow(row)
        except:
            print("Exception at row {}!".format(i))


OUTPUT_PATH='ndr.csv'

print('Writing results to {}'.format(OUTPUT_PATH))
with open (OUTPUT_PATH,'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    writer.writerow(['ndr'])#in all 64 emojis
    for i,row in enumerate(ndr):
        try:
            writer.writerow(row)
        except:
            print("Exception at row {}!".format(i))


print (returns[0])