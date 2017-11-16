import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

data1 = np.array(pd.read_csv('emoji_semeval_1.csv'))
data2 = np.array(pd.read_csv('semeval_1.csv'))[:,2]

bullish_idx=[]
bearish_idx=[]
for i in range(len(data2)):
    if (data2[i]==1):
        bullish_idx.append(i)
    else:
        bearish_idx.append(i)

bull_emoji=data1[bullish_idx]#1119
bear_emoji=data1[bearish_idx]#581

sum_bull=[]
for i in range(64):
    temp=sum(bull_emoji[:,i])/1119
    sum_bull.append(temp)

sum_bear=[]
for i in range(64):
    temp=sum(bear_emoji[:,i])/581
    sum_bear.append(temp)

difference=[]
for i in range(64):
    temp=sum_bull[i]-sum_bear[i]
    difference.append(temp)

difference=np.asarray(difference)
largest=difference.argsort()
print (largest)

percentage_bull=[]
percentage_bear=[]
sum_sum_bull=sum(sum_bull)
sum_sum_bear=sum(sum_bear)
print (sum_sum_bull)
for i in range(64):
    temp=sum_bull[i]/sum_sum_bull*100
    temp2=sum_bear[i]/sum_sum_bear*100
    percentage_bull.append(temp)
    percentage_bear.append(temp2)

#print (np.arange(64))
#print (percentage_bull)
#print (percentage_bear)

'''
#print percentage
plt.bar(range(len(sum_bull)),percentage_bull)
plt.title("Percentage of each emoji[bullish]")
plt.show()

plt.bar(range(len(sum_bear)),percentage_bear)
plt.title("Percentage of each emoji[bearish]")
plt.show()
'''

'''
plt.bar(range(len(sum_bull)),sum_bull)
plt.title("Sum of probability of each emoji[bullrish]")
plt.show()

plt.bar(range(len(sum_bear)),sum_bear)
plt.title("Sum of probability of each emoji[bearish]")
plt.show()
'''

'''
#two histo in one graph
x=np.arange(64)
total_width=0.8
n=2
width=total_width/n
plt.bar(x,sum_bull,width=width,label='bullish',fc = 'g')
plt.bar(x+width,sum_bear,width=width,label='bearish', fc = 'r')
plt.title("Sum of probability of each emoji")
plt.legend()
plt.show()
'''

'''
#can not use covered histo!!
x=np.arange(64)
plt.bar(x,sum_bull,label='bullish',fc = 'g')
plt.bar(x,sum_bear,bottom=sum_bull,label='bearish', fc = 'r')
plt.title("Scaled sum of probability of each emoji")
plt.legend()
plt.show()
'''

'''
#Difference of sum of prob. of each emoji
plt.bar(range(len(difference)),difference)
plt.title("Difference of scaled sum of prob. of each emoji [bullish-bearish]")
plt.show()
'''
