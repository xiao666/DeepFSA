import pandas as pd
import numpy as np

data1 = np.array(pd.read_csv('emoji_semeval_1.csv'))
data2 = np.array(pd.read_csv('semeval_1.csv'))[:,2]

'''
data1=np.array(pd.read_csv('microblog_test_64emoji_prob.csv'))
data2 = np.array(pd.read_csv('Microblogs_Testdata_withscores.csv'))[:,1]
for i in range(len(data2)):
    if (data2[i]>=0):
        data2[i]=1
    else:
        data2[i]=0

'''
#temp=top 3 bullish (57,33,6) - top 3 bearish (39,22,46)
#if judge(temp)>=0 bullish

#[39 22 46 51 35 27  5 34 52 45 56  1 43 25 19 37 32 55 14  3 29  0  2 12 38
 #42 13 20 49 26 28 41 48 23 30 58 62 60 59 18 47 50  8 24 11 54  4  9 61 36
 #21 16  7 15 63 10 44 53 40 31 17  6 33 57]

pos_idx=[48,23,30,58,62,60,59,18,47,50,8,24,11,54,4,9,61,6,21,16,7,15,63,10,44,53,40,31,17,6,33,57]

neg_idx=[39 ,22, 46, 51 ,35, 27,  5 ,34 ,52 ,45 ,56 , 1 ,43 ,25, 19 ,37 ,32 ,55, 14 , 3 ,29 , 0 , 2 ,12 ,38 ,42 ,13, 20 ,49 ,26 ,28 ,41]


print (len(neg_idx))

judge=[]
for i in range(len(data1)):
    temp=data1[i]
    #temp2=temp[57]+temp[33]+temp[6]-temp[39]-temp[22]-temp[46] +temp[17]+temp[31]-temp[51]-temp[35]
    temp2=temp[57]+temp[31]-temp[46]-temp[39]
	#temp2=sum(temp[pos_idx[0:3]])-sum(temp[neg_idx[0:3]])
    #temp2=temp.argsort()[-1]
    if (temp2 >=0):
        judge.append(1)
    else:
        judge.append(0)

print (len(judge))

#cal acc
correct=0
for i in range(len(data2)):
    if (data2[i]==judge[i]):
        correct=correct+1

print (correct)
print (correct/len(data2))
#'''