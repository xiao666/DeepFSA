import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates


prices= np.asarray(pd.read_csv('DJIA_table.csv')['Close'])[::-1]
dates= list(np.asarray(pd.read_csv('DJIA_table.csv')['Date'])[::-1])

print (prices[0])
print (prices[-1])

plt.plot()
#new_ticks=np.linspace(2008)
plt.xticks([0,1610,1988],['2008-08-08','2014-12-31',' 2016-07-01'])
plt.plot(prices)

plt.axvline(x=1610,color='black', linestyle='dashed')
plt.title('DJI close price')  
plt.ylabel('close price')  
#plt.xlabel('date')  
#plt.gcf().autofmt_xdate()

#plt.legend(['train_loss', 'test_loss','train_WCS','test_WCS'], loc='upper left')  #,'train_cos','val_cos'
plt.show() 