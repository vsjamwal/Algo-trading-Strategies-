#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np 
import pandas as pd 
import pandas_datareader as pdr 
import matplotlib.pyplot as plt 
from datetime import datetime


# In[35]:


#download data 
gld = pdr.get_data_yahoo("GLD",datetime(2021,1,1))
gld.head()
day = np.arange(1, len(gld) +1)
gld["day"]  = day 
gld.drop(columns=["Adj Close", "Volume"], inplace = True)
gld = gld[["day","Open","High","Low", "Close"]]
gld.head()
        


# In[36]:


gld.info()


# In[37]:


#moving avrage data frame 
gld['9-day'] = gld['Close'].rolling(9).mean()
gld['21-day'] = gld['Close'].rolling(21).mean()
gld


# In[38]:


#signal
gld['signal'] = np.where(gld['9-day'] > gld['21-day'], 1, 0)
gld['signal'] = np.where(gld['9-day'] < gld['21-day'], -1, gld['signal'])
gld.dropna(inplace=True)
gld


# In[39]:


#returns system vs holding 
gld['return'] = np.log(gld['Close']).diff()
gld['system_return'] = gld['signal'] * gld['return']
gld['entry'] = gld.signal.diff()
gld


# In[40]:


plt.plot(np.exp(gld['return']).cumprod(), label='Buy/Hold')
plt.plot(np.exp(gld['system_return']).cumprod(), label='System')
plt.legend(loc=2)
plt.grid(True, alpha=.3)


# In[41]:


#overall return 
(np.exp(gld['return']).cumprod()[-1] -1 )*100


# In[42]:


(np.exp(gld['system_return']).cumprod()[-1] -1)*100


# So our strategy outperformed the market by 10 fold 

# In[ ]:




