#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import pandas_datareader as pdr 
import matplotlib.pyplot as plt 
from datetime import datetime


# In[2]:


#download data 
gld = pdr.get_data_yahoo("AAPL",datetime(2021,1,1))
gld.head()
day = np.arange(1, len(gld) +1)
gld["day"]  = day 
gld.drop(columns=["Adj Close", "Volume"], inplace = True)
AAPL = gld[["day","Open","High","Low", "Close"]]
AAPL.head()


# In[3]:


AAPL.info()


# In[6]:


#moving avrage data frame 
AAPL['7-day'] = AAPL['Close'].rolling(9).mean()
AAPL['21-day'] = AAPL['Close'].rolling(21).mean()
AAPL


# In[7]:


#signal
AAPL['signal'] = np.where(AAPL['9-day'] > AAPL['21-day'], 1, 0)
AAPL['signal'] = np.where(AAPL['9-day'] < AAPL['21-day'], -1, AAPL['signal'])
AAPL.dropna(inplace=True)
AAPL


# In[9]:


#returns system vs holding 
AAPL['return'] = np.log(AAPL['Close']).diff()
AAPL['system_return'] = AAPL['signal'] * AAPL['return']
AAPL['entry'] = AAPL.signal.diff()
AAPL


# In[11]:


plt.plot(np.exp(AAPL['return']).cumprod(), label='Buy/Hold')
plt.plot(np.exp(AAPL['system_return']).cumprod(), label='System')
plt.legend(loc=2)
plt.grid(True, alpha=.3)


# In[12]:


#overall return 
(np.exp(AAPL['return']).cumprod()[-1] -1 )*100


# In[13]:


(np.exp(AAPL['system_return']).cumprod()[-1] -1)*100


# ##### So our strategy outperformed the market by 30%

# In[ ]:




