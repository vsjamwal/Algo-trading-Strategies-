#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np 
import pandas as pd 
import pandas_datareader as pdr 
import matplotlib.pyplot as plt 
from datetime import datetime


# In[38]:


#download data 
AAPL = pdr.get_data_yahoo("AAPL",datetime(2021,1,1))
AAPL.head()
day = np.arange(1, len(AAPL) +1)
AAPL["day"]  = day 
AAPL.drop(columns=["Adj Close"], inplace = True)
AAPL = AAPL[["day","Open","High","Low", "Close", "Volume"]]
AAPL.head()


# In[39]:


AAPL.info()


# In[40]:


#moving avrage data frame 
AAPL['7-day'] = AAPL['Close'].rolling(7).mean()
AAPL['21-day'] = AAPL['Close'].rolling(21).mean()
AAPL['1-day'] = AAPL['Volume']
AAPL['10-day'] = AAPL['Volume'].rolling(10).mean()

AAPL.dropna(inplace=True)
AAPL


# In[41]:


#signal

AAPL['signal'] = np.where(AAPL['7-day'] > AAPL['21-day'], 1, 0)
AAPL['signal'] = np.where(AAPL['7-day'] < AAPL['21-day'], -1, AAPL["signal"])
AAPL.dropna(inplace=True)


# In[42]:


#returns system vs holding 
AAPL['return'] = np.log(AAPL['Close']).diff()
AAPL['system_return'] = AAPL['signal'] * AAPL['return']
AAPL['entry'] = AAPL.signal.diff()
AAPL


# In[43]:


plt.plot(np.exp(AAPL['return']).cumprod(), label='Buy/Hold')
plt.plot(np.exp(AAPL['system_return']).cumprod(), label='System')
plt.legend(loc=2)
plt.grid(True, alpha=.3)


# In[44]:


#overall return 
(np.exp(AAPL['return']).cumprod()[-1] -1 )*100


# In[45]:


(np.exp(AAPL['system_return']).cumprod()[-1] -1)*100


# ##### So our strategy outperformed the market by 36%

# In[46]:


df1 = AAPL
from sklearn import metrics


# In[47]:


x = df1[["Open","High","Low"]]
y = df1["Close"]


# In[48]:


from sklearn.model_selection import train_test_split
x_train,x_test , y_train, y_test= train_test_split(x,y, random_state = 0)


# In[49]:


x_train.shape


# In[50]:


x_test.shape


# In[51]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
regressor = LinearRegression()


# In[52]:


regressor.fit(x_train, y_train)


# In[53]:


print(regressor.coef_)
print(regressor.intercept_)


# In[54]:


predicted = regressor.predict(x_test)


# In[55]:


df  = pd.DataFrame(y_test, predicted)
df.head()


# In[56]:


dfr=pd.DataFrame({"Actual  Price":y_test, "Predicted Price":predicted})
dfr


# In[59]:


#accuracy of prediction
regressor.score(x_test,y_test)


# In[58]:


dfr.head(20).plot()


# In[ ]:





# In[ ]:




