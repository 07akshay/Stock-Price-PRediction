#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


file = 'Downloads/tata-motors_news.csv'
df = pd.read_csv(file)
df.columns=['S.No.','Date','News']
df = df.sort_values(by ='Date')
df = df.drop(columns = ['S.No.'])
df.reset_index(drop =True,inplace =True)
df['Date'] = df['Date'].str.split(' ').str[0]
df


# In[3]:


import nltk
nltk.download('vader_lexicon')


# In[4]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[5]:


sid = SentimentIntensityAnalyzer()
df['score'] = df['News'].apply(lambda News:sid.polarity_scores(News))
df['net_score'] = df['score'].apply(lambda score:score['compound'])
df


# In[6]:


df_new = df.drop(columns =['score','News'])
df_new
df_grp = df_new.groupby('Date')
df_mean = df_grp.mean()
df_f = df_mean.reset_index(drop =True,inplace = True)


# In[7]:


import datetime
maxCol=lambda x: max(x.min(), x.max(), key=abs)
df_l = df_new.groupby('Date',as_index = False)['net_score'].apply(maxCol)
df_l['Date'] =  pd.to_datetime(df_l['Date'])
df_l.set_index('Date',inplace = True)

df_l= df_l.resample('D').sum()
final = []
for value in df_l["net_score"]: 
    if value == 0 and flag==0: 
        final.append(fac*val)
        if fac >= 0.2:
            fac = fac-0.1
        else:
            fac=0
    elif value == 0 and flag ==1:
        final.append(fac*val)
        flag=0
        fac=0.8
    else:
        val = value
        final.append(value)
        flag=1
        fac=1.2
df_l['final_sentiment'] = final
df_l['rescaled_sentiment'] = (df_l['final_sentiment'] +1)/2
df_l.drop(['net_score'],axis =1,inplace= True)
df_l.drop(['final_sentiment'],axis =1,inplace= True)
df_l


# In[8]:


df_l.to_csv('tata-motors_senti.csv', index=True)


# In[9]:


gb = pd.read_csv('tata-motors_senti.csv')
gb


# In[ ]:




