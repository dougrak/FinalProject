#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#Load csv data
tables = pd.read_csv('Olympics_Athletics_Results2.csv')


# In[3]:


tables.head(10)


# In[4]:


#Clean data

#Drop unwanted rows (those who didn't place and athletes without a name)
tables.dropna(axis=0, subset=['Position'], inplace=True)
tables.dropna(axis=0, subset=['Athlete'], inplace=True)

#drop 2020 rows as they are empty (the IOC has not posted results)
tables = tables[tables['Year'] != 2020]
tables


# In[5]:


#Drop duplicates and reset index
tables.drop_duplicates(ignore_index=True, inplace=True)


# In[6]:


tables.tail(60)


# In[7]:


#Remove = (for ties)
if 'Position' in tables.columns:
    if tables['Position'].dtype == 'object':
        tables['Position'] = tables['Position'].str.replace('=', '')

#Remove w from Final
if 'Final' in tables.columns:
    if tables['Final'].dtype == 'object':
        tables['Final'] = tables['Final'].str.replace('w', '')
    
#Remove - from marathon times
if 'Time' in tables.columns:
    if tables['Time'].dtype == 'object':
        tables['Time'] = tables['Time'].str.replace('-', ':')


# In[8]:


all_events = tables['Event'].unique().tolist()
all_events


# In[9]:


#Events to keep
races = ('100 metres', '200 metres', '400 metres', '800 metres', 
          '1,500 metres', '5,000 metres', '10,000 metres', 'Marathon')

distances = ('Long Jump', 'Triple Jump', 'Shot Put', 'Discus Throw', 
             'Hammer Throw', 'Javelin Throw')

heights = ('High Jump', 'Pole Vault')

wanted_events = races + distances + heights

not_events = list(set(all_events).difference(wanted_events))
not_events


# In[10]:


tables.tail(60)


# In[11]:


#Examine times to find what is not NaN
not_nan = []
for idx, row in tables.iterrows():
    if tables.loc[idx, 'Time'] != 'NaN'       and tables.loc[idx, 'Time'] not in not_nan:
        not_nan.append(tables.loc[idx, 'Time'])

print(not_nan)


# In[12]:


tables.tail(60)


# In[13]:


#Remove unwanted events
tables = tables[~tables.Event.isin(not_events)]
tables.tail(60)


# In[14]:


#Iterate over rows to clean and swap columns
for idx, row in tables.iterrows():
    
    #Remove strange strings from Final and Time. Replace with NaNs
    nono_str = ['est', 'e', 'â', 'close', 'at', 'nan']
    for s in nono_str:
        if s in str(tables.loc[idx, 'Final']):
            tables.loc[idx, 'Final'] = 'NaN' 
        if s in str(tables.loc[idx, 'Time']):
            tables.loc[idx, 'Time'] = 'NaN'
    tables.loc[idx, 'Final'] = str(tables.loc[idx, 'Final']).split()[0]
        
    #Transfer Final values to appropriate columns
    for r in races:
        if (tables.loc[idx, 'Event'] == r
               and tables.loc[idx, 'Time'] == 'NaN'):
            tables.loc[idx, 'Time'] = tables.loc[idx, 'Final']
 
    for d in distances:
        if (tables.loc[idx, 'Event'] == d
               and tables.loc[idx, 'Distance'] == 'NaN'):
            tables.loc[idx, 'Distance'] = tables.loc[idx, 'Final']
            
    for h in heights:
        if (tables.loc[idx, 'Event'] == h
               and tables.loc[idx, 'Height'] == 'NaN'):
            tables.loc[idx, 'Height'] = tables.loc[idx, 'Final']
            
#Drop Final column
tables.drop(columns=['Final'], inplace=True)


# In[15]:


tables.head()


# In[16]:


#Reorder columns
tables = tables[['Event', 'Year', 'Gender', 'Position', 'Medal', 
                             'Athlete', 'NOC', 'Time', 'Height', 'Distance']]


# In[17]:


def getSeconds(time_str):
    if time_str.count(':') >= 2:
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif time_str.count(':') == 1:
        m, s = time_str.split(':')
        return int(m) * 60 + float(s)
    elif time_str.count('.') == 1:
        return float(time_str)
    else:
        return float(time_str)


# In[18]:


#Convert time to seconds for easier manipulation in Tableau
for idx, row in tables.iterrows():
    if tables.loc[idx, 'Time'] != 'NaN':
        tables.loc[idx, 'Time'] = getSeconds(tables.loc[idx, 'Time'])


# In[19]:


tables.head(10)


# In[20]:


#Add empty rows for years without competition
add_years = [1916, 1940, 1944]

for year in add_years:
    for event in wanted_events:
        for gender in ['M', 'F']:
            tables.append({'Event' : [event], 'Year' : [year], 
                          'Gender' : [gender], 'Position' : [1],
                          'Medal' : ['Gold'], 'Athlete' : ['NaN'],
                          'NOC' : ['NaN'], 'Time' : ['NaN'],
                          'Height' : ['NaN'], 'Distance' : ['NaN']},
                          ignore_index=True)


# In[21]:


tables[tables['Year'] == 1916]


# In[22]:


tables.dtypes


# In[23]:


tables


# In[24]:


tables.to_csv('Olympics_Athletics_Results_cleaned.csv')


# In[ ]:





# In[ ]:





# In[ ]:




