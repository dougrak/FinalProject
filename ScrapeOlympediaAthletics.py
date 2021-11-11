#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
from urllib.parse import urljoin


# In[2]:


base_site = 'https://www.olympedia.org/editions'


# In[3]:


response = requests.get(base_site)
response.status_code


# In[4]:


html = response.content
soup = BeautifulSoup(html, 'lxml')

#create file with the games list
with open('Olympics_Games_List_lxml.html', 'wb') as file:
    file.write(soup.prettify('utf-8'))


# In[5]:


#Find the summer games list
seasons = soup.find_all('table', {'class': 'table-striped'})


# In[6]:


#Find all links in html using anchor tag
links = seasons[0].find_all('a')


# In[7]:


#Grab just the links to different "editions" (years) of the Olympics
editions = []
edition_urls = []
for link in links:
    if (link.get('href').startswith('/editions/')) and (link.get('href') not in editions):
        editions.append(link.get('href'))
        edition_urls = [urljoin(base_site, ed) for ed in editions]
        
#Remove unwanted urls for future games
edition_urls.remove('https://www.olympedia.org/editions/63')
edition_urls.remove('https://www.olympedia.org/editions/64')
edition_urls.remove('https://www.olympedia.org/editions/372')
edition_urls[-1]


# In[8]:


#Visit each of the games' pages, each discipline page, and each of the event pages

tables = pd.DataFrame()

i = 0

#Loop over each games' page to get a list of discipline (sport) urls
for ed in edition_urls:
    ed_response = requests.get(ed)
    
    if ed_response.status_code == 200: #if response, print out confirmation
        print ('URL #{0}: {1}'.format(i+1, ed))
        i += 1
    else: #if no response, skip and continue
        print('Status code {0}: Skipping URL#{1}: {2}'.format(ed_response.status_code, i+1, ed))
        i +=1
        continue
    
    ed_html = ed_response.content
    ed_soup = BeautifulSoup(ed_html, 'lxml')
    
    #Create year variable to add to end table
    year = ed_soup.find('h1').text[0:4]
    
    #Look for tables
    if ed_soup.find('table', {'class': 'table-striped'}):
        disciplines = ed_soup.find('table', {'class': 'table-striped'})
    else:
        print ("No table for {0}".format(ed))
        continue
    
    if disciplines.find('a'):
        #Look for athletics
        for d in disciplines.find_all('a'):
            if d.text == 'Athletics':
                athletics_link = d.get('href')
    else:
        continue
    
    athletics_url = urljoin(base_site, athletics_link)
    
    #Query athletics
    athletics_response = requests.get(athletics_url)
    
    #If no response, declare, skip and continue
    if athletics_response.status_code != 200: 
        print('Status code {0}: Skipping URL#{1}: {2}'.format(athletics_response.status_code, i+1, athletics_url))
        continue
    
    #Create athletics soup
    athletics_html = athletics_response.content
    athletics_soup = BeautifulSoup(athletics_html)
    
    #Search for and create events table
    if athletics_soup.find('table', {'class': 'table-striped'}):
        events = athletics_soup.find_all('table', {'class': 'table-striped'})
    else:
        print ("No table for {0}".format(discipline))
        continue
    
    #Create urls for events and weed out unwanted events
    events_links = events[0].find_all('a')
    unwanted = ('hampion', 'tanding', 'elay', 'team', 'both', 'athlon')
    
    events_urls = []
    for e in events_links:
        if any(u in e.text for u in unwanted) is True: 
            pass
        else:
            events_urls.append(urljoin(base_site, e.get('href')))

    #Loop over event_urls to visit each page
    for event in events_urls:
        #create event soup
        event_response = requests.get(event)
        event_html = event_response.content
        event_soup = BeautifulSoup(event_html, 'lxml')
        
        
        #Create event_name and gender variables for end table
        event_gender = event_soup.find('h1').text.split(', ')
        event_name = event_gender[0]
        if event_gender[1] == 'Men':
            gender = 'M'
        else:
            gender = 'F'

        #Loop over each event page to pull tables of athletes
        if event:
            table = pd.read_html(event)
            table = table[1]

            #Preliminary data cleaning
            
            #Identify and rename medal column
            #Some events do not have gold
            if len(table.columns[table.loc[0].isin(['Gold'])].values.tolist()) > 0                or len(table.columns[table.loc[0].isin(['Silver'])].values.tolist()) > 0:
                try:
                    table.columns[table.loc[0].isin(['Gold'])].values.tolist()
                    medal_index = table.columns[table.loc[0].isin(['Gold'])].values.tolist()
                    table.rename(columns={medal_index[0]: 'Medal'}, inplace=True)
                except IndexError:
                    try: 
                        table.columns[table.loc[0].isin(['Silver'])].values.tolist()
                        medal_index = table.columns[table.loc[0].isin(['Silver'])].values.tolist()
                        table.rename(columns={medal_index[0]: 'Medal'}, inplace=True)
                    except IndexError:
                        print('{0} failed to scrape'.format(event))

            #Remove all but medalists
            table.dropna(axis=0, subset=['Medal'], inplace=True)

            #Remove Time column if final is also present
            if 'Final' in table.columns and 'Time' in table.columns:
                table.drop(columns=['Time'], inplace=True)
            
            #Rename Pos
            table.rename(columns={'Pos': 'Position'}, inplace=True)
     
            #Add new columns
            table['Event'] = event_name
            table['Gender'] = gender
            table['Year'] = year
            
            #Drop columns not in list
            cols = ['Position', 'Athlete', 'NOC', 'Final', 'Distance', 
                    'Height', 'Time', 'Medal', 'Event', 'Gender', 'Year', 'Time (H)']
            table = table[table.columns.intersection(cols)]
            
            #Append event table to tables
            tables = tables.append(table)
            
        else:
            continue
    
    #Indicate year is done
    print('URL #{0} complete'.format(i))


# In[9]:


tables.to_csv('Olympics_Athletics_Results.csv', index = False, header = True)


# In[ ]:




