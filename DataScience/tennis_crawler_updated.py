import urllib
from bs4 import BeautifulSoup
import pandas as pd
import re
import numpy as np
import random

from selenium import webdriver
import time

CHROME_WEBDRIVER_LOCATION='driver/chromedriver'

#the years for which we will be retrieving data
years=[2012,2013,2014,2015]
#the types of surfaces we will get data for
surfaces=['clay','grass','hard','carpet']

def get_url_data(url):
    driver = webdriver.Chrome(CHROME_WEBDRIVER_LOCATION)
    driver.get(url)
    #wait until the next retrieval so that we don't overload the server and get locked out
    time.sleep(5)
    content = driver.page_source
    driver.close()
    return content

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z

def read_basic(table):
    """
    Read the table with the basic information from ATP
    """
    data={}
    elements=table.find_all('td')
    bday=table.find_all(attrs={'class':'table-birthday'})[0].contents[0].strip()
    bday=bday.replace('(',"").replace(")","")
    data['birthday']=bday
    
    turned_pro=elements[1].find_all(attrs={'class':'table-big-value'})[0].contents[0].strip()
    if len(turned_pro)==0:
        turned_pro=0
    data['turned_pro']=int(turned_pro)
    
    weight=elements[2].find_all(attrs={'class':'table-weight-kg-wrapper'})[0].contents[0].strip()
    weight=weight.replace("(","").replace(")","").replace("kg","")
    data['weight']=int(weight)
    
    height=elements[3].find_all(attrs={'class':'table-height-cm-wrapper'})[0].contents[0].strip()
    height=height.replace("(","").replace(')',"").replace('cm',"")
    data['height']=int(height)
    
    plays=elements[6].find_all(attrs={'class':'table-value'})[0].contents[0].strip()
    data['plays']=plays
    
    return data
        

def read_table(table):
    """
    read a stats table from ATP
    """
    elements=table.find_all('td')
    df={}
    for i in range(0,len(elements)-1,2):
        el1=elements[i].contents[0].strip()
        el2=elements[i+1].contents[0].strip()
        
        if el2.find('%')>-1:
            el2=el2.replace('%',"")
            el2=float(el2)
            el2=0.01*el2
        elif el2.find(',')>-1:
            el2=el2.replace(',',"")
            el2=int(el2)
        
        df[el1]=el2
    return df
        
def extract_urls(body):
    """
    simple function to extract a url from the atpworld tour
    """
    return re.findall("\/en\/players\/\w+-\w+\/\w+\/overview",body)
    
content=get_url_data('http://www.atpworldtour.com/en/stats/leaderboard?page=serve')
links=extract_urls(content)

#Extract the player URLs
#extract the main page
links.append(extract_urls(get_url_data('http://www.atpworldtour.com/en/rankings/singles')))

#extract the rest of the pages
start=100
finish=1000
for i in range(1,5):
    ran=str(start+1)+'-'+str(start+100)
    print('getting ranks:'+ran)
    links.append(extract_urls(get_url_data('http://www.atpworldtour.com/en/rankings/singles/?rankDate=2016-6-13&countryCode=all&rankRange='+ran)))
    start=start+100    
    

#unfold the list and make sure there are no duplicates
links=[item for sublist in links for item in sublist]
links=set(links)
#http://www.atpworldtour.com/en/players/ivo-karlovic/k336/player-stats?year=2014&surfaceType=all

store=[]


#If the .csv file does not exist, then create a new pandas dataframe to store results
try:
    original=pd.read_csv('player_stats.csv')
except:
    original=pd.DataFrame({'name':[np.nan]})

links=random.sample(links,len(links))

#Now access each player page and collect the stats
for l in links:
    #time.sleep(1)
    url='http://www.atpworldtour.com'+l.replace('overview','player-stats?year=%d&surfaceType=%s')
    
    for y in years:
        for surf in surfaces:
            try:
                name=re.findall('[A-Za-z][a-z]+-[A-Za-z][a-z]+',l)[0].replace('-',' ')
                #make sure we have not retrieved this player
                if not np.any(original['name'].values==name):
                    new_url=url % (y,surf)
                    print('\ntrying : '+new_url)
                    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
                    headers={'User-Agent':user_agent,}
                    request=urllib.request.Request(new_url,None,headers)
                    response = urllib.request.urlopen(request)
                    content=response.read()
                    soup=BeautifulSoup(content)    
                    
                    tables=soup.find_all('table')
                    
                    try:
                        print(len(tables))
                        basic_table=tables[0]   
                        basic_data=read_basic(basic_table)    
                    
                        try:
                            service_table=tables[1]
                            service_data=read_table(service_table)
                            ret_table=tables[2]
                            ret_data=read_table(ret_table)    
                        except:
                            pass

                        raw_dict=merge_two_dicts(basic_data,service_data)
                        raw_dict=merge_two_dicts(raw_dict,ret_data)
                        raw_dict=merge_two_dicts(raw_dict,{'name':name,'year':y,'surface':surf})
                        df=pd.DataFrame(raw_dict,index=[name+'.'+str(y)+'.'+surf])
                        store.append(df)
                        final=pd.concat(store)
                        try:
                            final['birthday']=pd.to_datetime(final['birthday'])
                            final['last name']=final['name'].apply(lambda x: x.split(' ')[1].lower())
                        except:
                            pass
                        final.to_csv('player_stats.csv')
                        print('\n retrieved :'+name)
                    except Exception as e:
                        print('\nCOULD NOT GET '+name+':'+str(y)+':'+surf+'\n'+str(e))
            except Exception as e:
                    print('Total exception' + str(e))
                    pass
                    
    
    

