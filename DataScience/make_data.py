import pandas as pd
from random import random
import numpy as np

#player=pd.read_csv('player_stats.csv')
#player['birthday']=pd.to_datetime(player['birthday'])
files=[2015,2016,2017,2018]
missing=[]    
alldata=[]

for f in files:
    games=pd.read_csv(str(f)+'.csv')
    #keep only completed games
    #games=games[games['Comment']=='Completed']
    games['Date']=pd.to_datetime(games['Date'])
   
    #games.loc[(games['FTR'] == 'H'), 'winner'] = 1
    #games.loc[(games['FTR'] == 'A'), 'winner'] = -1
    #games.loc[(games['FTR'] == 'D'), 'winner'] = 0

    #games['Winner'] = games['FTR'].apply(lambda str: f(str))
    #if games['FTR']
    #turn surface to lowrcase
    #games['Surface']=games['Surface'].apply(str.lower)
    alldata.append(games)
    # for index,row in games.iterrows():
    #     print(index)
    #     dummy={}
        
    #     #dummy['month']=row['Date'].month
        
    #     #get the players' names
    #     win_draw_lose=row['FTR']
    #     #winner_name=row['Winner'].split(' ')[0].lower()
    #     #loser_name=row['Loser'].split(' ')[0].lower()
        
    #     #winner_data=player[player['last name']==winner_name]
    #     #loser_data=player[player['last name']==loser_name]
        
    #     #check if the player exists in the crawled data file
    #     #if len(winner_data)==0:
       
    #     #     winner_data['rank']=row['WRank']
    #     #     loser_data['rank']=row['LRank']
            
    #     #     winner_data['B365']=row['B365W']
    #     #     loser_data['B365']=row['B365L']
            
    #     #     winner_data['EX']=row['EXW']
    #     #     loser_data['EX']=row['EXL']
            
    #     #     winner_data['LB']=row['LBW']
    #     #     loser_data['LB']=row['LBL']
            
    #     #     winner_data['PS']=row['PSW']
    #     #     loser_data['PS']=row['PSL']
                
     
    #     #     if np.all(loser_data['turned_pro']>0):
    #     #         loser_data['experience']=row['Date'].year-loser_data['turned_pro']
    #     #     else:
    #     #         loser_data['experience']=0
            
    #         #drop uneeded variables, and the pandas index.
    #         #winner_data=winner_data.drop(['last name','Unnamed: 0','year','name'],axis=1)
    #         #loser_data=loser_data.drop(['last name','Unnamed: 0','year','name'],axis=1)  
      
          
    #         #except:
    #         #    pass

missing=set(missing)        
data=pd.concat(alldata,axis=0)
data.to_csv('data.csv',index=False)
        
            
    
    
        
    
    
