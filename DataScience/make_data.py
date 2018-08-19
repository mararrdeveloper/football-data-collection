import pandas as pd
from random import random
import numpy as np

player=pd.read_csv('data/player_stats.csv')
player['birthday']=pd.to_datetime(player['birthday'])
files=[2014,2015,2016]
missing=[]    
alldata=[]

for f in files:
    games=pd.read_csv('data/' + str(f)+'.csv')
    #keep only completed games
    games=games[games['Comment']=='Completed']
    games['Date']=pd.to_datetime(games['Date'])
    #turn surface to lowrcase
    games['Surface']=games['Surface'].apply(str.lower)
    
    for index,row in games.iterrows():
        print(index)
        dummy={}

        dummy['month']=row['Date'].month
        
        #get the players' names
        winner_name=row['Winner'].split(' ')[0].lower()
        loser_name=row['Loser'].split(' ')[0].lower()
        
        winner_data=player[player['last name']==winner_name]
        loser_data=player[player['last name']==loser_name]
        
        #check if the player exists in the crawled data file
        if len(winner_data)==0:
            missing.append(winner_name)            
        
        if len(loser_data)==0:
            missing.append(loser_name)
            
        
        if len(winner_data)>0 and len(loser_data)>0:
            
            #get the previous years' data. If we would be getting this year's data we would be overfitting
            winner_data=winner_data[winner_data['year']==(row['Date'].year-1)]
            winner_data=winner_data[winner_data['surface']==row['Surface']]        
            
            loser_data=loser_data[loser_data['year']==(row['Date'].year-1)]
            loser_data=loser_data[loser_data['surface']==row['Surface']]        
            
            winner_data['points']=row['WPts']
            loser_data['points']=row['LPts']
            winner_data['rank']=row['WRank']
            loser_data['rank']=row['LRank']
            
            winner_data['B365']=row['B365W']
            loser_data['B365']=row['B365L']
            
            winner_data['EX']=row['EXW']
            loser_data['EX']=row['EXL']
            
            winner_data['LB']=row['LBW']
            loser_data['LB']=row['LBL']
            
            winner_data['PS']=row['PSW']
            loser_data['PS']=row['PSL']
                
            
            #number of years since turned pro
            if np.all(winner_data['turned_pro']>0):
                winner_data['experience']=row['Date'].year-winner_data['turned_pro']
            else:
                winner_data['experience']=0
            
            if np.all(loser_data['turned_pro']>0):
                loser_data['experience']=row['Date'].year-loser_data['turned_pro']
            else:
                loser_data['experience']=0
            
            #drop uneeded variables, and the pandas index.
            winner_data=winner_data.drop(['last name','Unnamed: 0','year','name'],axis=1)
            loser_data=loser_data.drop(['last name','Unnamed: 0','year','name'],axis=1)  
            
            try:               
                winner_data['birthday']=row['Date']-winner_data['birthday'] 
                winner_data['birthday']=winner_data['birthday'].iloc[0].days/365.0
                
                loser_data['birthday']=row['Date']-loser_data['birthday']  
                loser_data['birthday']=loser_data['birthday'].iloc[0].days/365.0
                
                #assign the winner randomly to the left or the right hand side.
                if random()>0.5:
                    dummy['winner']=0
                    dummy=pd.Series(dummy)
                    data=[winner_data.values[0],loser_data.values[0],dummy.values]
                    data=np.hstack(data)
                    df=pd.DataFrame(data)
                    df=df.transpose()
                    df.columns=np.hstack(['player0_'+winner_data.columns,'player1_'+loser_data.columns,dummy.index])
                else:
                    dummy['winner']=1
                    dummy=pd.Series(dummy)
                    data=[winner_data.values[0],loser_data.values[0],dummy.values]
                    data=np.hstack(data)
                    df=pd.DataFrame(data)
                    df=df.transpose()
                    df.columns=np.hstack(['player0_'+loser_data.columns,'player1_'+winner_data.columns,dummy.index])
                
                alldata.append(df)
            except:
                pass

missing=set(missing)        
data=pd.concat(alldata,axis=0)
data.to_csv('data/football_data.csv',index=False)
        
            
    
    
        
    
    
