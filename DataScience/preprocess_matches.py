import os
from xml.etree import ElementTree as ET

f = []
count = 0
for path, subdirs, files in os.walk("Matches"):
    for name in files:
        file_path = os.path.join(path, name)
        print(file_path)
        try:
            xml = ET.parse(file_path)    
            root_element = xml.getroot()
            #print(root_element.attrib)
            
        #for child in root_element:
            #print(child)
        except:
            count+=1
            continue
        
        country = root_element.find('country').text
        league = root_element.find('league').text
        season = root_element.find('season').text
        stage = root_element.find('stage').text
        match_id = root_element.find('matchId').text

        home_team_id = root_element.find('homeTeamId').find('value').text
        away_team_id = root_element.find('awayTeamId').find('value').text

        home_team_full_name = root_element.find('homeTeamFullName').find('value').text
        away_team_full_name = root_element.find('awayTeamFullName').find('value').text
        
        home_team_acronym = root_element.find('homeTeamAcronym').find('value').text
        away_team_acronym = root_element.find('awayTeamAcronym').find('value').text

        home_team_players = root_element.find('homePlayers').findall('value')
        away_team_players = root_element.find('awayPlayers').findall('value')
        
        for player in home_team_players:
            #print(player.text)
            pass

        for player in away_team_players:
            #print(player.text)
            pass

        home_team_players_ids = root_element.find('homePlayersId').findall('value')
        away_team_players_ids = root_element.find('awayPlayersId').findall('value')

        for player in home_team_players_ids:
            pass
            #print(player.text)       

        for player in away_team_players_ids:
            pass
            #print(player.text)

        
        goals = root_element.find('goal').findall('value')
        for goal in goals:
            #print(goal.find('elapsed').text)
            pass

        print(home_team_acronym + " " + away_team_acronym)

print(count)