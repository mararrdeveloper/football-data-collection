/****** Script for SelectTopNRows command from SSMS  ******/
with goals as  
    (SELECT
        [MatchId]
        ,count(distinct ExternalId) as team_goals
        ,[TeamId]
    FROM [FootballData].[ENETSCORES].[Goals] g
    GROUP BY g.MatchId,g.TeamId),
        goals_before_80 as  
            (SELECT
                [MatchId]
                ,1 as team_goals_before_80
                ,[TeamId]
            FROM [FootballData].[ENETSCORES].[Goals] g
            WHERE minute < 80
            GROUP BY g.MatchId,g.TeamId),
            
        possession as (
            SELECT
                [MatchId]
                ,avg(CAST ( [HomePossession] AS int )) as HomeTeamPossession
                ,avg(CAST ( [AwayPossession] AS int )) as AwayTeamPossession
                ,[TeamId]
            FROM [FootballData].[ENETSCORES].[Possessions]
            GROUP BY MatchId, TeamId),

            /*
            red_cards as  (SELECT
                [MatchId]
                ,count(distinct ExternalId) as team_goals
                ,[TeamId]
            FROM [FootballData].[ENETSCORES].Cards g
            where card_type = 'r'
            group by g.MatchId,g.TeamId),
            yellow_cards as  (SELECT
                [MatchId]
                ,count(distinct ExternalId) as team_goals
                ,[TeamId]
            FROM [FootballData].[ENETSCORES].Cards g
            where card_type = 'y'
            group by g.MatchId,g.TeamId), */
            corners AS (
            SELECT MatchId, TeamId, count(distinct externalid) as team_corners
            FROM [FootballData].[ENETSCORES].[Corners]
            GROUP BY MatchId, TeamId
            ),
            shots_on as (
            select MatchId, TeamId, count(distinct externalid) as team_shots_on
                from [FootballData].[ENETSCORES].ShotOns
            group by MatchId, TeamId
            ),
            shots_off as (
            select MatchId, TeamId, count(distinct externalid) as team_shots_off
            from [FootballData].[ENETSCORES].ShotOffs
            group by MatchId, TeamId
            ),

            crosses as (
            select MatchId, TeamId, count(distinct externalid) as team_crosses
            from [FootballData].[ENETSCORES].Crosses
            group by MatchId, TeamId
            ),

            matches as (
            -- Top N matches 
            SELECT top 30	 
                [MatchId]
                ,[Date]
                ,[HomeTeamId]
                ,[AwayTeamId]
				,[HomeTeamFullName]
				,[AwayTeamFullName]
            FROM [FootballData].[ENETSCORES].[Matches] m
            -- Id parameter for home or away
            where HomeTeamId = 10252 
            order by date desc
            ),

            match_stats as (
            select 
            mm.MatchId,
			mm.HomeTeamFullName,
			mm.AwayTeamFullName,
            mm.HomeTeamId,
            mm.AwayTeamId,
            mm.date,
            coalesce(max(case when gg.TeamId = mm.HomeTeamId then team_goals end),0) as HomeTeamGoals, 
            coalesce(max(case when gg.TeamId = mm.AwayTeamId then team_goals end),0) as AwayTeamGoals, 

            coalesce(max(case when cc.TeamId = mm.AwayTeamId then team_corners end),0) as AwayTeamCorners, 
            coalesce(max(case when cc.TeamId = mm.HomeTeamId then team_corners end),0) as HomeTeamCorners, 

            coalesce(max(case when son.TeamId = mm.HomeTeamId then team_shots_on end),0) as HomeTeamShotsOn, 
            coalesce(max(case when son.TeamId = mm.AwayTeamId then team_shots_on end),0) as AwayTeamShotsOn, 

            coalesce(max(case when soff.TeamId = mm.HomeTeamId then team_shots_off end),0) as HomeTeamShotsOff, 
            coalesce(max(case when soff.TeamId = mm.AwayTeamId then team_shots_off end),0) as AwayTeamShotsOff, 

            coalesce(max(case when cr.TeamId = mm.HomeTeamId then team_crosses end),0) as HomeTeamCrosses, 
            coalesce(max(case when cr.TeamId = mm.AwayTeamId then team_crosses end),0) as AwayTeamCrosses, 

            coalesce(max(case when g80.TeamId = mm.HomeTeamId then team_goals_before_80 end),0) as HomeTeamGoalsBefore80min, 
            coalesce(max(case when g80.TeamId = mm.AwayTeamId then team_goals_before_80 end),0) as AwayTeamGoalsBefore80min,

            coalesce(max (team_goals_before_80 ),0) as MatchGoalsBefore80min,
            pp.HomeTeamPossession as home_team_possession_avg,
            pp.AwayTeamPossession as away_team_possession_avg
         
            from matches mm 
            left join goals gg on gg.MatchId = mm.MatchId
            left join corners cc on cc.MatchId = mm.MatchId
            left join shots_on son on son.MatchId = mm.MatchId
            left join shots_off soff  on soff.MatchId = mm.MatchId
            left join crosses cr  on cr.MatchId = mm.MatchId
            left join goals_before_80 g80 on g80.MatchId = mm.MatchId
            left join possession pp on pp.MatchId = mm.matchid
            --Before data
            where mm.Date <  '2016-04-30 10:00:00.000'
            group by mm.Date,
            mm.MatchId,
            mm.HomeTeamId,
            mm.AwayTeamId,
			mm.HomeTeamFullName,
			mm.AwayTeamFullName,
            pp.HomeTeamPossession,
            pp.AwayTeamPossession
            )
select t.*,
case 
	when HomeTeamGoals > AwayTeamGoals then 'H'
	when HomeTeamGoals = AwayTeamGoals then 'D'
	when HomeTeamGoals < AwayTeamGoals then 'A'
	end as match_result
from match_stats t

