with matches as (
SELECT
	  [MatchId]
      ,[League]
      ,[Country]
      ,[HomeTeamFullName]
      ,[AwayTeamFullname]
      ,[AwayTeamId]
      ,[HomeTeamId]
      ,[Date]
      ,[Season]
      ,[Stage]
FROM  [FootballData].[ENETSCORES].[Matches]
where Date >  '2017-01-05 15:00:00'
and League = 'Premier League'),
goals_before_80 as
    (SELECT
        [MatchId]
        ,1 as team_goals_before_80
        ,[TeamId]
    FROM [FootballData].[ENETSCORES].[Goals] g
    WHERE minute < 80
    GROUP BY g.MatchId,g.TeamId),
goals as
(SELECT
    [MatchId]
    ,count(distinct ExternalId) as team_goals
    ,[TeamId]
FROM [FootballData].[ENETSCORES].[Goals] g
GROUP BY g.MatchId,g.TeamId),
corners_before_30 AS (
                    SELECT MatchId, TeamId, 1 as team_corners_before_30_min
                    FROM [FootballData].[ENETSCORES].[Corners]
					where minute < 30
                    GROUP BY MatchId, TeamId),
match_stats as (
                        select distinct
						mm.MatchId,
						mm.League,
						mm.Country,
						mm.HomeTeamFullName,
						mm.AwayTeamFullname,
						mm.AwayTeamId,
						mm.HomeTeamId,
						mm.Date,
						mm.Season,
						mm.Stage,
                        coalesce(max(case when gg.TeamId = mm.HomeTeamId then team_goals end),0) as HomeTeamGoals,
                        coalesce(max(case when gg.TeamId = mm.AwayTeamId then team_goals end),0) as AwayTeamGoals,
                        coalesce(max(case when g80.TeamId = mm.HomeTeamId then team_goals_before_80 end),0) as HomeTeamGoalsBefore80min,
                        coalesce(max(case when g80.TeamId = mm.AwayTeamId then team_goals_before_80 end),0) as AwayTeamGoalsBefore80min,
                        coalesce(max (team_goals_before_80 ),0) as MatchGoalsBefore80min,
						coalesce(max (team_corners_before_30_min ),0) as MatchCornersBefore30min

                        from matches mm
                        left join goals gg on gg.MatchId = mm.MatchId
                        left join corners_before_30 cc on cc.MatchId = mm.MatchId
                        left join goals_before_80 g80 on g80.MatchId = mm.MatchId

                        --Before data
                        group by
								mm.MatchId,
								mm.League,
								mm.Country,
								mm.HomeTeamFullName,
								mm.AwayTeamFullname,
								mm.AwayTeamId,
								mm.HomeTeamId,
								mm.Date,
								mm.Season,
								mm.Stage
		)
		        select t.*,
        case
            when HomeTeamGoals > AwayTeamGoals then 'H'
            when HomeTeamGoals = AwayTeamGoals then 'D'
            when HomeTeamGoals < AwayTeamGoals then 'A'
            end as match_result
        from match_stats t


