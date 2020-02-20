/****** Script for SelectTopNRows command from SSMS  ******/
with goals as  (SELECT
      [MatchId]
      ,count(distinct ExternalId) as team_goals
      ,[TeamId]
  FROM [FootballData].[ENETSCORES].[Goals] g
  group by g.MatchId,g.TeamId),

  goals_before_80 as  (SELECT
      [MatchId]
      ,1 as team_goals_before_80
      ,[TeamId]
  FROM [FootballData].[ENETSCORES].[Goals] g
  where minute < 80
  group by g.MatchId,g.TeamId)
  ,
  corners as (
  select MatchId, TeamId, count(distinct externalid) as team_corners
  from [FootballData].[ENETSCORES].[Corners]
  group by MatchId, TeamId
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
SELECT top 10 
      [MatchId]
      ,[Date]
      ,[HomeTeamId]
      ,[AwayTeamId]
  FROM [FootballData].[ENETSCORES].[Matches] m
  -- Id parameter for home or away
  where HomeTeamId = 10252 
  order by date desc
),
match_stats as (
select 

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

mm.MatchId,
mm.HomeTeamId,
mm.AwayTeamId,
mm.date
from matches mm 
left join goals gg on gg.MatchId = mm.MatchId
left join corners cc on cc.MatchId = mm.MatchId
left join shots_on son on son.MatchId = mm.MatchId
left join shots_off soff  on soff.MatchId = mm.MatchId
left join crosses cr  on cr.MatchId = mm.MatchId
left join goals_before_80 g80 on g80.MatchId = mm.MatchId
--Before data
where mm.Date <  '2016-04-30 10:00:00.000'
group by 
mm.Date,
mm.MatchId,
mm.HomeTeamId,
mm.AwayTeamId
)

select  HomeTeamId, --AwayTeamId
sum( HomeTeamGoals) as home_goals_sum,
sum( AwayTeamGoals) as away_goals_sum,
sum( HomeTeamCorners) as home_corners_sum,
sum( AwayTeamCorners) as away_corners_sum,
sum( HomeTeamShotsOn) as home_shot_on_sum,
sum( AwayTeamShotsOn) as away_shot_on_sum,
sum( HomeTeamShotsOn) as home_shot_off_sum,
sum( AwayTeamShotsOn) as away_shot_off_sum,
sum( HomeTeamCrosses) as home_crosses_sum,
sum( AwayTeamCrosses) as away_crosses_sum,
sum(case when HomeTeamGoalsBefore80min  > 0 then 1 else null end) as home_games_with_goals_before_80,
sum(case when AwayTeamGoalsBefore80min  > 0 then 1 else null end ) as away_games_with_goals_before_80,
sum( case when MatchGoalsBefore80min  > 0 then 1 else null end ) as total_matches_with_goals_before_80
from match_stats 
group by HomeTeamId --AwayTeamId

