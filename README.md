# summer26-nba-head-to-head-probability
Team project: summer26-nba-head-to-head-probability

**NBA Head-to-Head Win Probability Prediction**

This project aims to develop a machine learning system to predict the win probability of NBA teams in head-to-head matchups using historical data and combining team-level win/loss records with individual player performance metrics.

**Stakeholders**
* Coaches, owners, prediction markets, gamblers

**KPIs**

* Program that predicts the probability of success based upon past results
* Game by game: predicts the probability of success just on the next game
* Playoff series: given the success of the season before the playoffs, predicts the probability of each team winning the playoffs
* Championship success: this is essentially the same as game by game, but could predict who wins the series, i.e., four out of seven possible games
* Team construction: based upon the data, predict the characteristics of the team that makes it more likely to win

**Data sets**

* nba_api python package: interfaces with nba.com to extract data: [link](https://nba-apidocumentation.knowledgeowl.com/help)
* Kaggle sets
  * NBA dataset: Box scores and stats: [link](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores)
  * NBA database: [link](https://www.kaggle.com/datasets/wyattowalsh/basketball)

