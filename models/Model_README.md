# 1. Logistic Regression Model (Generalized Linear Models)

## 1.1 Introduction

The goal of this project is to build a machine learning model to predict the probability of an NBA team winning a head-to-head game. We base our predictions on historical game results and team performance statistics. Because the target variable has only two outcomes (win or loss), logistic regression is a great baseline model for this classification task.

Unlike normal linear regression, logistic regression calculates the probability that an event will happen. Let $p_i$ be the probability that the home team wins game $i$. We define the logistic regression model as:

$$p_i = P(Y_i=1 \mid X_i)$$

where

$$p_i=\frac{\exp(\beta_0+\beta_1X_{i1}+\cdots+\beta_kX_{ik})}{1+\exp(\beta_0+\beta_1X_{i1}+\cdots+\beta_kX_{ik})}$$

We can also write this model in log-odds form:

$$\log\left(\frac{p_i}{1-p_i}\right) =\beta_0+\sum_{j=1}^{k}\beta_jX_{ij}$$

The coefficients $\beta_j$ show how much a change in the predictor variables changes the log-odds of the home team winning.

---

## 1.2 Response Variable and Predictor Variables

### Response Variable ($Y$)

The dependent variable is defined as:

$$Y_i= \begin{cases} 1, & \text{if the home team wins the game}, \\ 0, & \text{otherwise}. \end{cases}$$

So, the model predicts $P(Y_i=1)$, which is the probability that the home team wins.

### Predictor Variables ($X$)

Our explanatory variables are the differences between Team A (home team) and Team B (away team) using rolling 10-game statistics. For any statistic $S$:

$$S_{\text{diff}}=S_{\text{Team A/Home}}-S_{\text{Team B/Away}}$$

If the value is positive, it means Team A performed better than Team B in that specific statistic over their last ten games.

The predictors include:
> [!NOTE]
> needs final confirmation; data for each team will be broken down by home and away games.


* `teamScore_roll10_diff`: Team A's average score in its last 10 home games minus Team B's average score in its last 10 home games.
* `opponentScore_roll10_diff`: Team A's average opponent score in its last 10 home games minus Team B's average opponent score in its last 10 home games.
* `fieldGoalsPercentage_roll10_diff`: The difference in field goal percentage over the last 10 games between home team A and away team B.
* `threePointersPercentage_roll10_diff`: The difference in three-point shooting percentage over the last 10 games between home team A and away team B.
* `freeThrowsPercentage_roll10_diff`: The difference in free throw percentage over the last 10 games between home team A and away team B.
* `reboundsOffensive_roll10_diff`: The difference in average offensive rebounds over the last 10 games between home team A and away team B.
* `reboundsDefensive_roll10_diff`: The difference in average defensive rebounds over the last 10 games between home team A and away team B.
* `assists_roll10_diff`: The difference in average assists over the last 10 games between home team A and away team B.
* `turnovers_roll10_diff`: The difference in average turnovers over the last 10 games between home team A and away team B.
* `steals_roll10_diff`: The difference in average steals over the last 10 games between home team A and away team B.
* `blocks_roll10_diff`: The difference in average blocks over the last 10 games between home team A and away team B.
* `plusMinusPoints_roll10_diff`: The difference in average plus-minus points over the last 10 games between home team A and away team B.
* `win_roll10_diff`: The difference in total wins over the last 10 games between home team A and away team B.

We **standardized** all predictor variables before fitting the model so we can easily compare the size of the coefficients.

---

## 1.3 Model 1: Full Logistic Regression Model

The first model includes all thirteen rolling-difference features:

$$\log\left(\frac{p_i}{1-p_i}\right) = \beta_0+ \sum_{j=1}^{13} \beta_j S_{\text{diff},j}$$

We split the dataset into training and testing sets using a time-based 80% split. This ensures we do not leak future information into the past training data. Specifically, we set the split date at the 80th percentile of all game dates. Games before this date were used for training, and games on or after were saved for out-of-sample testing.

The out-of-sample performance for this full model is:

* **Accuracy:** 0.6355
* **ROC-AUC:** 0.6875
* **Log Loss:** 0.6332

These numbers show that the model captures a decent predictive signal, but it is still far from a perfect predictor for NBA games.

---

## 1.4 Model 2: Logistic Regression with LASSO-Selected Features

To reduce overfitting and automatically select the best features, we also fitted an L1-regularized (LASSO) logistic regression. Variables with coefficients shrunk to zero were removed.

The most important features retained by LASSO were:

* `plusMinusPoints_roll10_diff`
* `win_roll10_diff`
* `fieldGoalsPercentage_roll10_diff`
* `threePointersPercentage_roll10_diff`
* `freeThrowsPercentage_roll10_diff`
* `reboundsOffensive_roll10_diff`
* `reboundsDefensive_roll10_diff`
* `assists_roll10_diff`
* `turnovers_roll10_diff`
* `blocks_roll10_diff`

The final model formula is:

$$\log\left(\frac{p_i}{1-p_i}\right)= \beta_0+\sum_{j=1}^{10}\beta_j S_{\text{diff},j}$$

The largest positive coefficient belonged to `plusMinusPoints_roll10_diff`, and the second largest belonged to `win_roll10_diff`.

We used the exact same time-based 80% split to test this model. Its out-of-sample performance was:

* **Accuracy:** 0.6357
* **ROC-AUC:** 0.6875
* **Log Loss:** 0.6332

Because the results are almost identical to the full model, it tells us that the removed variables did not really add much useful predictive information anyway.

---

## 1.5 Model 3: Preseason Forecasting Model

We created a third evaluation setup to mimic how a real preseason forecast works. For any season $t$, we train the model only on data from the previous season ($t-1$), and then try to predict the matches in season $t$.

Formally:

$$\text{Training Data} = \{ \text{Season } t-1 \}$$

$$\text{Test Data} = \{ \text{Season } t \}$$

This rolling backtesting setup keeps future information safe from leaking and shows how well our model handles completely new seasons.

Here are the forecasting results:

| Season | Accuracy | ROC-AUC | Log Loss |
| --- | --- | --- | --- |
| 2021 | 0.626 | 0.655 | 0.654 |
| 2022 | 0.586 | 0.599 | 0.672 |
| 2023 | 0.635 | 0.680 | 0.640 |
| 2024 | 0.636 | 0.689 | 0.632 |
| 2025 | 0.650 | 0.704 | 0.621 |

The results show that our model's performance is relatively stable across different seasons, with the ROC-AUC staying between 0.60 and 0.70. The model performed best in the 2025 season, which suggests that recent form and rolling performance statistics carry real, meaningful signals for predicting NBA winners.

---

# 2. Generalized Additive Models

*(Content to be added)*

---

# 3. ARMA (Autoregressive Moving Average) Model

*(Content to be added)*

# 4. Random Forest model

*(Content to be added)*
