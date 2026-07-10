# NBA Head-to-Head Win Probability Prediction

This project aims to develop a machine learning system to predict the win probability of NBA teams in head-to-head matchups using historical data and combining team-level win/loss records with individual player performance metrics.

**Stakeholders**
* Sports betting platforms
* Prediction markets
* Gamblers
* General managers
* Coaches

**KPIs**
* Prediction accuracy
* ROC-AUC

**Data sets**
* Primary data set: [NBA Kaggle data set](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores)
* Supplementary data set: [nba_api python package](https://nba-apidocumentation.knowledgeowl.com/help)


---

# Model Selection

The goal of this project is to develop a machine learning system to predict the probability of an NBA team winning a head-to-head matchup. Predictions are based on historical game outcomes, team-level performance, and rolling statistical features that summarize recent form.

To evaluate predictive performance, we compare three increasingly sophisticated modeling approaches using the same dataset and evaluation framework.

- **Baseline Models:** Simple heuristic-based predictors that establish benchmark performance for all machine learning methods.
- **Logistic Regression (Generalized Linear Models):** An interpretable probabilistic classifier that models the relationship between recent team performance and game outcomes.
- **Generalized Additive Models (GAMs):** Extend logistic regression by allowing nonlinear relationships while maintaining interpretability.
- **Tree-Based Ensembles (Random Forest, Extra Trees, XGBoost):** Capture complex nonlinear interactions and higher-order feature relationships that linear models may miss.

By comparing these approaches against common baseline predictors, we assess the trade-off between model complexity, interpretability, and predictive performance.

---

# Data

## Dataset

All models are trained and evaluated using NBA games from **2010–2025**, including both **regular season** and **playoff** games. Data is collected through the `nba_api` Python package and combined into a single dataset for model development and evaluation.

---

## Response Variable

The response variable is a binary indicator of the game outcome:

$$
Y_i = 
\begin{cases} 
1, & \text{if the home team wins}, \\ 
0, & \text{if the away team wins}. 
\end{cases}
$$

---

## Predictor Variables

Each predictor represents the difference between the **Home** and **Away** teams using rolling statistics from the previous ten games.

For any statistic $S$,

$$
S_{\text{diff}} = S_{\text{Home}} - S_{\text{Away}}
$$

The full feature set includes:

- `teamScore_roll10_diff`: Difference in average points scored over the previous 10 games.
- `fieldGoalsPercentage_roll10_diff`: Difference in field goal shooting percentage over the previous 10 games.
- `threePointersPercentage_roll10_diff`: Difference in three-point shooting percentage over the previous 10 games.
- `freeThrowsPercentage_roll10_diff`: Difference in free throw shooting percentage over the previous 10 games.
- `reboundsOffensive_roll10_diff`: Difference in offensive rebounds over the previous 10 games.
- `reboundsDefensive_roll10_diff`: Difference in defensive rebounds over the previous 10 games.
- `assists_roll10_diff`: Difference in assists over the previous 10 games, reflecting ball movement and playmaking.
- `turnovers_roll10_diff`: Difference in turnovers over the previous 10 games, measuring ball security.
- `steals_roll10_diff`: Difference in steals over the previous 10 games, indicating defensive pressure.
- `blocks_roll10_diff`: Difference in blocks over the previous 10 games, measuring rim protection.
- `plusMinusPoints_roll10_diff`: Difference in average plus/minus over the previous 10 games, representing overall team efficiency.
- `win_roll10_diff`: Difference in wins over the previous 10 games, capturing recent team form.
- `rest_days_diff`: Difference in the number of rest days before the game.
- `b2b_diff`: Difference in back-to-back game status between the two teams.

---

# 0. Baseline Models

To provide meaningful benchmarks, two simple baseline models are evaluated alongside every machine learning model.

## Baseline 1: Home Team Wins

This baseline predicts that the home team wins every game.

$$
\hat{Y} = 1
$$

It represents the historical home-court advantage in the NBA and provides the minimum benchmark that any predictive model should exceed.

---

## Baseline 2: Last-10 Game Record

This baseline predicts the winner using only the difference in each team's record over their previous ten games (`win_roll10_diff`).

Prediction rule:

- If `win_roll10_diff > 0`, predict a home-team win.
- If `win_roll10_diff < 0`, predict an away-team win.
- If both teams have the same recent record, predict a home-team win.

This baseline captures recent team momentum while remaining highly interpretable.

---

# 1. Logistic Regression Model

## 1.1 Introduction

Predicting the outcome of an NBA game is a binary classification problem. Logistic regression models the probability that the home team wins while providing interpretable coefficients for each predictor.

Let $p_i$ denote the probability that the home team wins game $i$.

$$
p_i = P(Y_i = 1 \mid X_i)
$$

The logistic regression model is defined as:

$$
p_i = \frac{\exp(\beta_0 + \beta_1 X_{i1} + \dots + \beta_k X_{ik})}{1 + \exp(\beta_0 + \beta_1 X_{i1} + \dots + \beta_k X_{ik})}
$$

or equivalently,

$$
\log\left(\frac{p_i}{1 - p_i}\right) = \beta_0 + \sum_{j=1}^{k} \beta_j X_{ij}
$$

Model coefficients quantify how each standardized predictor changes the log-odds of the home team winning.

---

## 1.2 Final Model

The final logistic regression model is trained using the **combined regular season and playoff dataset (2010–2025)**.

### Model Configuration

- **Dataset:** Combined regular season and playoff games
- **Features:** Full feature set
- **Feature Scaling:** `StandardScaler`
- **Model:** Logistic Regression
- **Regularization:** L1 (LASSO)
- **Solver:** `liblinear`
- **Regularization parameter:** `C = 0.1`

### Evaluation

A rolling five-year backtesting framework is used to simulate real-world prediction.

For each evaluation season (2020–2025):
- Train on the previous five seasons.
- Predict every game in the following season.
- Compute Accuracy, ROC-AUC, and Precision.

---

## 1.3 Results

### Overall Performance

| Model | Accuracy | ROC-AUC | Precision |
| :--- | :---: | :---: | :---: |
| Baseline 1 (Home Team Wins) | 0.554 | 0.500 | 0.554 |
| Baseline 2 (Last-10 Record) | 0.611 | 0.621 | 0.646 |
| **Logistic Regression (Full Features)** | **0.629** | **0.669** | **0.640** |

The logistic regression model improves predictive performance over both baseline methods by incorporating rolling team statistics, scheduling variables, and recent performance indicators.

### Rolling Five-Year Backtesting

| Season | Accuracy | ROC-AUC | Precision |
| :---: | :---: | :---: | :---: |
| 2020 | 0.5995 | 0.6459 | 0.6062 |
| 2021 | 0.6221 | 0.6494 | 0.6251 |
| 2022 | 0.6015 | 0.6114 | 0.6371 |
| 2023 | 0.6373 | 0.6924 | 0.6424 |
| 2024 | 0.6563 | 0.7079 | 0.6622 |
| 2025 | 0.6558 | 0.7086 | 0.6643 |

### Summary Statistics

| Metric | Mean | Standard Deviation |
| :--- | :---: | :---: |
| Accuracy | 0.6288 | 0.0253 |
| ROC-AUC | 0.6693 | 0.0397 |
| Precision | 0.6395 | 0.0222 |

The rolling five-year evaluation demonstrates that the logistic regression model consistently outperforms both baseline methods while maintaining stable predictive performance across seasons. Achieving an average accuracy of **62.9%** and an average ROC-AUC of **0.669**, the model confirms that rolling team performance statistics provide meaningful predictive signals for NBA game outcomes.

---

# 2. Generalized Additive Models (GAMs)

## 2.1 Introduction
While linear models assume structural relationships are completely straight lines, basketball performance often exhibits non-linear trends (e.g., diminishing returns on high-volume shooting or extreme turnover thresholds). Generalized Additive Models (GAMs) blend the interpretability of linear models with the flexibility of non-linear functions.

In a binary response task like predicting an NBA game outcome ($Y_i \in \{0, 1\}$), a **Logistic GAM** maps the probability of a home team victory $p_i = P(Y_i=1)$ via smooth, localized functions applied to individual features:

$$\log\left(\frac{p_i}{1-p_i}\right) = \beta_0 + \sum_{j=1}^{m} f_j(X_{ij})$$

Where $f_j(X_{ij})$ represent smooth functions (such as splines) or parametric linear components, automatically adapting to structural shifts in efficiency metrics.

---

## 2.2 Feature Selection and Interaction Architecture
Instead of using all engineered differences blindly, feature selection was refined to highlight a specific mixture of team differentials, baseline team anchors, and multi-variable tensor interactions. 

The targeted feature list (`py_fea`) contains:
1. `fieldGoalsPercentage_roll10_diff` (Index 0): Home vs. away shooting efficiency differential.
2. `plusMinusPoints_roll10_diff` (Index 1): Overall net rolling efficiency differential.
3. `plusMinusPoints_roll10_away` (Index 2): Raw contextual strength baseline of the away team.
4. `win_roll10_diff` (Index 3): Rolling win percentage velocity differential.
5. `turnovers_roll10_home` (Index 4): Precision control marker tracking home team possession stability.

### The PyGAM Model Specification
The pipeline utilizes `StandardScaler` followed by a custom `LogisticGAM` formula mixing smooth splines (`s`), linear constraints (`l`), and a bivariate tensor product interaction term (`te`):

$$\text{Formula Structure} = s(0) + l(1) + l(2) + s(3) + l(4) + te(0, 1)$$

* **Smooth Terms ($s$):** Applied to capture potential non-linearities in field goal percentages (Index 0) and win differentials (Index 3).
* **Linear Terms ($l$):** Enforce strict linear relationships on broader efficiency variables like rolling plus-minus records and baseline turnover counts.
* **Tensor Interaction ($te$(0,1)):** Evaluates joint structural effects between a team's field goal percentage differential and their net plus-minus differential.

---

## 2.3 Cross-Validation Framework and Performance
The model was validated using a randomized **5-Fold Cross-Validation** routine directly on the pipeline step execution to assess out-of-sample consistency. The cross-validation run demonstrated stable generalization properties across all folds:

* Accuracy Mean (STD): 0.632 (0.008)
* ROC Mean (STD): 0.678 (0.009)
* Precision Mean (STD): 0.700 (0.017)

---

## 2.4 Full Model Test Performance
After checking validation stability, the pipeline was fitted on the full training set (`xg_tr`, `yg_tr`) and evaluated blindly against the out-of-sample test split (`xg_te`, `yg_te`).


* **Accuracy:** $0.624$
* **ROC-AUC:** $0.675$
* **Precision:** $0.687$
 
---



# 3. Random Forest and Extra Trees Models

To evaluate non-linear interactions and variance reduction, tree-based ensemble methods were trained using a rolling cross-validation strategy: **training on a rolling 5-year window** and validating on the subsequent season (from 2015 to 2024).

### 3.1 Random Forest Classifier Performance & Feature Importances
The `RandomForestClassifier` was tuned using structural limits (`n_estimators=500`, `max_depth=10`, `min_samples_leaf=5`, `max_features=3`, `max_samples=500`) to prevent over-fitting on noisy basketball records.

| Validation Season | Accuracy | plusMinusPoints_roll10_diff | fieldGoalsPercentage_roll10_diff | win_roll10_diff | rest_days_diff | b2b_diff |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2015** | 0.6431 | 0.1544 | 0.0930 | 0.0949 | 0.0176 | 0.0111 |
| **2016** | 0.6301 | 0.1509 | 0.0947 | 0.0928 | 0.0189 | 0.0115 |
| **2017** | 0.6211 | 0.1522 | 0.0964 | 0.0867 | 0.0189 | 0.0090 |
| **2018** | 0.6374 | 0.1532 | 0.0922 | 0.0947 | 0.0175 | 0.0105 |
| **2019** | 0.6126 | 0.1515 | 0.0872 | 0.0844 | 0.0177 | 0.0092 |
| **2020** | 0.6009 | 0.1438 | 0.0896 | 0.0837 | 0.0175 | 0.0096 |
| **2021** | 0.6187 | 0.1413 | 0.0918 | 0.0773 | 0.0188 | 0.0104 |
| **2022** | 0.5951 | 0.1385 | 0.0907 | 0.0799 | 0.0164 | 0.0086 |
| **2023** | 0.6390 | 0.1308 | 0.0887 | 0.0706 | 0.0180 | 0.0108 |
| **2024** | 0.6489 | 0.1331 | 0.0890 | 0.0735 | 0.0187 | 0.0113 |

*Insight:* `plusMinusPoints_roll10_diff` consistently stands out as the most dominant predictive split variable (~13-15%), followed tightly by raw field goal efficiency differentials and rolling win metrics. Schedule factors like rest days and back-to-backs hold very little relative predictive importance within the ensemble trees.

### 3.2 Extra Trees Classifier Alternative
The `ExtraTreesClassifier` randomizes split thresholds rather than searching for the most discriminative path. This variant altered how the model evaluated feature contributions:

* **Accuracy Range:** Reached a peak accuracy of **65.71%** in the 2024 validation block.
* **Feature Distribution Shift:** Unlike Random Forest, Extra Trees placed an overwhelming weight on **both** `plusMinusPoints_roll10_diff` (~24-28%) and `win_roll10_diff` (~21-26%), while severely flattening the importance of auxiliary box-score statistics (such as three-point and free-throw percentage differentials, which dropped to ~2-4%).

---
# 4. XGBoost Model (Extreme Gradient Boosting)

To complement the bagging approaches, we implemented an **XGBoost Classifier**, a gradient boosting method that optimizes predictions sequentially by minimizing a specific loss function via gradient descent.

### 4.1 Training Setup & Hyperparameters
Following the chronological constraint of our time series data, the dataset was split with a strict **80% training / 20% validation split** without any shuffling to prevent temporal information leakage. 

The model configuration was set as follows:
 
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
 

### 4.2 Out-of-Sample Performance Results
Evaluating the model on the remaining chronological 20% test partition yielded the following results:

* **Accuracy:** 0.6328
* **Log Loss:** 0.6448
* **ROC AUC:** 0.6744

### 4.3 Comparative Evaluation
While XGBoost handles non-linear feature interactions smoothly, it achieves competitive baseline performance comparable to our regularized logistic regression and random forest benchmarks. This predictive ceiling across architectures demonstrates the high degree of baseline noise and unpredictability ("any given night" factor) embedded in regular-season NBA matchup dynamics.
