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

---

# 0. Model Selection

The goal of this project is to build a machine learning system to predict the probability of an NBA team winning a head-to-head matchup. Predictions are driven by historical game results, team-level win/loss records, and rolling individual/team performance metrics.

To establish a robust modeling framework, we evaluate several baseline and advanced models to understand their out-of-sample prediction performance:

* **Logistic Regression (Generalized Linear Models):** Our baseline statistical model, ideal for understanding feature importance and calculating direct, well-calibrated probabilities.
* **Generalized Additive Models (GAMs):** Implemented to capture non-linear relationships between team statistics without introducing excessive model complexity.
* **Tree-Based Ensembles (Random Forest, Extra Trees, XGBoost):** Implemented to capture complex feature interactions, structural thresholds, and non-linear predictive signals.

By comparing these methodologies, we aim to identify the optimal approach for reliable, real-world NBA game forecasting.

---

# 1. Logistic Regression Model

## 1.1 Introduction
Because predicting a win or loss is a binary classification problem, logistic regression serves as an excellent baseline. Unlike standard linear regression, logistic regression directly models the probability that a specific event occurs. 

Let $p_i$ be the probability that the home team wins game $i$. The logistic regression model is defined as:

$$p_i = P(Y_i=1 \mid X_i)$$

Represented mathematically via the logistic function:

$$p_i=\frac{\exp(\beta_0+\beta_1X_{i1}+\cdots+\beta_kX_{ik})}{1+\exp(\beta_0+\beta_1X_{i1}+\cdots+\beta_kX_{ik})}$$

Alternatively, this can be expressed in log-odds (logit) form:

$$\log\left(\frac{p_i}{1-p_i}\right) =\beta_0+\sum_{j=1}^{k}\beta_jX_{ij}$$

The coefficients $\beta_j$ quantify how a one-unit change in a standardized predictor variable affects the log-odds of the home team winning.

---

## 1.2 Response Variable and Predictor Variables

### Response Variable ($Y$)
The dependent variable is a binary indicator of the game's outcome:

$$Y_i= \begin{cases} 1, & \text{if the home team wins}, \\ 0, & \text{if the away team wins}. \end{cases}$$

### Predictor Variables ($X$)
To capture comparative, recent team form, our explanatory variables are calculated as the **difference between Team A (Home) and Team B (Away)** using rolling 10-game statistics. For any statistic $S$:

$$S_{\text{diff}} = S_{\text{Home}} - S_{\text{Away}}$$

**Core Predictors Used:**
* `teamScore_roll10_diff`: Average points scored difference.
* `fieldGoalsPercentage_roll10_diff`: Field goal percentage accuracy difference.
* `threePointersPercentage_roll10_diff`: 3pt shooting percentage difference.
* `freeThrowsPercentage_roll10_diff`: Free throw accuracy difference.
* `reboundsOffensive_roll10_diff / reboundsDefensive_roll10_diff`: Rebounding dominance differences.
* `assists_roll10_diff`: Ball movement/playmaking difference.
* `turnovers_roll10_diff`: Ball security difference.
* `steals_roll10_diff / blocks_roll10_diff`: Defensive disruption metrics.
* `plusMinusPoints_roll10_diff`: Overall net efficiency difference.
* `win_roll10_diff`: Recent win-rate velocity difference.

---

## 1.3 Chronological 80/20 Time Split Evaluation

To establish our static baseline performance, we implement a strict chronological 80/20 data split (training on the first 80% of games, testing on the remaining 20% to prevent temporal leakages). The model uses an $L_1$-regularized (LASSO) penalty framework ($C=0.1$, solver=`liblinear`).

### 1.3.1 Regular Season Models
Evaluating performance exclusively within regular season environments (`matchup_reg`):

#### Regular Season Baseline (`win_roll10_diff` only)
* **Accuracy:** 0.6302
* **ROC-AUC:** 0.6801
* **Log Loss:** 0.6383

#### Regular Season Full (All Features)
* **Accuracy:** 0.6495
* **ROC-AUC:** 0.7056
* **Log Loss:** 0.6226

| Top Regular Season Features | Coef |
| :--- | :---: |
| `plusMinusPoints_roll10_diff` | 0.4998 |
| `win_roll10_diff` | 0.1472 |
| `blocks_roll10_diff` | 0.0324 |
| `turnovers_roll10_diff` | -0.0254 |
| `threePointersPercentage_roll10_diff` | -0.0291 |
| `reboundsOffensive_roll10_diff` | -0.0482 |
| `b2b_diff` | -0.1173 |

---

### 1.3.2 Playoff Models
Evaluating model behavior strictly inside postseason environments (`matchup_po`):

#### Playoff Baseline (`win_roll10_diff` only)
* **Accuracy:** 0.5782
* **ROC-AUC:** 0.6179
* **Log Loss:** 0.6699

#### Playoff Full (All Features)
* **Accuracy:** 0.6400
* **ROC-AUC:** 0.6477
* **Log Loss:** 0.6507

| Top Playoff Features | Coef |
| :--- | :---: |
| `plusMinusPoints_roll10_diff` | 0.2923 |
| `fieldGoalsPercentage_roll10_diff` | 0.1329 |
| `rest_days_diff` | 0.0405 |
| `threePointersPercentage_roll10_diff` | -0.0791 |

*Insight:* In the playoffs, LASSO completely zeroes out several metrics including `win_roll10_diff` and `b2b_diff` (since back-to-backs do not occur in series play), highlighting the structural shift in playoff basketball settings.
---

### 1.3.3 Combined Models
Models trained and evaluated across the entire blended dataset (`matchup_clean`):

#### Combined Baseline
* **Accuracy:** 0.6297
* **ROC-AUC:** 0.6751
* **Log Loss:** 0.6411

#### Combined Full Model
* **Accuracy:** 0.6477
* **ROC-AUC:** 0.6991
* **Log Loss:** 0.6266

| Top Combined Features | Coef |
| :--- | :---: |
| `plusMinusPoints_roll10_diff` | 0.5024 |
| `win_roll10_diff` | 0.1291 |
| `blocks_roll10_diff` | 0.0367 |
| `fieldGoalsPercentage_roll10_diff` | 0.0193 |
| `turnovers_roll10_diff` | -0.0216 |
| `threePointersPercentage_roll10_diff` | -0.0370 |
| `b2b_diff` | -0.1107 |
---

 

## 1.4 Season-by-Season Backtesting Results

To simulate realistic production use cases, we execute a sequential backtest where the model trains on Season $t-1$ data to predict the entirety of Season $t$ regular season games.

| Season | Baseline Acc | Baseline AUC | Baseline LL | Full Acc | Full AUC | Full LL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2011** | 0.638 | 0.644 | 0.644 | 0.642 | 0.657 | 0.639 |
| **2012** | 0.643 | 0.669 | 0.630 | 0.647 | 0.676 | 0.625 |
| **2013** | 0.631 | 0.675 | 0.635 | 0.631 | 0.674 | 0.635 |
| **2014** | 0.657 | 0.685 | 0.627 | 0.666 | 0.696 | 0.623 |
| **2015** | 0.646 | 0.685 | 0.626 | 0.648 | 0.702 | 0.614 |
| **2016** | 0.619 | 0.640 | 0.652 | 0.626 | 0.654 | 0.646 |
| **2017** | 0.624 | 0.671 | 0.636 | 0.641 | 0.676 | 0.633 |
| **2018** | 0.630 | 0.649 | 0.643 | 0.651 | 0.672 | 0.632 |
| **2019** | 0.614 | 0.658 | 0.650 | 0.619 | 0.661 | 0.648 |
| **2020** | 0.596 | 0.635 | 0.661 | 0.607 | 0.650 | 0.654 |
| **2021** | 0.608 | 0.638 | 0.660 | 0.609 | 0.641 | 0.658 |
| **2022** | 0.568 | 0.595 | 0.674 | 0.600 | 0.618 | 0.665 |
| **2023** | 0.598 | 0.674 | 0.655 | 0.628 | 0.685 | 0.642 |
| **2024** | 0.633 | 0.676 | 0.638 | 0.655 | 0.710 | 0.620 |
| **2025** | 0.646 | 0.700 | 0.625 | 0.656 | 0.711 | 0.616 |

*Insight:* Performance scales positively across seasons, peaking in 2025 (AUC: 0.704), proving that rolling form metrics carry stable, predictive weight over time.

---

## 1.5 Postseason-Only Season-by-Season Backtesting

Running the same temporal backtest strictly on playoff data exposes high sample variance and predictive flatlines, caused by sparse game counts and uncalibrated probabilities across isolated short series.

| Season | Baseline Acc | Baseline AUC | Baseline LL | Full Acc | Full AUC | Full LL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2011** | 0.679 | 0.500 | 0.666 | 0.679 | 0.500 | 0.666 |
| **2012** | 0.635 | 0.500 | 0.668 | 0.635 | 0.595 | 0.662 |
| **2013** | 0.562 | 0.500 | 0.689 | 0.562 | 0.500 | 0.689 |
| **2014** | 0.407 | 0.500 | 0.693 | 0.407 | 0.500 | 0.693 |
| **2015** | 0.326 | 0.500 | 0.693 | 0.535 | 0.520 | 0.703 |
| **2016** | 0.570 | 0.500 | 0.684 | 0.570 | 0.718 | 0.669 |
| **2017** | 0.573 | 0.607 | 0.690 | 0.512 | 0.567 | 0.694 |
| **2018** | 0.561 | 0.500 | 0.687 | 0.561 | 0.500 | 0.687 |
| **2019** | 0.506 | 0.500 | 0.693 | 0.570 | 0.602 | 0.691 |
| **2020** | 0.418 | 0.500 | 0.693 | 0.527 | 0.536 | 0.692 |
| **2021** | 0.398 | 0.500 | 0.693 | 0.548 | 0.601 | 0.691 |
| **2022** | 0.411 | 0.500 | 0.693 | 0.411 | 0.500 | 0.693 |
| **2023** | 0.398 | 0.500 | 0.693 | 0.398 | 0.500 | 0.693 |
| **2024** | 0.433 | 0.500 | 0.693 | 0.400 | 0.454 | 0.696 |
| **2025** | 0.440 | 0.500 | 0.693 | 0.549 | 0.589 | 0.688 |

### Postseason Backtest Analysis Summary
* **Extreme Sample Variance:** Due to minor sample sizes (79 to 93 games per playoff year) and highly repetitive matchups within a 7-game series, single-season playoff accuracies fluctuate wildly—ranging from a high of **67.9%** (2011) down to a complete predictive failure of **32.6%** (2015).
* **The "Series Effect" Flattening:** Multiple playoff seasons show an ROC-AUC of exactly **0.5000** with a Log Loss of **0.6931**. This happens because LASSO regression completely zeroes out the coefficients for erratic rolling features when faced with sparse postseason data, forcing the model to output a uninformative, flat 50/50 probability map for every game.

---

## 1.6 Combined Dataset Season-by-Season Backtesting

Backtesting performances tracked over the fully integrated master dataset containing both Regular Season and Playoff games:

| Season | Baseline Acc | Baseline AUC | Baseline LL | Full Acc | Full AUC | Full LL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2011** | 0.647 | 0.640 | 0.642 | 0.651 | 0.653 | 0.638 |
| **2012** | 0.641 | 0.660 | 0.632 | 0.644 | 0.671 | 0.625 |
| **2013** | 0.622 | 0.659 | 0.642 | 0.626 | 0.668 | 0.638 |
| **2014** | 0.656 | 0.683 | 0.629 | 0.658 | 0.689 | 0.628 |
| **2015** | 0.647 | 0.679 | 0.627 | 0.652 | 0.697 | 0.616 |
| **2016** | 0.617 | 0.640 | 0.652 | 0.633 | 0.653 | 0.645 |
| **2017** | 0.626 | 0.668 | 0.635 | 0.649 | 0.671 | 0.633 |
| **2018** | 0.629 | 0.648 | 0.644 | 0.649 | 0.672 | 0.632 |
| **2019** | 0.608 | 0.652 | 0.654 | 0.616 | 0.660 | 0.650 |
| **2020** | 0.592 | 0.631 | 0.662 | 0.608 | 0.646 | 0.655 |
| **2021** | 0.609 | 0.633 | 0.661 | 0.610 | 0.641 | 0.658 |
| **2022** | 0.595 | 0.593 | 0.673 | 0.600 | 0.616 | 0.664 |
| **2023** | 0.597 | 0.671 | 0.655 | 0.633 | 0.682 | 0.642 |
| **2024** | 0.633 | 0.672 | 0.640 | 0.657 | 0.703 | 0.624 |
| **2025** | 0.641 | 0.692 | 0.628 | 0.656 | 0.708 | 0.617 |

### Combined Dataset Analysis Summary
* **Postseason Buffer Effect:** Blending the massive regular season data matrix back into the testing sets acts as a stabilizer. The presence of regular season games masks the hyper-volatility of pure playoff game evaluation, locking accuracy securely back into a predictable **60.0% to 65.8%** band across eras. 
* **Model Limits:** Despite leveraging cross-season performance variables, the combined evaluation maintains a persistent accuracy ceiling (~65.8% max in 2014). This underscores the irreducible noise embedded within multi-game series data, where strategic real-time player injuries and tactical adjustments deviate from regular season pacing rules.


## 1.7 Regular Season to Playoff Transfer Model

To evaluate the generalization limits of our baseline framework, we test how models trained on regular-season dynamics transfer to the high-intensity, strategic environment of the NBA Playoffs. The postseason presents unique modeling challenges: rotations shorten, game-planning becomes highly opponent-specific, and the raw "noise" of back-to-backs is eliminated.

We deployed an $L_1$-regularized (LASSO) Logistic Regression model ($C=0.1$, solver=`liblinear`) across two distinct validation structures.

### 1.7.1 Structure A: Aggregate Cross-Era Playoff Forecasting
First, we trained a single model on the historical regular season aggregate dataset and tested it blindly against all historical playoff games combined.

* **Accuracy:** 0.6120
* **ROC-AUC:** 0.6122
* **Log Loss:** 0.6570

*Insight:* The drop in ROC-AUC relative to regular-season splits shows that historical trends don't perfectly transfer to the postseason. Playoff matchups have structurally different defensive intensities and slower paces, which reduces the predictive power of regular-season rolling statistics.

---

### 1.7.2 Structure B: Same-Season Intratemporal Backtesting
To control for changing playing styles across different eras, we built a seasonal tracking loop. For each individual year from 2010 to 2025, the model trains exclusively on that year's regular-season games to predict its specific postseason outcomes.

| Season | Regular Games | Playoff Games | Accuracy | ROC-AUC | Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2010** | 1213 | 81 | 0.6667 | 0.6413 | 0.6265 |
| **2011** | 990 | 84 | 0.6667 | 0.6537 | 0.6179 |
| **2012** | 1229 | 85 | 0.6471 | 0.5651 | 0.6618 |
| **2013** | 1230 | 89 | 0.5281 | 0.4974 | 0.7110 |
| **2014** | 1230 | 81 | 0.5556 | 0.6105 | 0.6674 |
| **2015** | 1230 | 86 | 0.6163 | 0.6336 | 0.6362 |
| **2016** | 1230 | 79 | 0.5949 | 0.6327 | 0.6495 |
| **2017** | 1230 | 82 | 0.6585 | 0.5927 | 0.6499 |
| **2018** | 1230 | 82 | 0.6463 | 0.6630 | 0.6518 |
| **2019** | 1092 | 79 | 0.5823 | 0.5788 | 0.6960 |
| **2020** | 1080 | 91 | 0.5604 | 0.5929 | 0.6762 |
| **2021** | 1230 | 93 | 0.5484 | 0.5927 | 0.6718 |
| **2022** | 1230 | 90 | 0.6111 | 0.5701 | 0.6713 |
| **2023** | 1230 | 88 | 0.6136 | 0.6275 | 0.6668 |
| **2024** | 1225 | 90 | 0.6111 | 0.5892 | 0.6837 |
| **2025** | 1231 | 91 | 0.6044 | 0.6652 | 0.6409 |

*Key Takeaways:*
* **High Postseason Variance:** Predictive accuracy fluctuates wildly depending on the season, peaking at **66.67%** (2010, 2011) and dropping to a floor of **52.81%** in 2013, where regular-season data struggled to pick up on playoff adjustments.
* **Recent Stability:** The most recent **2025 Season** showed strong predictability (ROC-AUC of **0.6652**), indicating that our rolling performance metrics successfully captured the late-season form of playoff contenders.

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
* **Tensor Interaction ($te(0,1)$):** Evaluates joint structural effects between a team's field goal percentage differential and their net plus-minus differential.

---

## 2.3 Cross-Validation Framework and Performance
The model was validated using a randomized **5-Fold Cross-Validation** routine directly on the pipeline step execution to assess out-of-sample consistency. The cross-validation run demonstrated stable generalization properties across all folds:

* Acc Mean(STD) 0.637(0.009)
* ROC Mean(STD) 0.676(0.006)
* Prec Mean(STD) 0.656(0.015)

---

## 2.4 Full Model Test Performance
After checking validation stability, the pipeline was fitted on the full training set (`xg_tr`, `yg_tr`) and evaluated blindly against the out-of-sample test split (`xg_te`, `yg_te`).


* **Accuracy:** $0.629$
* **ROC-AUC:** $0.667$
* **Precision:** $0.647$
 
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
