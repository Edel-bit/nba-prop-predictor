# NBA Player Points Over/Under Predictor

A machine learning project that predicts whether an NBA player will go **over or under** a points line, with a Streamlit app for trying out game scenarios.

Built as a learning project to practice the full pipeline: data collection, feature engineering, time-aware model evaluation, and a simple UI. It is **not** a betting tool and makes no claim of beating sportsbooks.

## What it does

- Pulls game logs for 5 players (LeBron James, Anthony Edwards, Stephen Curry, Kevin Durant, Luka Doncic) for the 2023-24 and 2024-25 regular seasons from the NBA stats endpoint.
- Builds features from each player's *prior* games only (rolling scoring and minutes averages, rest days, home/away, and the player's own running-average line).
- Trains an XGBoost classifier to predict OVER/UNDER.
- Evaluates on a chronological holdout: train on 2023-24, test on 2024-25.
- Serves predictions through a Streamlit app.

## Project structure

```
fetch_multi_player_data.py        # downloads game logs -> data/raw/multi_player_stats.csv
feature_engineering_multi_v2.py   # builds features    -> data/processed/multi_player_features.csv
model_training_multi_v2.py        # trains + evaluates -> data/models/multi_player_model.pkl
app.py                            # Streamlit interface
requirements.txt
```

## How to run

```bash
git clone <this-repo-url>
cd <repo-folder>

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python fetch_multi_player_data.py        # 1. download the data
python feature_engineering_multi_v2.py   # 2. build features
python model_training_multi_v2.py        # 3. train + evaluate
streamlit run app.py                     # 4. launch the app
```

## Getting the data

The data and trained model are **not stored in this repo** (the `data/` folder is git-ignored). Running the three scripts above recreates everything:

- `fetch_multi_player_data.py` downloads each player's regular-season game logs for 2023-24 and 2024-25 from the public `stats.nba.com` JSON endpoint and saves them to `data/raw/multi_player_stats.csv`. It takes about a minute.
- `feature_engineering_multi_v2.py` turns that into `data/processed/multi_player_features.csv`.
- `model_training_multi_v2.py` trains the model, prints the holdout results, and saves `data/models/multi_player_model.pkl` for the app.

The app will show a "Model not found" message until you have run these steps in order. The NBA endpoint sometimes times out or returns errors. If the fetch fails, wait a few minutes and retry. Game-log data belongs to the NBA; it is downloaded for personal, educational use and is not redistributed here. Your exact accuracy may differ by a point or two from the numbers below, depending on your XGBoost version.

## Features used (10)

| Feature | Meaning |
|---|---|
| `PLAYER_ENCODED` | Which of the 5 players |
| `PTS_ROLLING_5`, `PTS_ROLLING_10` | Average points over the last 5 / 10 games (prior games only) |
| `PTS_TREND` | 5-game average minus 10-game average |
| `MIN_ROLLING_5`, `FGA_ROLLING_5` | Recent minutes and shot attempts |
| `DAYS_REST` | Days since the previous game (capped at 4) |
| `IS_HOME` | Home or away |
| `PLAYER_LINE` | The player's average points over their *prior* games, floored and shifted to an x.5 value like a sportsbook line |
| `LINE_MINUS_FORM` | `PLAYER_LINE` minus the 5-game average |

The target is `PTS > PLAYER_LINE`. Lines always end in .5, so a game can never land exactly on the line (no pushes).

## Results

Evaluated on a chronological holdout: trained only on 2023-24 (319 games), tested on 2024-25 (331 games).

| | Accuracy |
|---|---|
| XGBoost model | 56.2% |
| Baseline: always predict the majority class (UNDER) | 55.6% |
| Baseline: 5-game average above line -> OVER | 51.4% |

- ROC-AUC: 0.563 (0.5 is chance)
- 95% margin of error on accuracy: +/- 5.3 points (n = 331)

**Takeaway:** the model's accuracy is within the margin of error of the majority-class baseline, so it cannot be distinguished from a naive guess. With 5 players and about 330 test games, no real predictive edge can be claimed. The value of this project is the pipeline and the evaluation design (no leakage, chronological split, honest baselines), not a predictive edge.

Accuracy varies by roughly +/- 2 to 3 points across random seeds and XGBoost versions (in my tests the same code scored between about 54% and 58%), which is another reason not to read anything into small differences.

Why UNDER is the majority class: the line is a running average, and a player's average is pulled up by their big games, so more games land under it than over it (about 54% UNDER overall).

Per-player holdout accuracy ranged from 44% to 63% on only 50-79 games each. With samples that small this is noise, and I draw no conclusions from it. Feature importances are also nearly flat (all between 0.082 and 0.124), so I do not interpret them.

## Data leakage: what I found and fixed

An earlier version of this project overstated its accuracy. I found and fixed these problems:

1. **Leaky betting line.** Each player's line was their average over the entire dataset, which includes the games being predicted. The line is now the average of that player's *prior* games only.
2. **Shuffled cross-validation.** 5-fold CV with shuffling lets the model train on later games and test on earlier ones. I removed it and report only the chronological holdout.
3. **Line not seen by the model.** The line was only a label in the app. It is now a model input, so changing it changes the prediction.
4. **Redundant or dead features.** Several engineered features were rearrangements of other columns (or constants), and a game-spread feature was zero for every game because the spread data never loaded. I removed them.
5. **Opponent-defense feature.** It used full-season team ratings, which include the game being predicted and later games, and the live API call had silently fallen back to one hardcoded table for both seasons. Removing it changed accuracy by less than a point, so I dropped it rather than keep a feature I could not defend. Doing this properly would need date-aware ratings built only from prior games.

## Limitations

- **Small dataset.** About 650 games across 5 players, so accuracy estimates have wide error bars and feature-importance rankings are not reliable.
- **No opponent information.** The model does not know who the opponent is or how good their defense is.
- **Self-generated lines.** `PLAYER_LINE` is a running average, not a real sportsbook line. Real lines incorporate injuries, matchups and market information this model does not have. In the app, lines far from recent form are outside what the model trained on.
- **Uncalibrated probabilities.** The app can show high percentages (for example when the line is far below recent form), but the holdout results show the model has no reliable edge, so those numbers are not real confidence.
- **No injury or lineup data.** The model does not know who is out.
- **Points only, five players.** No other stat types, and adding a player requires retraining.
- **Not financial advice.** For education only.

## Possible next steps

- Add more players and seasons to shrink the margin of error (the main lever, more than model tuning).
- Add real, date-aware opponent ratings computed only from games before each matchup.
- Use rolling-origin (time-series) cross-validation for a more stable estimate.
- Compare against sportsbook lines if historical data becomes available.

## Tech stack

Python, pandas, XGBoost, scikit-learn, Streamlit, requests, joblib.