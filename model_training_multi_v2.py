from typing import List
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

FEATURE_COLUMNS: List[str] = [
    'PLAYER_ENCODED', 'PTS_ROLLING_5', 'PTS_ROLLING_10', 'PTS_TREND',
    'MIN_ROLLING_5', 'FGA_ROLLING_5', 'DAYS_REST', 'IS_HOME',
    'PLAYER_LINE', 'LINE_MINUS_FORM',
]

TARGET = 'TARGET_OVER'


def build_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=0.2,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
    )


def train(data_path: Path) -> xgb.XGBClassifier:
    print('\n' + '=' * 60)
    print('XGBoost TRAINING  (chronological holdout)')
    print('=' * 60)

    df = pd.read_csv(data_path)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    # Drop "pushes" (points exactly equal to the line) - sportsbooks refund these
    before = len(df)
    df = df[df['PTS'] != df['PLAYER_LINE']].reset_index(drop=True)
    print(f'\nDropped {before - len(df)} push games (points == line)')

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f'Missing columns: {missing}. Re-run feature_engineering_multi_v2.py')

    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    seasons = sorted(df['SEASON'].unique().tolist())
    print(f'Dataset  : {len(df)} games | {df["PLAYER_NAME"].nunique()} players')
    print(f'Seasons  : {seasons}')
    print(f'Features : {len(FEATURE_COLUMNS)}')
    print(f'Target   : {y.mean()*100:.1f}% OVER / {(1-y.mean())*100:.1f}% UNDER')

    # ------------------------------------------------------------------ #
    # Chronological split: train on earlier season(s), test on the latest
    # ------------------------------------------------------------------ #
    if len(seasons) < 2:
        raise SystemExit('Need at least 2 seasons for a chronological split.')

    test_season = seasons[-1]
    train_idx = df[df['SEASON'] != test_season].index
    test_idx = df[df['SEASON'] == test_season].index

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    print(f'\nTrain: {seasons[:-1]} -> {len(X_train)} games')
    print(f'Test : {test_season} -> {len(X_test)} games')

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    model = build_model()
    print('\nTraining ...')
    model.fit(X_train, y_train)
    print('Done')

    # ------------------------------------------------------------------ #
    # Evaluate on the holdout season
    # ------------------------------------------------------------------ #
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Baselines to compare against
    base_form = (df.loc[test_idx, 'PTS_ROLLING_5'] > df.loc[test_idx, 'PLAYER_LINE']).astype(int)
    acc_form = accuracy_score(y_test, base_form)
    acc_major = max(y_test.mean(), 1 - y_test.mean())

    print('\n' + '=' * 60)
    print(f'RESULTS ON HOLDOUT SEASON ({test_season})')
    print('=' * 60)
    print(f'\nModel accuracy                           : {acc*100:.2f}%')
    print(f'Baseline: 5-game avg above line -> OVER  : {acc_form*100:.2f}%')
    print(f'Baseline: always predict majority class  : {acc_major*100:.2f}%')
    print(f'ROC-AUC                                  : {roc_auc:.3f}')

    n = len(y_test)
    margin = 1.96 * (acc * (1 - acc) / n) ** 0.5 * 100
    print(f'\n95% margin of error on accuracy (n={n}): +/- {margin:.1f} points')
    best_base = max(acc_form, acc_major)
    margin_frac = margin / 100
    if acc - best_base > margin_frac:
        print('-> The model beats the best baseline by more than the margin of error.')
    elif acc > best_base:
        print('-> The model is above the best baseline, but within the margin of error: '
              'cannot be told apart from it.')
    else:
        print('-> The model does NOT beat the best baseline.')

    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=['UNDER', 'OVER']))

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print('               Pred UNDER   Pred OVER')
    print(f'  Actual UNDER    {cm[0][0]:4d}        {cm[0][1]:4d}')
    print(f'  Actual OVER     {cm[1][0]:4d}        {cm[1][1]:4d}')

    # ------------------------------------------------------------------ #
    # Feature importance
    # ------------------------------------------------------------------ #
    fi = (pd.DataFrame({'feature': FEATURE_COLUMNS,
                        'importance': model.feature_importances_})
            .sort_values('importance', ascending=False))

    print('\n' + '=' * 60)
    print('FEATURE IMPORTANCE (treat as rough - small dataset)')
    print('=' * 60)
    for _, row in fi.iterrows():
        bar = '#' * int(row['importance'] * 60)
        print(f'  {row["feature"]:22s} {row["importance"]:.3f}  {bar}')

    # ------------------------------------------------------------------ #
    # Per-player accuracy on holdout
    # ------------------------------------------------------------------ #
    print('\n' + '=' * 60)
    print('PER-PLAYER ACCURACY (holdout)')
    print('=' * 60)
    results = pd.DataFrame({
        'PLAYER_NAME': df.loc[test_idx, 'PLAYER_NAME'].values,
        'actual': y_test.values,
        'predicted': y_pred,
    })
    for name, grp in results.groupby('PLAYER_NAME'):
        pa = accuracy_score(grp['actual'], grp['predicted'])
        print(f'  {name:20s}: {pa*100:.1f}%  ({len(grp)} games)')

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    model_path = Path('data/models/multi_player_model.pkl')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f'\nModel saved -> {model_path}')

    feat_path = Path('data/models/feature_columns.txt')
    feat_path.write_text('\n'.join(FEATURE_COLUMNS))
    print(f'Feature list saved -> {feat_path}\n')

    return model


if __name__ == '__main__':
    data_path = Path('data/processed/multi_player_features.csv')

    if not data_path.exists():
        print(f'{data_path} not found.\n'
              'Run in order:\n'
              '  1. python fetch_multi_player_data.py\n'
              '  2. python feature_engineering_multi_v2.py\n'
              '  3. python model_training_multi_v2.py')
        raise SystemExit(1)

    train(data_path)