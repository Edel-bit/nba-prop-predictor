"""
Feature engineering for the NBA points over/under model.

Every feature for a game is computed ONLY from games played before it
(rolling windows use .shift(1)), and the line is an expanding mean of
previous games, so nothing from the future leaks into a row.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# Player ID -> numeric encoding (must stay stable for the app)
PLAYER_ENCODING = {
    '2544':    0,   # LeBron James
    '1630162': 1,   # Anthony Edwards
    '201939':  2,   # Stephen Curry
    '201142':  3,   # Kevin Durant
    '1629029': 4,   # Luka Doncic
}


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    all_processed = []

    for player_id in df['PLAYER_ID'].astype(str).unique():
        pdf = df[df['PLAYER_ID'].astype(str) == player_id].copy()

        pdf['GAME_DATE'] = pd.to_datetime(pdf['GAME_DATE'])
        pdf = pdf.sort_values('GAME_DATE').reset_index(drop=True)

        # Rolling form (previous games only)
        pdf['PTS_ROLLING_5'] = pdf['PTS'].rolling(5).mean().shift(1)
        pdf['PTS_ROLLING_10'] = pdf['PTS'].rolling(10).mean().shift(1)
        pdf['MIN_ROLLING_5'] = pdf['MIN'].rolling(5).mean().shift(1)
        pdf['FGA_ROLLING_5'] = pdf['FGA'].rolling(5).mean().shift(1)
        pdf['PTS_TREND'] = pdf['PTS_ROLLING_5'] - pdf['PTS_ROLLING_10']

        # Context
        pdf['DAYS_REST'] = pdf['GAME_DATE'].diff().dt.days.fillna(4).clip(upper=4)
        pdf['IS_HOME'] = pdf['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)

        pdf['PLAYER_ENCODED'] = PLAYER_ENCODING.get(player_id, -1)

        # Line = expanding mean of PREVIOUS games, snapped to x.5 (no pushes)
        pdf['PLAYER_LINE'] = np.floor(pdf['PTS'].expanding().mean().shift(1)) + 0.5
        pdf['TARGET_OVER'] = (pdf['PTS'] > pdf['PLAYER_LINE']).astype(int)
        pdf['LINE_MINUS_FORM'] = pdf['PLAYER_LINE'] - pdf['PTS_ROLLING_5']

        # Need 10 games of history before the first usable row
        pdf = pdf.dropna(subset=['PTS_ROLLING_10', 'PLAYER_LINE'])
        all_processed.append(pdf)

    combined = pd.concat(all_processed, ignore_index=True)

    feature_cols = [
        # Identifiers / meta
        'PLAYER_NAME', 'PLAYER_ID', 'GAME_DATE', 'MATCHUP', 'SEASON',
        # Target and reference
        'TARGET_OVER', 'PTS',
        # Model features
        'PLAYER_ENCODED',
        'PTS_ROLLING_5',
        'PTS_ROLLING_10',
        'PTS_TREND',
        'MIN_ROLLING_5',
        'FGA_ROLLING_5',
        'DAYS_REST',
        'IS_HOME',
        'PLAYER_LINE',
        'LINE_MINUS_FORM',
    ]
    return combined[feature_cols]


def print_feature_summary(df: pd.DataFrame) -> None:
    print('\n' + '=' * 60)
    print('FEATURE ENGINEERING SUMMARY')
    print('=' * 60)
    print(f'Total games : {len(df)}')
    print(f'Players     : {df["PLAYER_NAME"].nunique()}')
    print(f'Seasons     : {sorted(df["SEASON"].unique().tolist())}')

    print('\nGames per season:')
    for season, grp in df.groupby('SEASON'):
        print(f'  {season}: {len(grp)} games, {grp["TARGET_OVER"].mean()*100:.1f}% OVER')

    print('\nPer-player breakdown:')
    for player, grp in df.groupby('PLAYER_NAME'):
        print(f'  {player:20s}: {len(grp):3d} games  '
              f'avg {grp["PTS"].mean():.1f} pts  '
              f'avg line {grp["PLAYER_LINE"].mean():.1f}  '
              f'{grp["TARGET_OVER"].mean()*100:.0f}% OVER')

    print('\nFeature ranges:')
    for col in ('PTS_ROLLING_5', 'LINE_MINUS_FORM'):
        print(f'  {col:20s}: {df[col].min():.2f} to {df[col].max():.2f}')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    raw_path = Path('data/raw/multi_player_stats.csv')

    if not raw_path.exists():
        print(f'{raw_path} not found. Run fetch_multi_player_data.py first.')
        raise SystemExit(1)

    print('Loading raw data ...')
    raw = pd.read_csv(raw_path)

    print('Engineering features ...')
    processed = create_features(raw)
    print_feature_summary(processed)

    out_path = Path('data/processed/multi_player_features.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(out_path, index=False)
    print(f'Saved -> {out_path}')
    print('\nNext: python model_training_multi_v2.py')