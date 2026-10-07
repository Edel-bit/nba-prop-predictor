"""
Fetches two regular seasons of game logs (2023-24, 2024-25) for 5 NBA players
straight from stats.nba.com's JSON endpoint.

Only the player's own game logs are used. Opponent-defense ratings were removed
from the project because full-season team ratings include the game being
predicted (and later games), so they cannot be used honestly as a feature.
"""
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

PLAYERS = {
    'LeBron James':    '2544',
    'Anthony Edwards': '1630162',
    'Stephen Curry':   '201939',
    'Kevin Durant':    '201142',
    'Luka Doncic':     '1629029',
}

SEASONS = ['2023-24', '2024-25']

NBA_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'Accept': 'application/json',
}


def fetch_player_game_log(player_id: str, player_name: str, season: str) -> Optional[pd.DataFrame]:
    url = 'https://stats.nba.com/stats/playergamelog'
    params = {'PlayerID': player_id, 'Season': season, 'SeasonType': 'Regular Season'}

    print(f'  Fetching {player_name} | {season} ...')
    try:
        r = requests.get(url, headers=NBA_HEADERS, params=params, timeout=30)
        r.raise_for_status()
        rs = r.json()['resultSets'][0]
        df = pd.DataFrame(rs['rowSet'], columns=rs['headers'])
        df['PLAYER_NAME'] = player_name
        df['PLAYER_ID'] = player_id
        df['SEASON'] = season
        print(f'    OK: {len(df)} games')
        return df
    except Exception as e:
        print(f'    FAILED: {e}')
        return None


if __name__ == '__main__':
    print('=' * 60)
    print('FETCHING 2-SEASON MULTI-PLAYER NBA DATA')
    print(f'Players : {list(PLAYERS.keys())}')
    print(f'Seasons : {SEASONS}')
    print('=' * 60)

    all_frames = []

    for season in SEASONS:
        print(f'\n--- Season: {season} ---')
        for player_name, player_id in PLAYERS.items():
            df = fetch_player_game_log(player_id, player_name, season)
            if df is None or df.empty:
                continue
            all_frames.append(df)
            time.sleep(0.8)

    if not all_frames:
        print('\nNo data fetched. Exiting.')
        raise SystemExit(1)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.drop_duplicates(subset=['PLAYER_ID', 'Game_ID'], inplace=True)

    save_path = Path('data/raw/multi_player_stats.csv')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(save_path, index=False)

    print('\n' + '=' * 60)
    print('SUCCESS')
    print('=' * 60)
    print(f'Total games : {len(combined)}')
    print(f'Seasons     : {combined["SEASON"].unique().tolist()}')
    print('\nBreakdown by player and season:')
    for (name, season), grp in combined.groupby(['PLAYER_NAME', 'SEASON']):
        print(f'  {name:18s} {season}: {len(grp):3d} games  avg {grp["PTS"].mean():.1f} pts')
    print(f'\nSaved -> {save_path}')
    print('\nNext: python feature_engineering_multi_v2.py')