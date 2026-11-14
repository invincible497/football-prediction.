"""
Feature Engineering Module for Football Match Outcome Prediction.
Calculates historical rolling form, team performance, home/away splits,
head-to-head records, rest days, and differentials strictly chronologically
with zero future data leakage.
"""

from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    'home_last5_pts',
    'away_last5_pts',
    'home_last5_gf',
    'away_last5_gf',
    'home_last5_ga',
    'away_last5_ga',
    'home_last5_gd',
    'away_last5_gd',
    'home_last5_win_rate',
    'away_last5_win_rate',
    'home_venue_win_pct',
    'away_venue_win_pct',
    'home_overall_win_pct',
    'away_overall_win_pct',
    'home_advantage',
    'h2h_matches_count',
    'h2h_home_win_pct',
    'h2h_away_win_pct',
    'h2h_draw_pct',
    'h2h_avg_home_goals',
    'h2h_avg_away_goals',
    'home_rest_days',
    'away_rest_days',
    'pts_diff_last5',
    'gf_diff_last5',
    'ga_diff_last5',
    'win_pct_diff'
]

TARGET_MAP = {'H': 0, 'D': 1, 'A': 2}
TARGET_INV_MAP = {0: '1', 1: 'X', 2: '2'}
TARGET_LABEL_MAP = {0: 'Home Win (1)', 1: 'Draw (X)', 2: 'Away Win (2)'}


def parse_date_obj(date_val: Any) -> datetime:
    """Helper to convert string or timestamp to datetime."""
    if isinstance(date_val, datetime):
        return date_val
    if isinstance(date_val, str):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d"):
            try:
                return datetime.strptime(date_val.strip(), fmt)
            except ValueError:
                pass
    return datetime(2023, 1, 1)


class TeamTracker:
    """Tracks chronological historical matches for a team with no future leakage."""

    def __init__(self, team_name: str):
        self.team_name = team_name
        self.matches: List[Dict[str, Any]] = []

    def add_match(self, date_dt: datetime, venue: str, goals_for: int, goals_against: int, result: str):
        """
        Record a completed match.
        venue: 'home' or 'away'
        result: 'W', 'D', 'L' from this team's perspective
        """
        pts = 3 if result == 'W' else (1 if result == 'D' else 0)
        self.matches.append({
            'date': date_dt,
            'venue': venue,
            'gf': goals_for,
            'ga': goals_against,
            'result': result,
            'pts': pts
        })

    def get_stats_before(self, cutoff_date: datetime) -> Dict[str, Any]:
        """Calculates rolling and cumulative stats strictly before cutoff_date."""
        past = [m for m in self.matches if m['date'] < cutoff_date]
        total_played = len(past)

        if total_played == 0:
            return {
                'last5_pts': 5.0,  # neutral priors
                'last5_gf': 1.2,
                'last5_ga': 1.2,
                'last5_gd': 0.0,
                'last5_win_rate': 0.33,
                'overall_win_pct': 0.35,
                'venue_win_pct': 0.35,
                'rest_days': 7.0,
                'recent_streak': []
            }

        # Last 5 matches
        last5 = past[-5:]
        n5 = len(last5)
        last5_pts = sum(m['pts'] for m in last5) * (5.0 / n5)  # normalized to 5 matches scale
        last5_gf = sum(m['gf'] for m in last5) / n5
        last5_ga = sum(m['ga'] for m in last5) / n5
        last5_gd = last5_gf - last5_ga
        last5_win_rate = sum(1 for m in last5 if m['result'] == 'W') / n5
        recent_streak = [m['result'] for m in last5]

        # Overall win percentage
        wins_total = sum(1 for m in past if m['result'] == 'W')
        overall_win_pct = wins_total / total_played

        # Rest days
        last_match_date = past[-1]['date']
        rest_days = max(1.0, min(30.0, float((cutoff_date - last_match_date).days)))

        return {
            'last5_pts': float(last5_pts),
            'last5_gf': float(last5_gf),
            'last5_ga': float(last5_ga),
            'last5_gd': float(last5_gd),
            'last5_win_rate': float(last5_win_rate),
            'overall_win_pct': float(overall_win_pct),
            'rest_days': float(rest_days),
            'recent_streak': recent_streak
        }

    def get_venue_win_pct_before(self, cutoff_date: datetime, venue: str) -> float:
        """Returns win rate specifically at home or away prior to cutoff date."""
        venue_matches = [m for m in self.matches if m['date'] < cutoff_date and m['venue'] == venue]
        if not venue_matches:
            return 0.40 if venue == 'home' else 0.28
        wins = sum(1 for m in venue_matches if m['result'] == 'W')
        return float(wins / len(venue_matches))


class H2HTracker:
    """Tracks Head-to-Head encounters between pairs of teams strictly chronologically."""

    def __init__(self):
        self.encounters: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    def _pair_key(self, t1: str, t2: str) -> Tuple[str, str]:
        return (t1, t2) if t1 < t2 else (t2, t1)

    def add_match(self, date_dt: datetime, home_team: str, away_team: str, hg: int, ag: int, ftr: str):
        key = self._pair_key(home_team, away_team)
        if key not in self.encounters:
            self.encounters[key] = []
        self.encounters[key].append({
            'date': date_dt,
            'home': home_team,
            'away': away_team,
            'hg': hg,
            'ag': ag,
            'ftr': ftr
        })

    def get_h2h_stats_before(self, home_team: str, away_team: str, cutoff_date: datetime) -> Dict[str, float]:
        key = self._pair_key(home_team, away_team)
        past = [m for m in self.encounters.get(key, []) if m['date'] < cutoff_date]
        total = len(past)

        if total == 0:
            return {
                'count': 0.0,
                'home_win_pct': 0.33,
                'away_win_pct': 0.33,
                'draw_pct': 0.34,
                'avg_home_goals': 1.3,
                'avg_away_goals': 1.1
            }

        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals = 0
        away_goals = 0

        for m in past:
            if m['home'] == home_team:
                # Home team was home
                home_goals += m['hg']
                away_goals += m['ag']
                if m['ftr'] == 'H':
                    home_wins += 1
                elif m['ftr'] == 'A':
                    away_wins += 1
                else:
                    draws += 1
            else:
                # Home team was away
                home_goals += m['ag']
                away_goals += m['hg']
                if m['ftr'] == 'A':
                    home_wins += 1
                elif m['ftr'] == 'H':
                    away_wins += 1
                else:
                    draws += 1

        return {
            'count': float(total),
            'home_win_pct': float(home_wins / total),
            'away_win_pct': float(away_wins / total),
            'draw_pct': float(draws / total),
            'avg_home_goals': float(home_goals / total),
            'avg_away_goals': float(away_goals / total)
        }


def build_match_features_dataset(df_matches: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Transforms historical matches into a fully leak-free feature dataframe.
    Every match's features are computed purely on past data.
    """
    if df_matches.empty:
        return pd.DataFrame(), pd.Series()

    # Ensure chronological order
    df = df_matches.copy()
    df['date_dt'] = df['match_date'].apply(parse_date_obj)
    df = df.sort_values(by=['date_dt', 'id']).reset_index(drop=True)

    team_trackers: Dict[str, TeamTracker] = {}
    h2h_tracker = H2HTracker()

    feature_rows = []
    targets = []

    # League home advantage prior
    total_home_wins = 0
    total_matches_so_far = 0

    for idx, row in df.iterrows():
        date_dt = row['date_dt']
        home_team = str(row['home_team'])
        away_team = str(row['away_team'])
        fthg = int(row['fthg'])
        ftag = int(row['ftag'])
        ftr = str(row['ftr']).upper()

        if home_team not in team_trackers:
            team_trackers[home_team] = TeamTracker(home_team)
        if away_team not in team_trackers:
            team_trackers[away_team] = TeamTracker(away_team)

        home_tracker = team_trackers[home_team]
        away_tracker = team_trackers[away_team]

        # Compute features using state BEFORE this match
        h_stats = home_tracker.get_stats_before(date_dt)
        a_stats = away_tracker.get_stats_before(date_dt)

        home_venue_win_pct = home_tracker.get_venue_win_pct_before(date_dt, 'home')
        away_venue_win_pct = away_tracker.get_venue_win_pct_before(date_dt, 'away')

        h2h_stats = h2h_tracker.get_h2h_stats_before(home_team, away_team, date_dt)

        home_adv = (total_home_wins / total_matches_so_far) if total_matches_so_far > 50 else 0.44

        feat = {
            'match_id': row.get('id', idx),
            'match_date': row['match_date'],
            'home_team': home_team,
            'away_team': away_team,
            'league_code': row['league_code'],
            'home_last5_pts': h_stats['last5_pts'],
            'away_last5_pts': a_stats['last5_pts'],
            'home_last5_gf': h_stats['last5_gf'],
            'away_last5_gf': a_stats['last5_gf'],
            'home_last5_ga': h_stats['last5_ga'],
            'away_last5_ga': a_stats['last5_ga'],
            'home_last5_gd': h_stats['last5_gd'],
            'away_last5_gd': a_stats['last5_gd'],
            'home_last5_win_rate': h_stats['last5_win_rate'],
            'away_last5_win_rate': a_stats['last5_win_rate'],
            'home_venue_win_pct': home_venue_win_pct,
            'away_venue_win_pct': away_venue_win_pct,
            'home_overall_win_pct': h_stats['overall_win_pct'],
            'away_overall_win_pct': a_stats['overall_win_pct'],
            'home_advantage': home_adv,
            'h2h_matches_count': h2h_stats['count'],
            'h2h_home_win_pct': h2h_stats['home_win_pct'],
            'h2h_away_win_pct': h2h_stats['away_win_pct'],
            'h2h_draw_pct': h2h_stats['draw_pct'],
            'h2h_avg_home_goals': h2h_stats['avg_home_goals'],
            'h2h_avg_away_goals': h2h_stats['avg_away_goals'],
            'home_rest_days': h_stats['rest_days'],
            'away_rest_days': a_stats['rest_days'],
            'pts_diff_last5': h_stats['last5_pts'] - a_stats['last5_pts'],
            'gf_diff_last5': h_stats['last5_gf'] - a_stats['last5_gf'],
            'ga_diff_last5': h_stats['last5_ga'] - a_stats['last5_ga'],
            'win_pct_diff': h_stats['overall_win_pct'] - a_stats['overall_win_pct']
        }

        # Target outcome
        if ftr in TARGET_MAP:
            feature_rows.append(feat)
            targets.append(TARGET_MAP[ftr])

        # UPDATE state AFTER recording features for this match
        h_res = 'W' if ftr == 'H' else ('D' if ftr == 'D' else 'L')
        a_res = 'W' if ftr == 'A' else ('D' if ftr == 'D' else 'L')
        home_tracker.add_match(date_dt, 'home', fthg, ftag, h_res)
        away_tracker.add_match(date_dt, 'away', ftag, fthg, a_res)
        h2h_tracker.add_match(date_dt, home_team, away_team, fthg, ftag, ftr)

        if ftr == 'H':
            total_home_wins += 1
        total_matches_so_far += 1

    X = pd.DataFrame(feature_rows)
    y = pd.Series(targets, name='target')
    return X, y


def extract_features_for_new_match(
    home_team: str,
    away_team: str,
    match_date_str: str,
    historical_matches_df: pd.DataFrame
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Extracts features for an upcoming match between two teams on a future or custom date,
    strictly using all historical matches prior to that date.
    Returns:
      (feature_dict, explanation_context)
    """
    date_dt = parse_date_obj(match_date_str)
    
    # Filter historical matches prior to this fixture date
    df = historical_matches_df.copy()
    df['date_dt'] = df['match_date'].apply(parse_date_obj)
    df = df[df['date_dt'] <= date_dt].sort_values(by=['date_dt', 'id']).reset_index(drop=True)

    team_trackers: Dict[str, TeamTracker] = {}
    h2h_tracker = H2HTracker()
    total_home_wins = 0
    total_matches = len(df)

    for _, row in df.iterrows():
        m_dt = row['date_dt']
        h = str(row['home_team'])
        a = str(row['away_team'])
        fthg = int(row['fthg'])
        ftag = int(row['ftag'])
        ftr = str(row['ftr']).upper()

        if h not in team_trackers:
            team_trackers[h] = TeamTracker(h)
        if a not in team_trackers:
            team_trackers[a] = TeamTracker(a)

        h_res = 'W' if ftr == 'H' else ('D' if ftr == 'D' else 'L')
        a_res = 'W' if ftr == 'A' else ('D' if ftr == 'D' else 'L')
        team_trackers[h].add_match(m_dt, 'home', fthg, ftag, h_res)
        team_trackers[a].add_match(m_dt, 'away', ftag, fthg, a_res)
        h2h_tracker.add_match(m_dt, h, a, fthg, ftag, ftr)

        if ftr == 'H':
            total_home_wins += 1

    home_tracker = team_trackers.get(home_team, TeamTracker(home_team))
    away_tracker = team_trackers.get(away_team, TeamTracker(away_team))

    h_stats = home_tracker.get_stats_before(date_dt)
    a_stats = away_tracker.get_stats_before(date_dt)
    home_venue_win_pct = home_tracker.get_venue_win_pct_before(date_dt, 'home')
    away_venue_win_pct = away_tracker.get_venue_win_pct_before(date_dt, 'away')
    h2h_stats = h2h_tracker.get_h2h_stats_before(home_team, away_team, date_dt)
    home_adv = (total_home_wins / total_matches) if total_matches > 50 else 0.44

    features = {
        'home_last5_pts': h_stats['last5_pts'],
        'away_last5_pts': a_stats['last5_pts'],
        'home_last5_gf': h_stats['last5_gf'],
        'away_last5_gf': a_stats['last5_gf'],
        'home_last5_ga': h_stats['last5_ga'],
        'away_last5_ga': a_stats['last5_ga'],
        'home_last5_gd': h_stats['last5_gd'],
        'away_last5_gd': a_stats['last5_gd'],
        'home_last5_win_rate': h_stats['last5_win_rate'],
        'away_last5_win_rate': a_stats['last5_win_rate'],
        'home_venue_win_pct': home_venue_win_pct,
        'away_venue_win_pct': away_venue_win_pct,
        'home_overall_win_pct': h_stats['overall_win_pct'],
        'away_overall_win_pct': a_stats['overall_win_pct'],
        'home_advantage': home_adv,
        'h2h_matches_count': h2h_stats['count'],
        'h2h_home_win_pct': h2h_stats['home_win_pct'],
        'h2h_away_win_pct': h2h_stats['away_win_pct'],
        'h2h_draw_pct': h2h_stats['draw_pct'],
        'h2h_avg_home_goals': h2h_stats['avg_home_goals'],
        'h2h_avg_away_goals': h2h_stats['avg_away_goals'],
        'home_rest_days': h_stats['rest_days'],
        'away_rest_days': a_stats['rest_days'],
        'pts_diff_last5': h_stats['last5_pts'] - a_stats['last5_pts'],
        'gf_diff_last5': h_stats['last5_gf'] - a_stats['last5_gf'],
        'ga_diff_last5': h_stats['last5_ga'] - a_stats['last5_ga'],
        'win_pct_diff': h_stats['overall_win_pct'] - a_stats['overall_win_pct']
    }

    context = {
        'home_team': home_team,
        'away_team': away_team,
        'match_date': match_date_str,
        'home_recent_streak': h_stats.get('recent_streak', []),
        'away_recent_streak': a_stats.get('recent_streak', []),
        'h_stats': h_stats,
        'a_stats': a_stats,
        'h2h_stats': h2h_stats,
        'home_venue_win_pct': home_venue_win_pct,
        'away_venue_win_pct': away_venue_win_pct
    }

    return features, context
