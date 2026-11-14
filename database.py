"""
Database management module for Football Match Outcome Prediction System.
Uses SQLite for persistent storage of leagues, teams, historical matches, and predictions.
"""

import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
import pandas as pd

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "football_prediction.db")


def get_connection(db_path: str = DB_FILE) -> sqlite3.Connection:
    """Creates a connection to the SQLite database with row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_FILE) -> None:
    """Initializes tables for leagues, teams, matches, and prediction history."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Leagues table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leagues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                country TEXT NOT NULL
            )
        """)

        # Teams table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                league_code TEXT NOT NULL,
                UNIQUE(name, league_code),
                FOREIGN KEY (league_code) REFERENCES leagues(code)
            )
        """)

        # Matches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_date TEXT NOT NULL,
                season TEXT NOT NULL,
                league_code TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                fthg INTEGER NOT NULL,
                ftag INTEGER NOT NULL,
                ftr TEXT NOT NULL,
                hthg INTEGER,
                htag INTEGER,
                hs INTEGER,
                as_ INTEGER,
                hst INTEGER,
                ast INTEGER,
                hy INTEGER,
                ay INTEGER,
                hr INTEGER,
                ar INTEGER,
                b365h REAL,
                b365d REAL,
                b365a REAL,
                UNIQUE(match_date, home_team, away_team),
                FOREIGN KEY (league_code) REFERENCES leagues(code)
            )
        """)

        # Predictions history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                match_date TEXT,
                league_code TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                predicted_outcome TEXT NOT NULL, -- '1', 'X', or '2'
                prob_home REAL NOT NULL,
                prob_draw REAL NOT NULL,
                prob_away REAL NOT NULL,
                confidence REAL NOT NULL,
                model_name TEXT NOT NULL,
                factors_json TEXT,
                actual_outcome TEXT,            -- '1', 'X', or '2'
                is_correct INTEGER              -- 1 if correct, 0 if incorrect, NULL if unverified
            )
        """)

        # Indices for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league_code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at)")

        # Seed initial leagues
        leagues_data = [
            ("E0", "Premier League", "England"),
            ("SP1", "La Liga", "Spain"),
            ("D1", "Bundesliga", "Germany"),
            ("I1", "Serie A", "Italy"),
            ("F1", "Ligue 1", "France")
        ]
        cursor.executemany("""
            INSERT OR IGNORE INTO leagues (code, name, country)
            VALUES (?, ?, ?)
        """, leagues_data)

        conn.commit()


def save_matches_to_db(df: pd.DataFrame, db_path: str = DB_FILE) -> int:
    """Inserts a cleaned dataframe of matches into SQLite."""
    if df.empty:
        return 0

    init_db(db_path)
    inserted_count = 0

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Update teams
        teams_to_insert = set()
        for _, row in df.iterrows():
            teams_to_insert.add((row['HomeTeam'], row['Div']))
            teams_to_insert.add((row['AwayTeam'], row['Div']))

        cursor.executemany("""
            INSERT OR IGNORE INTO teams (name, league_code)
            VALUES (?, ?)
        """, list(teams_to_insert))

        # Insert matches
        for _, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO matches (
                        match_date, season, league_code, home_team, away_team,
                        fthg, ftag, ftr, hthg, htag, hs, as_, hst, ast, hy, ay, hr, ar,
                        b365h, b365d, b365a
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(row['Date']),
                    str(row.get('Season', '2023/2024')),
                    str(row['Div']),
                    str(row['HomeTeam']),
                    str(row['AwayTeam']),
                    int(row['FTHG']),
                    int(row['FTAG']),
                    str(row['FTR']),
                    int(row.get('HTHG', 0)) if pd.notnull(row.get('HTHG')) else None,
                    int(row.get('HTAG', 0)) if pd.notnull(row.get('HTAG')) else None,
                    int(row.get('HS', 0)) if pd.notnull(row.get('HS')) else None,
                    int(row.get('AS', 0)) if pd.notnull(row.get('AS')) else None,
                    int(row.get('HST', 0)) if pd.notnull(row.get('HST')) else None,
                    int(row.get('AST', 0)) if pd.notnull(row.get('AST')) else None,
                    int(row.get('HY', 0)) if pd.notnull(row.get('HY')) else None,
                    int(row.get('AY', 0)) if pd.notnull(row.get('AY')) else None,
                    int(row.get('HR', 0)) if pd.notnull(row.get('HR')) else None,
                    int(row.get('AR', 0)) if pd.notnull(row.get('AR')) else None,
                    float(row.get('B365H', 0.0)) if pd.notnull(row.get('B365H')) else None,
                    float(row.get('B365D', 0.0)) if pd.notnull(row.get('B365D')) else None,
                    float(row.get('B365A', 0.0)) if pd.notnull(row.get('B365A')) else None
                ))
                inserted_count += 1
            except Exception:
                continue

        conn.commit()

    return inserted_count


def get_all_matches(db_path: str = DB_FILE, league_code: Optional[str] = None) -> pd.DataFrame:
    """Fetches matches from the database ordered chronologically."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        query = "SELECT * FROM matches"
        params = []
        if league_code and league_code != "ALL":
            query += " WHERE league_code = ?"
            params.append(league_code)
        query += " ORDER BY match_date ASC, id ASC"

        df = pd.read_sql_query(query, conn, params=params)
    return df


def get_leagues_list(db_path: str = DB_FILE) -> List[Dict[str, str]]:
    """Returns all leagues."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT code, name, country FROM leagues ORDER BY name ASC")
        return [dict(row) for row in cursor.fetchall()]


def get_teams_by_league(league_code: str, db_path: str = DB_FILE) -> List[str]:
    """Returns list of team names for a specified league."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if league_code == "ALL":
            cursor.execute("SELECT DISTINCT name FROM teams ORDER BY name ASC")
        else:
            cursor.execute("SELECT DISTINCT name FROM teams WHERE league_code = ? ORDER BY name ASC", (league_code,))
        return [row[0] for row in cursor.fetchall()]


def insert_prediction(
    league_code: str,
    home_team: str,
    away_team: str,
    predicted_outcome: str,
    prob_home: float,
    prob_draw: float,
    prob_away: float,
    confidence: float,
    model_name: str,
    factors: Dict[str, Any],
    match_date: Optional[str] = None,
    db_path: str = DB_FILE
) -> int:
    """Inserts a new prediction record into SQLite."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                match_date, league_code, home_team, away_team,
                predicted_outcome, prob_home, prob_draw, prob_away,
                confidence, model_name, factors_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            match_date,
            league_code,
            home_team,
            away_team,
            predicted_outcome,
            round(prob_home, 4),
            round(prob_draw, 4),
            round(prob_away, 4),
            round(confidence, 4),
            model_name,
            json.dumps(factors)
        ))
        conn.commit()
        return cursor.lastrowid


def get_predictions_history(db_path: str = DB_FILE) -> pd.DataFrame:
    """Returns all prediction history records."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        df = pd.read_sql_query("""
            SELECT 
                p.id,
                p.created_at,
                p.match_date,
                p.league_code,
                l.name as league_name,
                p.home_team,
                p.away_team,
                p.predicted_outcome,
                p.prob_home,
                p.prob_draw,
                p.prob_away,
                p.confidence,
                p.model_name,
                p.actual_outcome,
                p.is_correct,
                p.factors_json
            FROM predictions p
            LEFT JOIN leagues l ON p.league_code = l.code
            ORDER BY p.id DESC
        """, conn)
    return df


def update_prediction_actual(pred_id: int, actual_outcome: str, db_path: str = DB_FILE) -> bool:
    """Updates actual outcome for a prediction and marks if prediction was correct."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT predicted_outcome FROM predictions WHERE id = ?", (pred_id,))
        row = cursor.fetchone()
        if not row:
            return False
        
        predicted = row[0]
        is_correct = 1 if predicted.strip().upper() == actual_outcome.strip().upper() else 0
        
        cursor.execute("""
            UPDATE predictions 
            SET actual_outcome = ?, is_correct = ?
            WHERE id = ?
        """, (actual_outcome.strip().upper(), is_correct, pred_id))
        conn.commit()
        return True


def get_dashboard_stats(db_path: str = DB_FILE) -> Dict[str, Any]:
    """Aggregates high level stats for the Streamlit dashboard."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM matches")
        total_matches = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT name) FROM teams")
        total_teams = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT code) FROM leagues")
        total_leagues = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]

        cursor.execute("""
            SELECT 
                COUNT(*) as verified_count,
                SUM(is_correct) as correct_count
            FROM predictions
            WHERE is_correct IS NOT NULL
        """)
        v_row = cursor.fetchone()
        verified_count = v_row[0] or 0
        correct_count = v_row[1] or 0
        prediction_accuracy = (correct_count / verified_count * 100) if verified_count > 0 else None

        # Outcomes distribution in matches
        cursor.execute("""
            SELECT ftr, COUNT(*) as count 
            FROM matches 
            GROUP BY ftr
        """)
        match_outcomes = {row[0]: row[1] for row in cursor.fetchall()}

    return {
        "total_matches": total_matches,
        "total_teams": total_teams,
        "total_leagues": total_leagues,
        "total_predictions": total_predictions,
        "verified_predictions": verified_count,
        "correct_predictions": correct_count,
        "prediction_accuracy": prediction_accuracy,
        "match_outcomes": match_outcomes
    }
