# =============================================================================
# Pronostiqueur NBA – Value Bets (version améliorée – Niveau 1 : Advanced Stats)
# Ajout NET_RATING, OFF_RATING, DEF_RATING, PACE via LeagueDashTeamStats
# =============================================================================

import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
import time
import random
from datetime import date, timedelta
from typing import Dict, Optional

from nba_api.stats.endpoints import (
    scoreboardv2,
    leaguestandingsv2,
    leaguedashteamstats
)
from nba_api.stats.static import teams
from lightgbm import LGBMClassifier

# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

MODEL_PATH = "nba_model_advanced.pkl"
CACHE_TTL_STATS = 3600       # 1 heure
CACHE_TTL_ADVANCED = 3600
CACHE_TTL_SCOREBOARD = 900   # 15 min

TEAM_DATA = teams.get_teams()
TEAM_DICT = {t["teamId"]: t["abbreviation"] for t in TEAM_DATA}
TEAM_FULLNAME = {t["teamId"]: t["fullName"] for t in TEAM_DATA}

CURRENT_SEASON = "2025-26"   # Mets à jour si besoin pour 2026-27 etc.

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def parse_record(record: str) -> float:
    if not record or record in ("—", "-", "None", ""):
        return 0.50
    try:
        w, l = map(int, record.split("-"))
        return w / (w + l) if (w + l) > 0 else 0.50
    except:
        return 0.50


@st.cache_data(ttl=CACHE_TTL_STATS, show_spinner=False)
def load_basic_standings() -> Dict[int, dict]:
    try:
        standings = leaguestandingsv2.LeagueStandingsV2().get_data_frames()[0]
        stats = {}
        for _, row in standings.iterrows():
            tid = int(row["TEAM_ID"])
            stats[tid] = {
                "abbreviation": TEAM_DICT.get(tid, "???"),
                "full_name": TEAM_FULLNAME.get(tid, "Unknown"),
                "win_pct": float(row["W_PCT"]),
                "last10_pct": parse_record(row["L10"]),
                "home_pct": parse_record(row["HOME_RECORD"]),
                "road_pct": parse_record(row["ROAD_RECORD"]),
            }
        return stats
    except Exception as e:
        st.error(f"Erreur standings basiques : {e}")
        return {}


@st.cache_data(ttl=CACHE_TTL_ADVANCED, show_spinner=False)
def load_advanced_stats(season: str = CURRENT_SEASON) -> Dict[int, dict]:
    """Récupère NET_RATING, OFF_RATING, DEF_RATING, PACE – un seul appel"""
    try:
        adv = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season"
        )
        df = adv.get_data_frames()[0]

        advanced = {}
        for _, row in df.iterrows():
            tid = int(row["TEAM_ID"])
            advanced[tid] = {
                "net_rating": float(row.get("NET_RATING", 0.0)),
                "off_rating": float(row.get("OFF_RATING", 0.0)),
                "def_rating": float(row.get("DEF_RATING", 0.0)),
                "pace": float(row.get("PACE", 0.0)),
            }
        st.info(f"Advanced stats chargées pour {len(advanced)} équipes ({season})")
        return advanced
    except Exception as e:
        st.warning(f"Erreur advanced stats ({season}) : {e} → fallback à 0")
        return {}


def get_games_for_date(target_date: date, basic_stats: dict, adv_stats: dict) -> pd.DataFrame:
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=target_date.strftime("%Y-%m-%d"))
        games = sb.game_header.get_data_frame()

        if games.empty:
            return pd.DataFrame()

        rows = []
        for _, g in games.iterrows():
            home_id = int(g["HOME_TEAM_ID"])
            away_id = int(g["VISITOR_TEAM_ID"])

            h_basic = basic_stats.get(home_id, {"win_pct":0.5, "last10_pct":0.5, "home_pct":0.5, "road_pct":0.5})
            a_basic = basic_stats.get(away_id, {"win_pct":0.5, "last10_pct":0.5, "home_pct":0.5, "road_pct":0.5})

            h_adv = adv_stats.get(home_id, {"net_rating":0.0, "off_rating":0.0, "def_rating":0.0, "pace":100.0})
            a_adv = adv_stats.get(away_id, {"net_rating":0.0, "off_rating":0.0, "def_rating":0.0, "pace":100.0})

            rows.append({
                "game_id": g["GAME_ID"],
                "date": target_date.strftime("%Y-%m-%d"),
                "home_id": home_id,
                "away_id": away_id,
                "home": h_basic["abbreviation"],
                "away": a_basic["abbreviation"],
                "home_full": h_basic["full_name"],
                "away_full": a_basic["full_name"],
                "home_score": int(g.get("PTS_HOME", 0)),
                "away_score": int(g.get("PTS_AWAY", 0)),
                "status_text": g["GAME_STATUS_TEXT"].strip(),
                "status_id": int(g["GAME_STATUS_ID"]),
                # Basic
                "home_win_pct": h_basic["win_pct"],
                "away_win_pct": a_basic["win_pct"],
                "home_form": h_basic["last10_pct"],
                "away_form": a_basic["last10_pct"],
                "home_home_pct": h_basic["home_pct"],
                "away_road_pct": a_basic["road_pct"],
                # Advanced
                "home_net": h_adv["net_rating"],
                "away_net": a_adv["net_rating"],
                "home_off": h_adv["off_rating"],
                "away_off": a_adv["off_rating"],
                "home_def": h_adv["def_rating"],
                "away_def": a_adv["def_rating"],
                "home_pace": h_adv["pace"],
                "away_pace": a_adv["pace"],
            })

        time.sleep(0.7)
        return pd.DataFrame(rows)

    except Exception as e:
        st.warning(f"Erreur matchs {target_date:%Y-%m-%d} : {e}")
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL_SCOREBOARD)
def load_recent_games(days: int, basic: dict, advanced: dict) -> pd.DataFrame:
    games_list = []
    progress = st.progress(0.0)
    today = date.today()

    for i in range(days):
        d = today - timedelta(days=i)
        progress.text(f"Chargement {d:%d/%m/%Y} …")
        day_df = get_games_for_date(d, basic, advanced)
        if not day_df.empty:
            games_list.append(day_df)
        progress.progress((i + 1) / days)
        time.sleep(0.25)

    progress.empty()

    if not games_list:
        return pd.DataFrame()
    return pd.concat(games_list, ignore_index=True)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Anciennes features
    df["win_pct_diff"]   = df["home_win_pct"]   - df["away_win_pct"]
    df["form_diff"]      = df["home_form"]       - df["away_form"]
    df["venue_diff"]     = df["home_home_pct"]   - df["away_road_pct"]
    df["home_advantage"] = 0.065

    # Nouvelles – Advanced
    df["net_diff"]   = df["home_net"] - df["away_net"]
    df["off_diff"]   = df["home_off"] - df["away_off"]
    df["def_diff"]   = df["home_def"] - df["away_def"]   # note : plus élevé = meilleure défense
    df["pace_diff"]  = df["home_pace"] - df["away_pace"]

    # Target
    df["home_win"] = np.nan
    finished = (df["status_id"] == 3)
    df.loc[finished, "home_win"] = (df.loc[finished, "home_score"] > df.loc[finished, "away_score"]).astype(int)

    df["match_label"] = df["home_full"] + " vs " + df["away_full"]

    return df


# =============================================================================
# MODELE – plus de features
# =============================================================================

FEATURES = [
    "win_pct_diff", "form_diff", "venue_diff", "home_advantage",
    "net_diff", "off_diff", "def_diff", "pace_diff"
]

def get_or_train_model(df: pd.DataFrame):
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            st.sidebar.success("Modèle avancé chargé")
            return model
        except:
            pass

    train = df[df["home_win"].notna()]
    if len(train) < 50:
        st.sidebar.warning("Peu de matchs → modèle placeholder")
        model = LGBMClassifier(n_estimators=120, max_depth=5, random_state=42, verbosity=-1)
        X_fake = pd.DataFrame(np.random.normal(0, 5, (500, len(FEATURES))), columns=FEATURES)
        y_fake = (X_fake["net_diff"] + X_fake["def_diff"] > np.random.normal(0, 8, 500)).astype(int)
        model.fit(X_fake, y_fake)
    else:
        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=7,
            colsample_bytree=0.80,
            random_state=42,
            verbosity=-1
        )
        model.fit(train[FEATURES], train["home_win"])
        acc = model.score(train[FEATURES], train["home_win"])
        st.sidebar.success(f"Modèle avancé entraîné – {len(train)} matchs – acc train : {acc:.1%}")

    try:
        joblib.dump(model, MODEL_PATH)
    except:
        pass

    return model


def predict_all(df: pd.DataFrame, model) -> pd.DataFrame:
    df = df.copy()
    probas = model.predict_proba(df[FEATURES])[:, 1]

    df["proba_home"] = probas
    df["proba_away"] = 1 - probas

    df["cote_home"] = np.round(1 / np.maximum(probas, 0.04) * np.random.uniform(0.91, 0.99, len(df)), 2)
    df["cote_away"] = np.round(1 / np.maximum(1-probas, 0.04) * np.random.uniform(0.91, 0.99, len(df)), 2)

    df["ev_home"]  = df["proba_home"] * df["cote_home"] - 1
    df["ev_away"]  = df["proba_away"] * df["cote_away"] - 1
    df["best_ev"]  = df[["ev_home", "ev_away"]].max(axis=1)
    df["best_side"] = np.where(df["ev_home"] > df["ev_away"], "Domicile", "Extérieur")

    return df

# =============================================================================
# INTERFACE
# =============================================================================

def main():
    st.set_page_config(page_title="NBA Value Bets • Advanced", layout="wide", page_icon="🏀")

    st.title("🏀 NBA Pronostics – Advanced Stats")
    st.caption("Win% + forme + NET/OFF/DEF RATING + PACE – saison " + CURRENT_SEASON)

    with st.sidebar:
        st.header("Options")
        days = st.slider("Jours d'historique", 2, 14, 7)
        ev_threshold = st.slider("Seuil EV min (%)", 3, 12, 6) / 100

    with st.spinner("Chargement standings & advanced stats…"):
        basic_stats = load_basic_standings()
        adv_stats   = load_advanced_stats()
        if not basic_stats:
            st.error("Impossible de charger les données de base.")
            st.stop()

    with st.spinner(f"Récupération matchs ({days} jours)…"):
        df_raw = load_recent_games(days, basic_stats, adv_stats)

    if df_raw.empty:
        st.warning("Aucun match dans la fenêtre.")
        st.stop()

    df = prepare_features(df_raw)
    model = get_or_train_model(df)
    df_pred = predict_all(df, model)

    upcoming = df_pred[df_pred["status_id"].isin([1, 2])].copy()
    finished = df_pred[df_pred["status_id"] == 3].copy()

    tab1, tab2 = st.tabs(["À venir / Live", "Terminés"])

    with tab1:
        if upcoming.empty:
            st.info("Pas de match prévu/en cours.")
        else:
            st.subheader("Meilleurs value bets")
            show_upcoming_games(upcoming, ev_threshold)

    with tab2:
        st.subheader("Historique")
        show_finished_games(finished)


def show_upcoming_games(df: pd.DataFrame, seuil: float):
    df_show = df.sort_values("best_ev", ascending=False).copy()

    df_show["proba_home"] = df_show["proba_home"].map("{:.0%}".format)
    df_show["best_ev"]    = (df_show["best_ev"] * 100).map("{:+.1f}%".format)

    cols = ["date", "match_label", "status_text", "proba_home", "cote_home", "cote_away", "best_side", "best_ev"]

    st.dataframe(
        df_show[cols].rename(columns={
            "match_label": "Match",
            "status_text": "Statut",
            "proba_home": "Proba dom.",
            "best_side": "Meilleur pari",
            "best_ev": "EV"
        }),
        hide_index=True,
        use_container_width=True
    )

    strong = df_show[df_show["best_ev"] > seuil]
    if not strong.empty:
        top = strong.iloc[0]
        st.success(f"**Top pari** : {top['match_label']} – {top['best_side']} @ {top['cote_home' if top['best_side']=='Domicile' else 'cote_away']} (EV {top['best_ev']})")


def show_finished_games(df: pd.DataFrame):
    if df.empty:
        st.info("Aucun match terminé.")
        return

    df = df.sort_values("date", ascending=False).head(25).copy()
    df["correct"] = np.where(
        (df["proba_home"] > 0.5) == (df["home_win"] == 1),
        "✔", "✘"
    )

    st.dataframe(
        df[["date", "match_label", "home_score", "away_score", "proba_home", "correct"]].rename(columns={
            "match_label": "Match",
            "proba_home": "Proba dom.",
            "correct": "Prédiction"
        }),
        hide_index=True,
        use_container_width=True
    )


if __name__ == "__main__":
    main()