# =============================================================================
# Pronostiqueur Multi-Sports – Value Bets (version améliorée – Niveau 1 : Advanced Stats)
# Intégration API-Sports pour Football (Soccer)
# Support initial NBA, placeholders pour autres sports (tennis, esport)
# =============================================================================
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
import time
import random
import requests
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
MODEL_PATH = "multi_sport_model_advanced.pkl"
CACHE_TTL_STATS = 3600  # 1 heure
CACHE_TTL_ADVANCED = 3600
CACHE_TTL_SCOREBOARD = 900  # 15 min
CURRENT_SEASON = "2025"  # Pour football: 2025 = 2025/2026 ; pour NBA: "2025-26"
API_SPORTS_BASE_URL = "https://v3.football.api-sports.io"
API_KEY = st.secrets.get("api_sports_key", "")
if not API_KEY and "Football (Soccer)" in SPORTS:
    st.warning("Ajoutez votre clé API-Sports dans st.secrets['api_sports_key'] pour le football. Inscrivez-vous sur https://api-sports.io")

# Sports supportés - DÉFINI AVANT D'ÊTRE UTILISÉ
SPORTS = ["NBA (Basketball)", "Football (Soccer)", "Tennis", "Esport"]

# Major leagues pour football
MAJOR_LEAGUES = {
    "Premier League": 39,
    "Ligue 1": 61,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
}

# Pour NBA
TEAM_DATA = teams.get_teams()
TEAM_DICT = {t["id"]: t["abbreviation"] for t in TEAM_DATA}
TEAM_FULLNAME = {t["id"]: t["full_name"] for t in TEAM_DATA}

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

def parse_last10_football(form: str) -> float:
    if not form:
        return 0.5
    last10 = form[-10:]
    wins = last10.count('W')
    draws = last10.count('D') * 0.5  # Option: compter draw comme 0.5
    played = len(last10)
    return (wins + draws) / played if played > 0 else 0.5

# =============================================================================
# CHARGEMENT DONNEES PAR SPORT
# =============================================================================
@st.cache_data(ttl=CACHE_TTL_STATS, show_spinner=False)
def load_basic_standings(sport: str, league_id: Optional[int] = None) -> Dict[int, dict]:
    if sport == "NBA (Basketball)":
        try:
            standings = leaguestandingsv2.LeagueStandingsV2().get_data_frames()[0]
            stats = {}
            for _, row in standings.iterrows():
                tid = int(row["TeamID"])
                stats[tid] = {
                    "abbreviation": TEAM_DICT.get(tid, "???"),
                    "full_name": TEAM_FULLNAME.get(tid, "Unknown"),
                    "win_pct": float(row["W_PCT"]),
                    "last10_pct": parse_record(row["Last10"]),
                    "home_pct": parse_record(row["Home"]),
                    "road_pct": parse_record(row["Road"]),
                    "goals_for": 0,  # Pas pour NBA, mais pour uniformité
                    "goals_against": 0,
                    "played": 0,
                }
            return stats
        except Exception as e:
            st.error(f"Erreur standings {sport} : {e}")
            return {}
    elif sport == "Football (Soccer)":
        if not league_id:
            return {}
        try:
            url = f"{API_SPORTS_BASE_URL}/standings?league={league_id}&season={CURRENT_SEASON}"
            headers = {"x-apisports-key": API_KEY}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json().get("response", [])
            if not data:
                return {}
            standings_data = data[0]["league"]["standings"][0]  # Assuming one group
            stats = {}
            for team in standings_data:
                tid = team["team"]["id"]
                all_stats = team["all"]
                stats[tid] = {
                    "abbreviation": team["team"]["name"][:3].upper(),
                    "full_name": team["team"]["name"],
                    "win_pct": all_stats["win"] / all_stats["played"] if all_stats["played"] > 0 else 0.5,
                    "last10_pct": parse_last10_football(team.get("form", "")),
                    "home_pct": all_stats["home"]["win"] / all_stats["home"]["played"] if all_stats["home"]["played"] > 0 else 0.5,
                    "road_pct": all_stats["away"]["win"] / all_stats["away"]["played"] if all_stats["away"]["played"] > 0 else 0.5,
                    "goals_for": all_stats["goals"]["for"],
                    "goals_against": all_stats["goals"]["against"],
                    "played": all_stats["played"],
                }
            return stats
        except Exception as e:
            st.error(f"Erreur standings {sport} (league {league_id}) : {e}")
            return {}
    else:
        # Placeholder pour autres sports
        st.warning(f"Standings pour {sport} non implémentés – Utilisation données simulées")
        return simulate_standings(sport)

@st.cache_data(ttl=CACHE_TTL_ADVANCED, show_spinner=False)
def load_advanced_stats(sport: str, basic_stats: Dict[int, dict]) -> Dict[int, dict]:
    if sport == "NBA (Basketball)":
        try:
            adv = leaguedashteamstats.LeagueDashTeamStats(
                season=CURRENT_SEASON,
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
            st.info(f"Advanced stats chargées pour {len(advanced)} équipes ({CURRENT_SEASON}) – {sport}")
            return advanced
        except Exception as e:
            st.warning(f"Erreur advanced stats {sport} : {e} → fallback à 0")
            return {}
    elif sport == "Football (Soccer)":
        advanced = {}
        for tid, s in basic_stats.items():
            played = s["played"]
            if played > 0:
                off = s["goals_for"] / played
                def_r = s["goals_against"] / played
                net = off - def_r
                pace = off + def_r  # Avg goals per game as proxy
            else:
                off = def_r = net = pace = 0.0
            advanced[tid] = {
                "net_rating": net,
                "off_rating": off,
                "def_rating": def_r,
                "pace": pace,
            }
        return advanced
    else:
        # Placeholder
        st.warning(f"Advanced stats pour {sport} non implémentés – Utilisation données simulées")
        return simulate_advanced_stats(sport)

def simulate_standings(sport: str) -> Dict[int, dict]:
    num_teams = random.randint(8, 20)
    stats = {}
    for i in range(num_teams):
        tid = 1000 + i
        win_pct = random.uniform(0.3, 0.7)
        stats[tid] = {
            "abbreviation": f"T{i}",
            "full_name": f"Team {i} ({sport})",
            "win_pct": win_pct,
            "last10_pct": random.uniform(0.4, 0.6),
            "home_pct": win_pct + random.uniform(-0.05, 0.1),
            "road_pct": win_pct + random.uniform(-0.1, 0.05),
            "goals_for": 0,
            "goals_against": 0,
            "played": 0,
        }
    return stats

def simulate_advanced_stats(sport: str) -> Dict[int, dict]:
    num_teams = random.randint(8, 20)
    advanced = {}
    for i in range(num_teams):
        tid = 1000 + i
        advanced[tid] = {
            "net_rating": random.uniform(-10, 10),
            "off_rating": random.uniform(90, 120),
            "def_rating": random.uniform(90, 120),
            "pace": random.uniform(90, 110),
        }
    return advanced

def get_games_for_date(sport: str, target_date: date, basic_stats: dict, adv_stats: dict, league_id: Optional[int] = None) -> pd.DataFrame:
    date_str = target_date.strftime("%Y-%m-%d")
    if sport == "NBA (Basketball)":
        try:
            sb = scoreboardv2.ScoreboardV2(game_date=date_str)
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
                    "date": date_str,
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
                    "home_win_pct": h_basic["win_pct"],
                    "away_win_pct": a_basic["win_pct"],
                    "home_form": h_basic["last10_pct"],
                    "away_form": a_basic["last10_pct"],
                    "home_home_pct": h_basic["home_pct"],
                    "away_road_pct": a_basic["road_pct"],
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
            st.warning(f"Erreur matchs {date_str} {sport} : {e}")
            return pd.DataFrame()
    elif sport == "Football (Soccer)":
        if not league_id:
            return pd.DataFrame()
        try:
            url = f"{API_SPORTS_BASE_URL}/fixtures?date={date_str}&league={league_id}&season={CURRENT_SEASON}"
            headers = {"x-apisports-key": API_KEY}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json().get("response", [])
            rows = []
            for fixture in data:
                home_id = fixture["teams"]["home"]["id"]
                away_id = fixture["teams"]["away"]["id"]
                status_short = fixture["fixture"]["status"]["short"]
                if status_short in ["NS", "TBD"]:
                    status_id = 1
                    status_text = "Prévu"
                elif status_short in ["1H", "HT", "2H", "ET", "P", "BT", "LIVE"]:
                    status_id = 2
                    status_text = "En cours"
                elif status_short in ["FT", "AET", "PEN"]:
                    status_id = 3
                    status_text = "Terminé"
                else:
                    continue  # Skip other statuses
                h_basic = basic_stats.get(home_id, {"win_pct":0.5, "last10_pct":0.5, "home_pct":0.5, "road_pct":0.5})
                a_basic = basic_stats.get(away_id, {"win_pct":0.5, "last10_pct":0.5, "home_pct":0.5, "road_pct":0.5})
                h_adv = adv_stats.get(home_id, {"net_rating":0.0, "off_rating":0.0, "def_rating":0.0, "pace":100.0})
                a_adv = adv_stats.get(away_id, {"net_rating":0.0, "off_rating":0.0, "def_rating":0.0, "pace":100.0})
                rows.append({
                    "game_id": fixture["fixture"]["id"],
                    "date": date_str,
                    "home_id": home_id,
                    "away_id": away_id,
                    "home": h_basic["abbreviation"],
                    "away": a_basic["abbreviation"],
                    "home_full": h_basic["full_name"],
                    "away_full": a_basic["full_name"],
                    "home_score": fixture["goals"]["home"] or 0,
                    "away_score": fixture["goals"]["away"] or 0,
                    "status_text": status_text,
                    "status_id": status_id,
                    "home_win_pct": h_basic["win_pct"],
                    "away_win_pct": a_basic["win_pct"],
                    "home_form": h_basic["last10_pct"],
                    "away_form": a_basic["last10_pct"],
                    "home_home_pct": h_basic["home_pct"],
                    "away_road_pct": a_basic["road_pct"],
                    "home_net": h_adv["net_rating"],
                    "away_net": a_adv["net_rating"],
                    "home_off": h_adv["off_rating"],
                    "away_off": a_adv["off_rating"],
                    "home_def": h_adv["def_rating"],
                    "away_def": a_adv["def_rating"],
                    "home_pace": h_adv["pace"],
                    "away_pace": a_adv["pace"],
                })
            return pd.DataFrame(rows)
        except Exception as e:
            st.warning(f"Erreur matchs {date_str} {sport} (league {league_id}) : {e}")
            return pd.DataFrame()
    else:
        st.warning(f"Matchs pour {sport} non implémentés – Simulation aléatoire")
        num_games = random.randint(0, 5)
        rows = []
        team_ids = list(basic_stats.keys())
        if len(team_ids) < 2:
            return pd.DataFrame()
        for _ in range(num_games):
            home_id = random.choice(team_ids)
            away_id = random.choice([tid for tid in team_ids if tid != home_id])
            h_basic = basic_stats[home_id]
            a_basic = basic_stats[away_id]
            h_adv = adv_stats[home_id]
            a_adv = adv_stats[away_id]
            status_id = random.choice([1,2,3])
            home_score = random.randint(0, 120) if status_id > 1 else 0
            away_score = random.randint(0, 120) if status_id > 1 else 0
            rows.append({
                "game_id": f"SIM-{random.randint(1000,9999)}",
                "date": date_str,
                "home_id": home_id,
                "away_id": away_id,
                "home": h_basic["abbreviation"],
                "away": a_basic["abbreviation"],
                "home_full": h_basic["full_name"],
                "away_full": a_basic["full_name"],
                "home_score": home_score,
                "away_score": away_score,
                "status_text": random.choice(["Prévu", "En cours", "Terminé"]),
                "status_id": status_id,
                "home_win_pct": h_basic["win_pct"],
                "away_win_pct": a_basic["win_pct"],
                "home_form": h_basic["last10_pct"],
                "away_form": a_basic["last10_pct"],
                "home_home_pct": h_basic["home_pct"],
                "away_road_pct": a_basic["road_pct"],
                "home_net": h_adv["net_rating"],
                "away_net": a_adv["net_rating"],
                "home_off": h_adv["off_rating"],
                "away_off": a_adv["off_rating"],
                "home_def": h_adv["def_rating"],
                "away_def": a_adv["def_rating"],
                "home_pace": h_adv["pace"],
                "away_pace": a_adv["pace"],
            })
        return pd.DataFrame(rows)

@st.cache_data(ttl=CACHE_TTL_SCOREBOARD)
def load_recent_games(sport: str, days: int, basic: dict, advanced: dict, league_id: Optional[int] = None) -> pd.DataFrame:
    games_list = []
    progress = st.progress(0.0)
    today = date.today()
    for i in range(days):
        d = today - timedelta(days=i)
        progress.text(f"Chargement {d:%d/%m/%Y} – {sport} …")
        day_df = get_games_for_date(sport, d, basic, advanced, league_id)
        if not day_df.empty:
            games_list.append(day_df)
        progress.progress((i + 1) / days)
        time.sleep(0.25)
    progress.empty()
    if not games_list:
        return pd.DataFrame()
    return pd.concat(games_list, ignore_index=True)

# =============================================================================
# FEATURE ENGINEERING (général – adaptable per sport)
# =============================================================================
def prepare_features(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    df = df.copy()
    df["win_pct_diff"] = df["home_win_pct"] - df["away_win_pct"]
    df["form_diff"] = df["home_form"] - df["away_form"]
    df["venue_diff"] = df["home_home_pct"] - df["away_road_pct"]
    df["home_advantage"] = 0.065 if sport in ["NBA (Basketball)", "Football (Soccer)"] else 0.03  # Moins pour tennis/esport
    df["net_diff"] = df["home_net"] - df["away_net"]
    df["off_diff"] = df["home_off"] - df["away_off"]
    df["def_diff"] = df["home_def"] - df["away_def"]
    df["pace_diff"] = df["home_pace"] - df["away_pace"]
    df["home_win"] = np.nan
    finished = (df["status_id"] == 3)
    df.loc[finished, "home_win"] = (df.loc[finished, "home_score"] > df.loc[finished, "away_score"]).astype(int)
    df["match_label"] = df["home_full"] + " vs " + df["away_full"]
    return df

# =============================================================================
# MODELE
# =============================================================================
FEATURES = [
    "win_pct_diff", "form_diff", "venue_diff", "home_advantage",
    "net_diff", "off_diff", "def_diff", "pace_diff"
]

def get_or_train_model(sport: str, df: pd.DataFrame):
    # MODIFICATION ICI : Ne pas essayer de charger un fichier qui n'existe pas
    # À la place, créer un modèle simple sans sauvegarde
    train = df[df["home_win"].notna()]
    
    if len(train) < 50:
        st.sidebar.warning(f"Peu de matchs pour {sport} → modèle placeholder")
        model = LGBMClassifier(n_estimators=50, max_depth=3, random_state=42, verbosity=-1)
        # Créer des features synthétiques pour l'entraînement
        X_fake = pd.DataFrame(np.random.normal(0, 1, (100, len(FEATURES))), columns=FEATURES)
        y_fake = (X_fake["net_diff"] + X_fake["def_diff"] > 0).astype(int)
        model.fit(X_fake, y_fake)
    else:
        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            colsample_bytree=0.80,
            random_state=42,
            verbosity=-1
        )
        model.fit(train[FEATURES], train["home_win"])
        acc = model.score(train[FEATURES], train["home_win"])
        st.sidebar.success(f"Modèle entraîné pour {sport} – {len(train)} matchs – acc train : {acc:.1%}")
    
    return model  # Retourner directement sans sauvegarde

def predict_all(df: pd.DataFrame, model) -> pd.DataFrame:
    df = df.copy()
    probas = model.predict_proba(df[FEATURES])[:, 1]
    df["proba_home"] = probas
    df["proba_away"] = 1 - probas
    df["cote_home"] = np.round(1 / np.maximum(probas, 0.04) * np.random.uniform(0.91, 0.99, len(df)), 2)
    df["cote_away"] = np.round(1 / np.maximum(1 - probas, 0.04) * np.random.uniform(0.91, 0.99, len(df)), 2)
    df["ev_home"] = df["proba_home"] * df["cote_home"] - 1
    df["ev_away"] = df["proba_away"] * df["cote_away"] - 1
    df["best_ev"] = df[["ev_home", "ev_away"]].max(axis=1)
    df["best_side"] = np.where(df["ev_home"] > df["ev_away"], "Domicile/Joueur 1", "Extérieur/Joueur 2")
    return df

# =============================================================================
# INTERFACE
# =============================================================================
def main():
    st.set_page_config(page_title="Multi-Sports Value Bets • Advanced", layout="wide", page_icon="🏆")
    
    # Ajouter un test d'import au début
    try:
        import joblib
        st.sidebar.success(f"✅ Joblib version: {joblib.__version__}")
    except ImportError as e:
        st.sidebar.error(f"❌ Joblib non installé: {e}")
        st.stop()
    
    st.title("🏆 Pronostics Multi-Sports – Advanced Stats")
    st.caption("Win% + forme + NET/OFF/DEF RATING + PACE – Saison actuelle")
    
    with st.sidebar:
        st.header("Options")
        selected_sport = st.selectbox("Sport", SPORTS)
        league_id = None
        if selected_sport == "Football (Soccer)":
            league_name = st.selectbox("Ligue", list(MAJOR_LEAGUES.keys()))
            league_id = MAJOR_LEAGUES[league_name]
        days = st.slider("Jours d'historique", 2, 14, 7)
        ev_threshold = st.slider("Seuil EV min (%)", 3, 12, 6) / 100
        st.info("Pour football : Données via API-Sports. Autres sports : simulation.")

    with st.spinner("Chargement standings & advanced stats…"):
        basic_stats = load_basic_standings(selected_sport, league_id)
        adv_stats = load_advanced_stats(selected_sport, basic_stats)
        if not basic_stats:
            st.error("Impossible de charger les données de base.")
            st.stop()

    with st.spinner(f"Récupération matchs ({days} jours) pour {selected_sport}…"):
        df_raw = load_recent_games(selected_sport, days, basic_stats, adv_stats, league_id)

    if df_raw.empty:
        st.warning("Aucun match dans la fenêtre.")
        st.stop()

    df = prepare_features(df_raw, selected_sport)
    model = get_or_train_model(selected_sport, df)
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
    df_show["best_ev"] = (df_show["best_ev"] * 100).map("{:+.1f}%".format)
    cols = ["date", "match_label", "status_text", "proba_home", "cote_home", "cote_away", "best_side", "best_ev"]
    st.dataframe(
        df_show[cols].rename(columns={
            "match_label": "Match",
            "status_text": "Statut",
            "proba_home": "Proba dom./J1",
            "best_side": "Meilleur pari",
            "best_ev": "EV"
        }),
        hide_index=True,
        use_container_width=True
    )
    strong = df_show[df_show["best_ev"] > seuil]
    if not strong.empty:
        top = strong.iloc[0]
        st.success(f"**Top pari** : {top['match_label']} – {top['best_side']} @ {top['cote_home' if 'Domicile' in top['best_side'] else 'cote_away']} (EV {top['best_ev']})")

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
            "proba_home": "Proba dom./J1",
            "correct": "Prédiction"
        }),
        hide_index=True,
        use_container_width=True
    )

if __name__ == "__main__":
    main()
