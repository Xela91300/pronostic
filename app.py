import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
from datetime import datetime
import random  # simulation cotes / features

# ─── CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pronos IA Live + Value Bets", layout="wide")
st.title("Pronostics IA Temps Réel + Value Bets ⚽🎾🏀")
st.caption(f"Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | Auto-refresh")

# Charger le modèle LightGBM (football pour l'exemple – adapte pour tennis/nba)
@st.cache_resource
def load_model():
    try:
        model = joblib.load("football_model_calibrated.pkl")  # ou "models/football_model_calibrated.pkl"
        st.success("Modèle LightGBM chargé avec succès !")
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle : {e}")
        st.info("Utilisation d'une simulation par défaut. Crée un .pkl avec joblib.dump()")
        return None

model = load_model()

# Sidebar
sport = st.sidebar.selectbox("Sport", ["Football", "Tennis", "NBA"])
ligues = st.sidebar.multiselect(
    "Filtrer par ligue/tournoi",
    options=["Premier League", "Ligue 1", "Bundesliga", "ATP Australian Open", "WTA AO", "NBA", "EuroLeague"],
    default=["Premier League", "Ligue 1", "ATP Australian Open", "NBA"]
)
refresh_sec = st.sidebar.slider("Rafraîchissement (s)", 30, 180, 60)
value_threshold = st.sidebar.slider("Seuil Value Bet (%)", 3, 15, 5)

placeholder = st.empty()

# ─── DONNÉES MATCHS SIMULÉES (remplace par API réelle) ──────────────────────
def get_matches(sport, selected_ligues):
    data = []
    
    if sport == "Football":
        raw = [
            {"match": "Man City vs Wolves", "ligue": "Premier League", "status": "LIVE 65'", "home_xG": 2.1, "away_xG": 0.4, "form_home": 8, "form_away": 3},
            {"match": "OM vs Lens", "ligue": "Ligue 1", "status": "LIVE 45'", "home_xG": 1.3, "away_xG": 1.1, "form_home": 6, "form_away": 5},
            {"match": "Rennes vs Lorient", "ligue": "Ligue 1", "status": "À venir", "home_xG": 1.6, "away_xG": 0.9, "form_home": 7, "form_away": 4},
            {"match": "Leverkusen vs Brême", "ligue": "Bundesliga", "status": "À venir", "home_xG": 2.0, "away_xG": 1.2, "form_home": 9, "form_away": 5},
        ]
    elif sport == "Tennis":
        raw = [
            {"match": "Ruud vs Cilic", "ligue": "ATP Australian Open", "status": "LIVE", "rank_diff": -120, "surface_win_p1": 0.78},
            {"match": "Rybakina vs Valentova", "ligue": "WTA AO", "status": "LIVE", "rank_diff": -450, "surface_win_p1": 0.85},
        ]
    else:  # NBA
        raw = [
            {"match": "Minnesota vs Golden State", "ligue": "NBA", "status": "Q3", "net_rating_home": 8.2, "rest_diff": 1},
            {"match": "Miami vs Utah", "ligue": "NBA", "status": "Mi-temps", "net_rating_home": 6.5, "rest_diff": 0},
        ]
    
    df = pd.DataFrame(raw)
    if not df.empty and "ligue" in df.columns:
        df = df[df["ligue"].isin(selected_ligues)]
    return df

# ─── SIMULATION COTES BOOKMAKER (remplace par The Odds API ou API-Sports odds/live) ──
def get_bookmaker_odds(proba_home):
    # Cote implicite = 1 / proba + marge ~5-8%
    margin = random.uniform(0.05, 0.08)
    cote_home = round(1 / (proba_home * (1 - margin)), 2) if proba_home > 0.05 else 50.0
    return cote_home

# ─── FEATURE ENGINEERING SIMPLIFIÉ + PRÉDICTION ─────────────────────────────
def predict_with_model(df, sport):
    if model is None:
        # Fallback simulation
        df["proba_home"] = [random.uniform(0.45, 0.85) for _ in df.index]
    else:
        if sport == "Football":
            # Exemple features attendues par ton modèle
            X = df[["home_xG", "away_xG", "form_home", "form_away"]].fillna(0)
            df["proba_home"] = model.predict_proba(X)[:, 0]  # classe 0 = home win (adapte selon ton target)
        elif sport == "Tennis":
            X = df[["rank_diff", "surface_win_p1"]].fillna(0)
            df["proba_home"] = model.predict_proba(X)[:, 1]  # adapte
        else:
            X = df[["net_rating_home", "rest_diff"]].fillna(0)
            df["proba_home"] = model.predict_proba(X)[:, 1]
    
    return df

# ─── CALCUL VALUE BET ──────────────────────────────────────────────────────
def calculate_value(row):
    proba = row["proba_home"]
    cote = row["cote_book"]
    if cote <= 1.01 or proba <= 0.01:
        return 0.0
    value = (proba * cote) - 1
    return value

# ─── BOUCLE TEMPS RÉEL ─────────────────────────────────────────────────────
while True:
    with placeholder.container():
        df = get_matches(sport, ligues)
        
        if df.empty:
            st.info("Aucun match correspond aux filtres ou au sport sélectionné.")
        else:
            st.subheader(f"{sport} – {len(df)} matchs (ligues filtrées)")
            
            # Prédiction IA (recalcul à chaque refresh)
            df = predict_with_model(df, sport)
            
            # Cotes simulées
            df["cote_book"] = df["proba_home"].apply(get_bookmaker_odds)
            
            # Value bets
            df["value"] = df.apply(calculate_value, axis=1)
            df["value_pct"] = df["value"] * 100
            
            # Affichage
            cols_to_show = ["match", "ligue", "status", "proba_home", "cote_book", "value_pct"]
            display_df = df[cols_to_show].copy()
            display_df["proba_home"] = display_df["proba_home"].apply(lambda x: f"{x:.0%}")
            display_df["cote_book"] = display_df["cote_book"].apply(lambda x: f"{x:.2f}")
            display_df["value_pct"] = display_df["value_pct"].apply(lambda x: f"+{x:.1f}%" if x > value_threshold else f"{x:.1f}%")
            
            def highlight_value(row):
                if row["value_pct"].startswith("+") and float(row["value_pct"][1:-1]) > value_threshold:
                    return ['background-color: #90EE90'] * len(row)
                return [''] * len(row)
            
            st.dataframe(
                display_df.style.apply(highlight_value, axis=1),
                use_container_width=True,
                hide_index=True
            )
            
            st.caption("Value bet = (proba × cote) - 1 → vert si > seuil choisi")
    
    time.sleep(refresh_sec)
    st.rerun()
