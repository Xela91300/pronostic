import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# ─── CONFIGURATION DE LA PAGE ───────────────────────────────────────────────
st.set_page_config(page_title="Pronos IA - Foot / Tennis / NBA", layout="wide")
st.title("Pronostics IA – Football • Tennis • NBA")
st.markdown("Sélectionne un sport et remplis les infos pour obtenir une prédiction rapide.")

# ─── SIDEBAR : CHOIX DU SPORT ──────────────────────────────────────────────
sport = st.sidebar.selectbox("Choisis le sport", ["Football", "Tennis", "NBA"])

# ─── FONCTIONS DE PRÉDICTION SIMPLIFIÉES (à adapter avec tes vrais modèles) ──

def predict_football(home, away):
    # Ici tu mets ton modèle entraîné ou une simulation réaliste
    # Pour tester : valeurs fictives plausibles
    proba_home = 0.58
    proba_draw = 0.24
    proba_away = 0.18
    return {
        "Victoire domicile (1)": f"{proba_home:.1%}",
        "Nul (N)": f"{proba_draw:.1%}",
        "Victoire extérieur (2)": f"{proba_away:.1%}"
    }

def predict_tennis(player1, player2):
    proba_p1 = 0.67
    return f"Probabilité de victoire de **{player1}** : {proba_p1:.1%}"

def predict_nba(home_team, away_team):
    proba_home_win = 0.62
    total_points = 228.4
    return {
        "Victoire domicile": f"{proba_home_win:.1%}",
        "Total points estimé": f"{total_points:.1f}"
    }

# ─── INTERFACE SELON LE SPORT CHOISI ───────────────────────────────────────

if sport == "Football":
    st.subheader("Football – Prédiction 1X2")
    col1, col2 = st.columns(2)
    with col1:
        home = st.text_input("Équipe domicile", "PSG")
    with col2:
        away = st.text_input("Équipe extérieur", "Lyon")

    if st.button("Prédire le match", type="primary"):
        if home.strip() and away.strip():
            result = predict_football(home, away)
            st.success("Prédiction :")
            for key, value in result.items():
                st.metric(key, value)
        else:
            st.warning("Remplis les deux équipes !")

elif sport == "Tennis":
    st.subheader("Tennis – Probabilité de victoire (ATP/WTA)")
    col1, col2 = st.columns(2)
    with col1:
        p1 = st.text_input("Joueur 1 (favori ?)", "Djokovic")
    with col2:
        p2 = st.text_input("Joueur 2", "Alcaraz")

    if st.button("Prédire", type="primary"):
        if p1.strip() and p2.strip():
            result = predict_tennis(p1, p2)
            st.success(result)
        else:
            st.warning("Indique les deux joueurs !")

elif sport == "NBA":
    st.subheader("NBA – Moneyline + Total points")
    col1, col2 = st.columns(2)
    with col1:
        home_nba = st.text_input("Équipe domicile", "Boston Celtics")
    with col2:
        away_nba = st.text_input("Équipe extérieur", "Los Angeles Lakers")

    if st.button("Prédire", type="primary"):
        if home_nba.strip() and away_nba.strip():
            result = predict_nba(home_nba, away_nba)
            st.success("Prédiction :")
            for key, value in result.items():
                st.metric(key, value)
        else:
            st.warning("Remplis les deux équipes !")

# ─── PIED DE PAGE ──────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Application prototype – Remplace les fonctions de prédiction par tes modèles LightGBM réels")
st.caption("Pour lancer : `streamlit run app.py` dans ton terminal")
