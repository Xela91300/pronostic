import streamlit as st
import pandas as pd
import time
from datetime import datetime
import random  # Pour simuler l'IA "en temps réel"

# ─── CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pronos IA Live - Foot/Tennis/NBA", layout="wide")
st.title("Pronostics IA en Temps Réel ⚽🎾🏀")
st.caption(f"Mis à jour le {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} CET | Refresh auto toutes les X secondes")

# Sidebar
sport = st.sidebar.selectbox("Sport", ["Football", "Tennis", "NBA"])
refresh_sec = st.sidebar.slider("Rafraîchissement automatique (secondes)", 30, 180, 60)
st.sidebar.caption("⚠️ Ne descends pas trop bas (quota API gratuit limité)")

placeholder = st.empty()  # Zone qui se rafraîchit

# ─── DONNÉES SIMULÉES / RÉELLES DU JOUR (remplace par API quand tu veux) ─────
def get_live_matches(sport):
    today = datetime.now().strftime("%Y-%m-%d")
    
    if sport == "Football":
        # Exemple plausibles 24/01/2026 - remplace par API réelle
        data = [
            {"match": "Manchester City vs Wolves", "score": "2-0", "status": "LIVE 65'", "proba_home": 0.85},
            {"match": "OM vs Lens", "score": "1-1", "status": "LIVE 45'", "proba_home": 0.55},
            {"match": "Rennes vs Lorient", "score": "0-0", "status": "À venir 19:00", "proba_home": 0.65},
            {"match": "Leverkusen vs Brême", "score": "-", "status": "À venir 20:30", "proba_home": 0.70},
        ]
    
    elif sport == "Tennis":
        data = [
            {"match": "Casper Ruud vs Marin Cilic", "score": "6-4 4-2", "status": "LIVE", "proba_p1": 0.72},
            {"match": "Elena Rybakina vs Tereza Valentova", "score": "6-2 3-1", "status": "LIVE", "proba_p1": 0.80},
            {"match": "Iga Swiatek vs Anna Kalinskaya", "score": "-", "status": "À venir 09:00", "proba_p1": 0.88},
            {"match": "Naomi Osaka vs Maddison Inglis", "score": "-", "status": "Terminé", "proba_p1": 0.75},
        ]
    
    else:  # NBA
        data = [
            {"match": "Minnesota vs Golden State", "score": "58-52", "status": "Q3", "proba_home": 0.62},
            {"match": "Miami vs Utah", "score": "45-38", "status": "Mi-temps", "proba_home": 0.68},
            {"match": "Boston vs Chicago", "score": "-", "status": "À venir 20:00", "proba_home": 0.58},
            {"match": "Lakers vs Dallas", "score": "-", "status": "À venir 20:30", "proba_home": 0.52},
        ]
    
    df = pd.DataFrame(data)
    return df

# ─── IA PRONOSTIC EN TEMPS RÉEL ─────────────────────────────────────────────
def ia_pronostic(row, sport):
    if sport == "Football":
        base = row["proba_home"]
        # Ajuste en live : si mène déjà → boost, si mené → baisse
        if "LIVE" in row["status"]:
            if row["score"].startswith("2-") or row["score"].startswith("3-"):
                base += 0.10
            elif row["score"].endswith("-2") or row["score"].endswith("-3"):
                base -= 0.15
        proba = min(0.95, max(0.30, base + random.uniform(-0.08, 0.08)))
        return f"🏠 Victoire domicile : **{proba:.0%}**"
    
    elif sport == "Tennis":
        proba = min(0.95, max(0.40, row["proba_p1"] + random.uniform(-0.06, 0.06)))
        return f"🎾 Vainqueur P1 : **{proba:.0%}**"
    
    else:  # NBA
        proba = min(0.90, max(0.45, row["proba_home"] + random.uniform(-0.07, 0.07)))
        return f"🏀 Victoire domicile : **{proba:.0%}**"

# ─── BOUCLE TEMPS RÉEL ─────────────────────────────────────────────────────
while True:
    with placeholder.container():
        df = get_live_matches(sport)
        
        if df.empty:
            st.info("Aucun match disponible pour le moment...")
        else:
            st.subheader(f"{sport} – Live & À venir ({len(df)} matchs)")
            
            # Ajoute colonne Prono IA
            df["Pronostic IA (refresh en direct)"] = df.apply(lambda row: ia_pronostic(row, sport), axis=1)
            
            # Mise en forme
            def style_live(row):
                if "LIVE" in row["status"] or "'" in row["status"] or "Q" in row["status"]:
                    return ['background-color: #e6ffe6'] * len(row)
                return [''] * len(row)
            
            st.dataframe(
                df.style.apply(style_live, axis=1),
                use_container_width=True,
                hide_index=True
            )
            
            st.caption("L'IA recalcule à chaque refresh → probas évoluent en live !")
    
    time.sleep(refresh_sec)
    st.rerun()  # Rafraîchit toute la page automatiquement
