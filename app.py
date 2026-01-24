import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import joblib
import random
from datetime import datetime, date

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pronos IA NBA Réel + Value", layout="wide")
st.title("Pronostics NBA – Scraping Basketball-Reference + Modèle LightGBM")
st.caption(f"Mis à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')} CET | Massy, FR")

refresh_sec = st.sidebar.slider("Rafraîchissement (secondes)", 120, 600, 300)  # 5 min recommandé
value_threshold = st.sidebar.slider("Seuil Value Bet (%)", 5, 20, 10)

placeholder = st.empty()

# ─── CHARGEMENT MODELE LIGHTGBM ────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("nba_model.pkl")  # change pour ton fichier .pkl
        st.success("Modèle LightGBM NBA chargé !")
        return model
    except Exception as e:
        st.warning("Modèle non trouvé → simulation proba activée")
        return None

model = load_model()

# ─── SCRAPING BASKETBALL-REFERENCE POUR MATCHS DU JOUR ─────────────────────
@st.cache_data(ttl=refresh_sec - 30)
def scrape_nba_games_today():
    today = date.today().strftime("%Y-%m-%d")
    url = f"https://www.basketball-reference.com/boxscores/?month={date.today().month}&day={date.today().day}&year={date.today().year}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code != 200:
            st.error(f"Erreur scraping : HTTP {resp.status_code}")
            return pd.DataFrame()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        games = []
        for game in soup.find_all("div", class_="game"):
            try:
                teams = game.find_all("a", class_="team-name")
                if len(teams) >= 2:
                    home = teams[1].text.strip() if len(teams) > 1 else "?"
                    away = teams[0].text.strip()
                    score_tag = game.find("div", class_="score")
                    score = score_tag.text.strip() if score_tag else "-"
                    status = game.find("div", class_="status") or "À venir"
                    status_text = status.text.strip() if hasattr(status, "text") else status
                    
                    games.append({
                        "match": f"{away} @ {home}",
                        "score": score,
                        "status": status_text,
                        "home_team": home,
                        "away_team": away
                    })
            except:
                continue
        
        df = pd.DataFrame(games)
        if df.empty:
            st.info("Aucun match trouvé aujourd'hui sur Basketball-Reference → simulation activée")
        return df
    except Exception as e:
        st.error(f"Erreur scraping : {str(e)}")
        return pd.DataFrame()

# ─── PREDICTION PROBA (modèle ou simulation) ───────────────────────────────
def predict_home_win_proba(row):
    if model is None:
        # Simulation réaliste
        base_proba = 0.55
        if "home" in row["match"].lower(): base_proba += 0.08  # home advantage
        return round(random.uniform(base_proba - 0.10, base_proba + 0.15), 3)
    
    try:
        # Adaptez à tes features réelles (ex: net rating, rest days...)
        features = pd.DataFrame([{
            "home_adv": 1.0,  # dummy
            "pace_diff": random.uniform(-5, 5)
        }])
        proba = model.predict_proba(features)[0][1]  # classe 1 = home win
        return round(proba, 3)
    except:
        return 0.55

# ─── AJOUT COTES SIMULEES + VALUE BETS ─────────────────────────────────────
def add_value_bets(df):
    df["proba_home"] = df.apply(predict_home_win_proba, axis=1)
    df["cote_home_sim"] = df["proba_home"].apply(lambda p: round(1 / (p * random.uniform(0.90, 0.97)), 2) if p > 0.1 else 5.0)
    df["value"] = df["proba_home"] * df["cote_home_sim"] - 1
    df["value_pct"] = df["value"] * 100
    return df

# ─── BOUCLE TEMPS RÉEL ─────────────────────────────────────────────────────
while True:
    with placeholder.container():
        df_nba = scrape_nba_games_today()
        
        if df_nba.empty:
            # Fallback simulation si scraping échoue
            df_nba = pd.DataFrame([
                {"match": "Minnesota @ Golden State", "score": "58-52 Q3", "status": "LIVE"},
                {"match": "Miami @ Utah", "score": "45-38 HT", "status": "Mi-temps"},
                {"match": "Boston vs Chicago", "score": "-", "status": "À venir"},
            ])
        
        df_nba = add_value_bets(df_nba)
        
        # Pronostic le plus sûr
        if not df_nba.empty:
            safest = df_nba.loc[df_nba["proba_home"].idxmax()]
            st.success(f"**Pronostic le plus sûr** : Victoire domicile **{safest['match']}** → {safest['proba_home']:.0%}")
            st.markdown(f"Score actuel : {safest['score']} | Statut : {safest['status']}")
        else:
            st.info("Aucun match NBA récupéré aujourd'hui.")
        
        # Tableau avec value bets
        disp = df_nba[["match", "score", "status", "proba_home", "cote_home_sim", "value_pct"]].copy()
        disp.columns = ["Match", "Score", "Statut", "Proba Domicile", "Cote D simulée", "Value %"]
        disp["Proba Domicile"] = disp["Proba Domicile"].apply(lambda x: f"{x:.0%}")
        disp["Cote D simulée"] = disp["Cote D simulée"].round(2)
        disp["Value %"] = disp["Value %"].round(1).apply(lambda x: f"+{x}%" if x > value_threshold else f"{x}%")
        
        def highlight_value(row):
            val = float(row["Value %"][:-1]) if row["Value %"].endswith("%") else 0
            if val > value_threshold:
                return ["background-color: #d4edda"] * len(row)
            return [""] * len(row)
        
        st.dataframe(
            disp.style.apply(highlight_value, axis=1),
            use_container_width=True,
            hide_index=True
        )
        
        st.caption("Scraping depuis Basketball-Reference.com – Proba via LightGBM ou simulation – Value = (proba × cote) - 1")
    
    time.sleep(refresh_sec)
    st.rerun()
