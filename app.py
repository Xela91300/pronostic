import streamlit as st
import pandas as pd
import joblib
import random
from datetime import datetime

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pronos IA Réels + Value Bets", layout="wide")
st.title("Pronostics IA – Matchs réels, Modèle LightGBM & Value Bets")
st.caption(f"Mis à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')} | Massy, FR")

tab_foot, tab_tennis, tab_nba = st.tabs(["Football ⚽", "Tennis 🎾", "NBA 🏀"])

# ─── CHARGEMENT DU MODELE LIGHTGBM ─────────────────────────────────────────
@st.cache_resource
def load_lightgbm_model():
    try:
        model = joblib.load("football_model.pkl")
        st.success("Modèle LightGBM chargé avec succès !")
        return model
    except Exception as e:
        st.warning("Modèle non trouvé → mode simulation activé")
        return None

model = load_lightgbm_model()

# ─── DONNÉES RÉELLES (scraping léger via CSV public football-data.co.uk) ────
@st.cache_data(ttl=3600)  # refresh toutes les heures
def load_real_football_data():
    try:
        # Dernière saison ou actuelle – URL publique
        url = "https://www.football-data.co.uk/mmz4281/2425/E0.csv"  # Premier League 24/25
        df = pd.read_csv(url)
        df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].tail(20)  # 20 derniers matchs
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df["proba_home"] = 0.55 + random.uniform(-0.10, 0.15)  # simulation base
        return df
    except:
        return pd.DataFrame()

# ─── PREDICTION AVEC MODELE OU SIMULATION ──────────────────────────────────
def predict_proba(row):
    if model is None:
        return random.uniform(0.45, 0.85)
    try:
        # Adaptez les features à ton modèle entraîné
        features = pd.DataFrame([{
            "FTHG": row.get("FTHG", 0),
            "FTAG": row.get("FTAG", 0),
            # Ajoute tes vraies features si tu les as
        }])
        proba = model.predict_proba(features)[0][1]  # classe 1 = home win
        return proba
    except:
        return random.uniform(0.45, 0.85)

# ─── CALCUL VALUE BET ──────────────────────────────────────────────────────
def add_value_bets(df):
    df["cote_home_sim"] = df["proba_home"].apply(lambda p: round(1 / p * random.uniform(0.92, 0.98), 2))
    df["value"] = df["proba_home"] * df["cote_home_sim"] - 1
    df["value_pct"] = df["value"] * 100
    return df

# ─── ONGLETS ───────────────────────────────────────────────────────────────

with tab_foot:
    st.subheader("Football – Matchs récents / simulés live")
    df_foot = load_real_football_data()

    if df_foot.empty:
        st.info("Impossible de charger les données réelles → simulation activée")
        df_foot = pd.DataFrame([
            {"Date": "2026-01-24", "HomeTeam": "PSG", "AwayTeam": "Lyon", "FTHG": 2, "FTAG": 1},
            {"Date": "2026-01-24", "HomeTeam": "Arsenal", "AwayTeam": "Man Utd", "FTHG": 1, "FTAG": 1},
        ])

    df_foot["proba_home"] = df_foot.apply(predict_proba, axis=1)
    df_foot = add_value_bets(df_foot)

    # Pronostic le plus sûr
    safest = df_foot.loc[df_foot["proba_home"].idxmax()]
    st.success(f"**Pronostic le plus sûr** : Victoire **{safest['HomeTeam']}** vs {safest['AwayTeam']} → {safest['proba_home']:.0%}")

    # Tableau avec value bets
    disp = df_foot[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "proba_home", "cote_home_sim", "value_pct"]].copy()
    disp.columns = ["Date", "Domicile", "Extérieur", "Buts D", "Buts E", "Proba Domicile", "Cote D simulée", "Value %"]
    disp["Proba Domicile"] = disp["Proba Domicile"].apply(lambda x: f"{x:.0%}")
    disp["Cote D simulée"] = disp["Cote D simulée"].round(2)
    disp["Value %"] = disp["Value %"].round(1).apply(lambda x: f"+{x}%" if x > 5 else f"{x}%")

    def highlight_value(row):
        if float(row["Value %"][:-1]) > 5:
            return ["background-color: #ccffcc"] * len(row)
        return [""] * len(row)

    st.dataframe(disp.style.apply(highlight_value, axis=1), use_container_width=True)


with tab_tennis:
    st.subheader("Tennis – Simulation (pas de scraping fiable gratuit)")
    st.info("Pour tennis réel : utilise API-Sports Tennis ou Jeff Sackmann GitHub datasets")
    # Exemple statique
    df_tennis = pd.DataFrame([
        {"match": "Alcaraz vs Paul", "proba_alcaraz": 0.78},
        {"match": "Swiatek vs Kalinskaya", "proba_swiatek": 0.88},
    ])
    safest_t = df_tennis.loc[df_tennis["proba_alcaraz"].idxmax() if "proba_alcaraz" in df_tennis else 0]
    st.success(f"**Plus sûr** : {safest_t['match']} → {safest_t.get('proba_alcaraz', safest_t.get('proba_swiatek', 0.80)):.0%}")
    st.dataframe(df_tennis)


with tab_nba:
    st.subheader("NBA – Simulation (pas de scraping fiable gratuit)")
    st.info("Pour NBA réel : utilise nba_api ou basketball-reference scraping")
    df_nba = pd.DataFrame([
        {"match": "Timberwolves vs Warriors", "proba_home": 0.68},
        {"match": "Celtics vs Bulls", "proba_home": 0.62},
    ])
    safest_n = df_nba.loc[df_nba["proba_home"].idxmax()]
    st.success(f"**Plus sûr** : Victoire domicile {safest_n['match']} → {safest_n['proba_home']:.0%}")
    st.dataframe(df_nba)

st.caption("Pour cotes réelles → The Odds API (clé à tester). Pour scraping live → Selenium/BeautifulSoup mais pas sur Streamlit Cloud (bloqué). Contacte-moi pour adapter avec une API qui marche !")
