# Pronostiqueur NBA Optimise - Sans Timeout

# Utilise leaguestandings au lieu de boxscore pour eviter les timeouts

# Features: Win%, forme recente, facteur domicile

import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
import time
import random
from datetime import date, timedelta
from nba_api.stats.endpoints import scoreboardv2, leaguestandings
from nba_api.stats.static import teams
from lightgbm import LGBMClassifier

# ============================================================================

# CONFIGURATION

# ============================================================================

MODEL_PATH = nba_model.pkl
CACHE_TTL = 3600  # 1 heure

team_list = teams.get_teams()
TEAM_DICT = {team[‘id’]: team[‘abbreviation’] for team in team_list}

# ============================================================================

# RECUPERATION DES DONNEES

# ============================================================================

@st.cache_data(ttl=CACHE_TTL)
def get_team_statistics():
# Recupere les statistiques d’equipe depuis les standings
# Un seul appel API pour toutes les equipes (rapide)
try:
standings = leaguestandings.LeagueStandings()
standings_df = standings.get_data_frames()[0]

```
    stats_dict = {}
    for _, row in standings_df.iterrows():
        team_id = row['TeamID']
        
        # Parser les records
        wins_last10 = parse_record_string(row['L10'])
        
        stats_dict[team_id] = {
            'win_pct': float(row['WinPCT']),
            'last10_pct': wins_last10,
            'games_played': int(row['W']) + int(row['L'])
        }
    
    return stats_dict

except Exception as e:
    st.error(f"Erreur recuperation standings: {e}")
    return {}
```

def parse_record_string(record):
# Convertit ‘7-3’ en 0.70
try:
w, l = record.split(’-’)
total = int(w) + int(l)
return int(w) / total if total > 0 else 0.5
except:
return 0.5

def fetch_games_for_date(target_date, team_stats):
# Recupere les matchs pour une date donnee
# Utilise les stats pre-chargees (pas d’appel boxscore)
try:
date_str = target_date.strftime(”%Y-%m-%d”)
scoreboard = scoreboardv2.ScoreboardV2(game_date=date_str)
games_df = scoreboard.get_data_frames()[0]

```
    if games_df.empty:
        return pd.DataFrame()
    
    games_list = []
    for _, game in games_df.iterrows():
        home_id = game['HOME_TEAM_ID']
        away_id = game['VISITOR_TEAM_ID']
        
        # Recuperer stats depuis le dict pre-charge
        home_stats = team_stats.get(home_id, {})
        away_stats = team_stats.get(away_id, {})
        
        games_list.append({
            'game_id': game['GAME_ID'],
            'date': date_str,
            'home_id': home_id,
            'away_id': away_id,
            'home_team': TEAM_DICT.get(home_id, 'UNK'),
            'away_team': TEAM_DICT.get(away_id, 'UNK'),
            'home_score': game.get('PTS_HOME', 0),
            'away_score': game.get('PTS_AWAY', 0),
            'status': game['GAME_STATUS_TEXT'],
            'home_win_pct': home_stats.get('win_pct', 0.5),
            'away_win_pct': away_stats.get('win_pct', 0.5),
            'home_form': home_stats.get('last10_pct', 0.5),
            'away_form': away_stats.get('last10_pct', 0.5)
        })
    
    time.sleep(0.5)  # Rate limit
    return pd.DataFrame(games_list)

except Exception as e:
    st.warning(f"Erreur date {target_date}: {e}")
    return pd.DataFrame()
```

def fetch_multiple_days(num_days, team_stats):
# Recupere les matchs sur plusieurs jours
all_games = []

```
progress = st.progress(0)
status_text = st.empty()

for i in range(num_days):
    target_date = date.today() - timedelta(days=i)
    status_text.text(f"Chargement {target_date.strftime('%d/%m/%Y')}...")
    
    day_games = fetch_games_for_date(target_date, team_stats)
    if not day_games.empty:
        all_games.append(day_games)
    
    progress.progress((i + 1) / num_days)

progress.empty()
status_text.empty()

if all_games:
    return pd.concat(all_games, ignore_index=True)
return pd.DataFrame()
```

# ============================================================================

# PREPARATION DES FEATURES

# ============================================================================

def engineer_features(df):
# Cree les features pour le modele
df = df.copy()

```
# Features differentielles
df['win_pct_diff'] = df['home_win_pct'] - df['away_win_pct']
df['form_diff'] = df['home_form'] - df['away_form']
df['home_advantage'] = 0.06  # ~6% avantage domicile NBA

# Target (seulement pour matchs termines)
df['home_win'] = np.nan
finished_mask = ~df['status'].str.contains('PM|venir', case=False, na=False)
df.loc[finished_mask, 'home_win'] = (df.loc[finished_mask, 'home_score'] > 
                                      df.loc[finished_mask, 'away_score']).astype(int)

# Label match
df['match'] = df['home_team'] + ' vs ' + df['away_team']

return df
```

# ============================================================================

# MODELE MACHINE LEARNING

# ============================================================================

def train_or_load_model(df):
# Charge le modele existant ou en entraine un nouveau
features = [‘win_pct_diff’, ‘form_diff’, ‘home_advantage’]

```
# Tenter de charger modele existant
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        st.sidebar.success("Modele charge")
        return model, features
    except:
        st.sidebar.warning("Rechargement modele echoue")

# Entrainer nouveau modele
train_data = df[df['home_win'].notna()].copy()

if len(train_data) < 30:
    # Modele par defaut si donnees insuffisantes
    st.sidebar.warning("Donnees insuffisantes - modele par defaut")
    model = LGBMClassifier(n_estimators=50, random_state=42, verbosity=-1)
    X_dummy = pd.DataFrame({
        'win_pct_diff': np.random.uniform(-0.3, 0.3, 150),
        'form_diff': np.random.uniform(-0.3, 0.3, 150),
        'home_advantage': [0.06] * 150
    })
    y_dummy = (X_dummy['win_pct_diff'] + X_dummy['form_diff'] + 
               X_dummy['home_advantage'] + np.random.normal(0, 0.1, 150) > 0).astype(int)
    model.fit(X_dummy, y_dummy)
else:
    # Entrainement sur vraies donnees
    X = train_data[features]
    y = train_data['home_win']
    model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbosity=-1
    )
    model.fit(X, y)
    
    accuracy = (model.predict(X) == y).mean()
    st.sidebar.success(f"Modele entraine ({len(train_data)} matchs, precision: {accuracy:.1%})")

# Sauvegarder
joblib.dump(model, MODEL_PATH)
return model, features
```

def predict_games(df, model, features):
# Genere les predictions
df = df.copy()
X = df[features]

```
# Probabilites
probas = model.predict_proba(X)
df['proba_home'] = probas[:, 1]
df['proba_away'] = probas[:, 0]

# Cotes simulees (basees sur probas)
df['cote_home'] = df['proba_home'].apply(
    lambda p: round(1 / max(p, 0.05) * random.uniform(0.93, 0.97), 2)
)
df['cote_away'] = df['proba_away'].apply(
    lambda p: round(1 / max(p, 0.05) * random.uniform(0.93, 0.97), 2)
)

# Value betting
df['value_home'] = (df['proba_home'] * df['cote_home'] - 1) * 100
df['value_away'] = (df['proba_away'] * df['cote_away'] - 1) * 100

df['best_bet'] = df.apply(
    lambda r: 'Domicile' if r['value_home'] > r['value_away'] else 'Exterieur',
    axis=1
)
df['best_value'] = df[['value_home', 'value_away']].max(axis=1)
df['is_value'] = df['best_value'] > 5  # Seuil 5%

return df
```

# ============================================================================

# INTERFACE STREAMLIT

# ============================================================================

def main():
st.set_page_config(page_title=“NBA Pronostics”, page_icon=“🏀”, layout=“wide”)

```
st.title("🏀 Pronostiqueur NBA")
st.caption("Predictions basees sur statistiques reelles via NBA API")

# Sidebar
st.sidebar.header("Parametres")
days = st.sidebar.slider("Jours de donnees", 1, 10, 5)
value_threshold = st.sidebar.slider("Seuil value bet (%)", 0, 15, 5)

# Recuperation donnees
with st.spinner("Chargement des statistiques NBA..."):
    team_stats = get_team_statistics()
    
    if not team_stats:
        st.error("Impossible de recuperer les donnees NBA")
        st.stop()
    
    st.success(f"Stats chargees pour {len(team_stats)} equipes")

with st.spinner("Recuperation des matchs..."):
    games_df = fetch_multiple_days(days, team_stats)
    
    if games_df.empty:
        st.warning("Aucun match trouve - Utilisation de donnees simulees")
        games_df = create_dummy_data()

# Preparation
games_df = engineer_features(games_df)
model, features = train_or_load_model(games_df)
predictions = predict_games(games_df, model, features)

# Filtres
upcoming = predictions[predictions['status'].str.contains('PM|venir', case=False, na=False)]
completed = predictions[predictions['home_win'].notna()]

# Affichage
tab1, tab2, tab3 = st.tabs(["Matchs a venir", "Historique", "Statistiques"])

with tab1:
    display_upcoming_games(upcoming, value_threshold)

with tab2:
    display_completed_games(completed)

with tab3:
    display_statistics(completed, model, features)
```

def display_upcoming_games(df, threshold):
# Affiche les matchs a venir
st.header(“Pronostics du jour”)

```
if df.empty:
    st.info("Aucun match prevu aujourd'hui")
    return

# Tri par value
df = df.sort_values('best_value', ascending=False)

# Tableau
display_df = df[[
    'match', 'proba_home', 'cote_home', 'cote_away', 
    'best_bet', 'best_value', 'is_value'
]].copy()

display_df['proba_home'] = display_df['proba_home'].apply(lambda x: f"{x:.1%}")
display_df['best_value'] = display_df['best_value'].apply(lambda x: f"{x:+.1f}%")
display_df['is_value'] = display_df['is_value'].map({True: '⭐', False: ''})

display_df.columns = ['Match', 'Proba Dom.', 'Cote Dom.', 'Cote Ext.', 
                      'Meilleur pari', 'Value %', 'Value?']

st.dataframe(display_df, use_container_width=True, hide_index=True)

# Meilleur pari
best = df.iloc[0]
if best['best_value'] > threshold:
    st.success(f"Pari recommande: {best['match']} - Prediction: {best['best_bet']} ({best['proba_home']:.1%} domicile) - Value: {best['best_value']:+.1f}%")
else:
    st.info("Aucune value bet significative detectee")
```

def display_completed_games(df):
# Affiche l’historique
st.header(“Matchs recents”)

```
if df.empty:
    st.info("Aucun match recent")
    return

df = df.sort_values('date', ascending=False).head(20)

display_df = df[[
    'date', 'match', 'home_score', 'away_score', 
    'proba_home', 'home_win'
]].copy()

display_df['proba_home'] = display_df['proba_home'].apply(lambda x: f"{x:.1%}")
display_df['correct'] = ((display_df['proba_home'].str.rstrip('%').astype(float) > 50) == 
                          (display_df['home_win'] == 1))
display_df['correct'] = display_df['correct'].map({True: 'OK', False: 'KO'})

display_df.columns = ['Date', 'Match', 'Score Dom.', 'Score Ext.', 
                      'Proba Dom.', 'Victoire Dom.', 'Correct']

st.dataframe(display_df.drop('Victoire Dom.', axis=1), use_container_width=True, hide_index=True)
```

def display_statistics(df, model, features):
# Affiche les statistiques du modele
st.header(“Performance du modele”)

```
if len(df) < 10:
    st.warning("Donnees insuffisantes pour statistiques")
    return

col1, col2, col3 = st.columns(3)

# Accuracy
predictions = model.predict(df[features])
accuracy = (predictions == df['home_win']).mean()
col1.metric("Precision globale", f"{accuracy:.1%}")

# Win% domicile
home_win_rate = df['home_win'].mean()
col2.metric("Victoires domicile", f"{home_win_rate:.1%}")

# Matchs analyses
col3.metric("Matchs analyses", len(df))

# Importance features
st.subheader("Importance des features")
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

st.bar_chart(importance_df.set_index('Feature'))
```

def create_dummy_data():
# Cree des donnees simulees en cas d’echec API
return pd.DataFrame([
{
‘game_id’: ‘sim1’, ‘date’: date.today().strftime(’%Y-%m-%d’),
‘home_id’: 1610612747, ‘away_id’: 1610612738,
‘home_team’: ‘LAL’, ‘away_team’: ‘BOS’,
‘home_score’: 0, ‘away_score’: 0, ‘status’: ‘7:00 PM ET’,
‘home_win_pct’: 0.62, ‘away_win_pct’: 0.58,
‘home_form’: 0.70, ‘away_form’: 0.60
},
{
‘game_id’: ‘sim2’, ‘date’: date.today().strftime(’%Y-%m-%d’),
‘home_id’: 1610612744, ‘away_id’: 1610612751,
‘home_team’: ‘GSW’, ‘away_team’: ‘BKN’,
‘home_score’: 0, ‘away_score’: 0, ‘status’: ‘7:30 PM ET’,
‘home_win_pct’: 0.55, ‘away_win_pct’: 0.48,
‘home_form’: 0.50, ‘away_form’: 0.40
}
])

if **name** == “**main**”:
main()