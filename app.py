import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
import re
import random
import pytz

# Configuration de la page
st.set_page_config(
    page_title="⚽ Football Analytics Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation des variables de session
if 'scraper' not in st.session_state:
    st.session_state.scraper = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# CSS personnalisé
def load_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .live-badge {
        background: #ff4757;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .match-row {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s;
    }
    
    .match-row:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CLASSE DE SCRAPING SIMPLIFIÉE ET FIABLE
# ============================================================================
class FootballScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
        }
    
    def get_matches(self, source="demo", league="Ligue 1"):
        """Récupère les matchs selon la source choisie"""
        if source == "demo":
            return self.get_demo_matches(league)
        elif source == "worldfootball":
            return self.scrape_worldfootball()
        elif source == "football_api":
            return self.get_api_matches()
        else:
            return self.get_demo_matches(league)
    
    def get_demo_matches(self, league="Ligue 1"):
        """Retourne des données de démo réalistes"""
        # Définir les équipes par ligue
        leagues_data = {
            "Ligue 1": {
                "teams": ["Paris SG", "Marseille", "Lyon", "Monaco", "Lille", "Nice", "Rennes", "Lens", "Strasbourg", "Montpellier"],
                "matches": [
                    ("Paris SG", "Marseille", 2, 1),
                    ("Lyon", "Lille", 1, 1),
                    ("Monaco", "Nice", 3, 2),
                    ("Rennes", "Lens", 2, 0),
                    ("Strasbourg", "Montpellier", 1, 0)
                ]
            },
            "Premier League": {
                "teams": ["Manchester City", "Liverpool", "Arsenal", "Chelsea", "Tottenham", "Manchester United", "Newcastle", "Aston Villa", "Brighton", "West Ham"],
                "matches": [
                    ("Manchester City", "Liverpool", 1, 1),
                    ("Arsenal", "Chelsea", 2, 2),
                    ("Tottenham", "Manchester United", 2, 1),
                    ("Newcastle", "Aston Villa", 3, 1),
                    ("Brighton", "West Ham", 2, 0)
                ]
            },
            "La Liga": {
                "teams": ["Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Valencia", "Real Betis", "Villarreal", "Athletic Bilbao", "Real Sociedad", "Celta Vigo"],
                "matches": [
                    ("Real Madrid", "Barcelona", 2, 1),
                    ("Atletico Madrid", "Sevilla", 1, 0),
                    ("Valencia", "Real Betis", 2, 2),
                    ("Villarreal", "Athletic Bilbao", 3, 1),
                    ("Real Sociedad", "Celta Vigo", 2, 0)
                ]
            },
            "Bundesliga": {
                "teams": ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen", "Eintracht Frankfurt", "Wolfsburg", "Borussia Mönchengladbach", "Freiburg", "Union Berlin", "Mainz"],
                "matches": [
                    ("Bayern Munich", "Borussia Dortmund", 4, 2),
                    ("RB Leipzig", "Bayer Leverkusen", 3, 1),
                    ("Eintracht Frankfurt", "Wolfsburg", 2, 2),
                    ("Borussia Mönchengladbach", "Freiburg", 1, 0),
                    ("Union Berlin", "Mainz", 2, 1)
                ]
            },
            "Serie A": {
                "teams": ["Juventus", "AC Milan", "Inter Milan", "Napoli", "Roma", "Lazio", "Atalanta", "Fiorentina", "Bologna", "Torino"],
                "matches": [
                    ("Juventus", "AC Milan", 1, 0),
                    ("Inter Milan", "Napoli", 2, 2),
                    ("Roma", "Lazio", 3, 2),
                    ("Atalanta", "Fiorentina", 2, 1),
                    ("Bologna", "Torino", 1, 1)
                ]
            }
        }
        
        # Récupérer les données de la ligue
        league_data = leagues_data.get(league, leagues_data["Ligue 1"])
        teams = league_data["teams"]
        base_matches = league_data["matches"]
        
        matches = []
        status_options = ['NS', 'LIVE', 'HT', 'FT']
        weights = [0.3, 0.2, 0.1, 0.4]  # Probabilités
        
        # Générer 8-10 matchs
        num_matches = random.randint(8, 10)
        
        for i in range(num_matches):
            if i < len(base_matches):
                # Utiliser les matchs prédéfinis
                home, away, h_score, a_score = base_matches[i]
            else:
                # Générer des matchs aléatoires
                home = random.choice(teams)
                away = random.choice([t for t in teams if t != home])
                h_score = random.randint(0, 4)
                a_score = random.randint(0, 4)
            
            # Choisir un statut
            status = random.choices(status_options, weights=weights)[0]
            
            if status == 'LIVE':
                elapsed = random.randint(30, 85)
                match_time = f"{elapsed}'"
            elif status == 'HT':
                elapsed = 45
                match_time = "HT"
            elif status == 'FT':
                elapsed = None
                match_time = "FT"
            else:  # NS
                elapsed = None
                hour = random.randint(14, 22)
                minute = random.choice(['00', '15', '30', '45'])
                match_time = f"{hour}:{minute}"
            
            matches.append({
                'home_team': home,
                'away_team': away,
                'home_score': str(h_score) if status in ['LIVE', 'HT', 'FT'] else "0",
                'away_score': str(a_score) if status in ['LIVE', 'HT', 'FT'] else "0",
                'status': status,
                'elapsed': str(elapsed) if elapsed else None,
                'league': league,
                'match_time': match_time,
                'source': 'demo'
            })
        
        return matches
    
    def scrape_worldfootball(self):
        """Tente de scraper WorldFootball.net"""
        try:
            url = "https://www.worldfootball.net/live_commentary/"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                matches = []
                
                # Recherche simplifiée
                text = soup.get_text()
                
                # Pattern pour trouver des scores
                pattern = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(\d+)[\-\:]\s*(\d+)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)'
                found_matches = re.findall(pattern, text)
                
                for match in found_matches[:10]:
                    home_team, home_score, away_score, away_team = match
                    
                    matches.append({
                        'home_team': home_team.strip(),
                        'away_team': away_team.strip(),
                        'home_score': home_score,
                        'away_score': away_score,
                        'status': 'FT',
                        'league': self.detect_league(home_team, away_team),
                        'match_time': 'FT',
                        'source': 'worldfootball'
                    })
                
                if matches:
                    return matches
            
            # Fallback sur démo
            return self.get_demo_matches("Ligue 1")
            
        except Exception as e:
            return self.get_demo_matches("Ligue 1")
    
    def get_api_matches(self):
        """Essaie l'API football-data.org"""
        try:
            # Essayer l'API gratuite
            url = "https://api.football-data.org/v4/matches"
            headers = {'X-Auth-Token': ''}  # Vide pour version gratuite
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                matches = []
                
                for match in data.get('matches', [])[:10]:
                    home_team = match.get('homeTeam', {}).get('name', 'Home')
                    away_team = match.get('awayTeam', {}).get('name', 'Away')
                    score = match.get('score', {})
                    
                    home_score = str(score.get('fullTime', {}).get('home', 0))
                    away_score = str(score.get('fullTime', {}).get('away', 0))
                    
                    status = match.get('status', 'FINISHED')
                    if status == 'LIVE':
                        status_display = 'LIVE'
                    elif status == 'FINISHED':
                        status_display = 'FT'
                    else:
                        status_display = 'NS'
                    
                    matches.append({
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'status': status_display,
                        'league': match.get('competition', {}).get('name', 'Unknown'),
                        'match_time': match.get('utcDate', '')[:10],
                        'source': 'football-data.org'
                    })
                
                return matches
            
            return self.get_demo_matches("Ligue 1")
            
        except:
            return self.get_demo_matches("Ligue 1")
    
    def detect_league(self, home_team, away_team):
        """Détecte la ligue basée sur les équipes"""
        teams = f"{home_team.lower()} {away_team.lower()}"
        
        if any(x in teams for x in ['psg', 'marseille', 'lyon', 'monaco', 'lille']):
            return "Ligue 1"
        elif any(x in teams for x in ['manchester', 'liverpool', 'arsenal', 'chelsea', 'tottenham']):
            return "Premier League"
        elif any(x in teams for x in ['real madrid', 'barcelona', 'atletico', 'sevilla', 'valencia']):
            return "La Liga"
        elif any(x in teams for x in ['bayern', 'dortmund', 'leipzig', 'frankfurt']):
            return "Bundesliga"
        elif any(x in teams for x in ['juventus', 'milan', 'inter', 'napoli', 'roma']):
            return "Serie A"
        else:
            return "Champions League"
    
    def get_standings(self, league="Ligue 1"):
        """Génère des classements réalistes"""
        # Classements par défaut basés sur les positions actuelles
        standings_data = {
            "Ligue 1": [
                {"team": "Paris SG", "points": 68, "gd": 45},
                {"team": "Marseille", "points": 60, "gd": 22},
                {"team": "Lyon", "points": 56, "gd": 18},
                {"team": "Lille", "points": 54, "gd": 15},
                {"team": "Monaco", "points": 52, "gd": 12},
                {"team": "Rennes", "points": 50, "gd": 10},
                {"team": "Nice", "points": 48, "gd": 8},
                {"team": "Lens", "points": 46, "gd": 5}
            ],
            "Premier League": [
                {"team": "Manchester City", "points": 70, "gd": 48},
                {"team": "Liverpool", "points": 68, "gd": 42},
                {"team": "Arsenal", "points": 65, "gd": 38},
                {"team": "Chelsea", "points": 58, "gd": 25},
                {"team": "Tottenham", "points": 56, "gd": 20},
                {"team": "Manchester United", "points": 54, "gd": 15},
                {"team": "Newcastle", "points": 52, "gd": 12},
                {"team": "Aston Villa", "points": 50, "gd": 8}
            ],
            "La Liga": [
                {"team": "Real Madrid", "points": 72, "gd": 40},
                {"team": "Barcelona", "points": 65, "gd": 35},
                {"team": "Atletico Madrid", "points": 62, "gd": 28},
                {"team": "Sevilla", "points": 55, "gd": 15},
                {"team": "Valencia", "points": 52, "gd": 12},
                {"team": "Real Betis", "points": 50, "gd": 10},
                {"team": "Villarreal", "points": 48, "gd": 8},
                {"team": "Athletic Bilbao", "points": 46, "gd": 5}
            ]
        }
        
        # Si la ligue n'est pas dans les données, créer un classement par défaut
        if league not in standings_data:
            # Liste d'équipes par ligue
            teams_by_league = {
                "Bundesliga": ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen", 
                               "Eintracht Frankfurt", "Wolfsburg", "Borussia Mönchengladbach", "Freiburg"],
                "Serie A": ["Juventus", "AC Milan", "Inter Milan", "Napoli", "Roma", "Lazio", "Atalanta", "Fiorentina"],
                "Champions League": ["Real Madrid", "Manchester City", "Bayern Munich", "Paris SG", 
                                     "Liverpool", "Barcelona", "Chelsea", "AC Milan"]
            }
            
            teams = teams_by_league.get(league, standings_data["Ligue 1"])
            standings_data[league] = []
            
            for i, team in enumerate(teams[:8], 1):
                points = 70 - (i * 3)
                gd = 40 - (i * 5)
                standings_data[league].append({"team": team, "points": points, "gd": gd})
        
        # Construire le classement complet
        standings = []
        for i, team_data in enumerate(standings_data.get(league, standings_data["Ligue 1"]), 1):
            matches = 28  # Nombre de matches joués
            wins = team_data["points"] // 3
            draws = team_data["points"] - (wins * 3)
            losses = matches - wins - draws
            
            goals_for = 40 + team_data["gd"]  # Buts pour
            goals_against = 40  # Buts contre
            
            standings.append({
                "position": i,
                "team": team_data["team"],
                "matches": str(matches),
                "wins": str(wins),
                "draws": str(draws),
                "losses": str(losses),
                "goals_for": str(goals_for),
                "goals_against": str(goals_against),
                "goal_diff": f"+{team_data['gd']}" if team_data['gd'] > 0 else str(team_data['gd']),
                "points": str(team_data["points"])
            })
        
        return standings

# ============================================================================
# FONCTIONS D'AFFICHAGE
# ============================================================================
def display_match(match):
    """Affiche un match de manière élégante"""
    col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 3, 2])
    
    with col1:
        if match['status'] == 'LIVE':
            st.markdown(f"<span class='live-badge'>{match['elapsed']}'</span>", unsafe_allow_html=True)
        elif match['status'] == 'HT':
            st.markdown("**HT**")
        elif match['status'] == 'FT':
            st.markdown("**FT**")
        else:
            st.markdown(f"**{match['match_time']}**")
    
    with col2:
        st.markdown(f"**{match['home_team']}**")
    
    with col3:
        score_color = "#ff4757" if match['status'] == 'LIVE' else "#2c3e50"
        st.markdown(
            f"<h3 style='text-align: center; color: {score_color}; margin: 0;'>{match['home_score']} - {match['away_score']}</h3>",
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(f"**{match['away_team']}**")
    
    with col5:
        source_icon = "🌐" if match.get('source') == 'worldfootball' else "🔧" if match.get('source') == 'football-data.org' else "🎮"
        st.caption(f"{source_icon} {match['league']}")
    
    st.divider()

def create_standings_chart(standings_df):
    """Crée un graphique du classement"""
    try:
        fig = go.Figure(data=[
            go.Bar(
                x=standings_df['team'],
                y=standings_df['points'].astype(int),
                marker_color=['#FFD700' if i == 1 else '#C0C0C0' if i == 2 else '#CD7F32' if i == 3 else '#3498db' for i in range(1, len(standings_df)+1)],
                text=standings_df['points'],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Points au classement",
            xaxis_title="Équipes",
            yaxis_title="Points",
            template="plotly_white",
            showlegend=False,
            height=400
        )
        
        return fig
    except Exception as e:
        st.warning(f"Erreur lors de la création du graphique : {str(e)}")
        return None

def create_goals_chart(matches):
    """Crée un graphique des buts"""
    try:
        # Calculer les buts par ligue
        leagues = {}
        for match in matches:
            league = match['league']
            if match['home_score'].isdigit() and match['away_score'].isdigit():
                goals = int(match['home_score']) + int(match['away_score'])
                if league not in leagues:
                    leagues[league] = []
                leagues[league].append(goals)
        
        # Préparer les données
        league_names = list(leagues.keys())
        avg_goals = [sum(goals)/len(goals) if goals else 0 for goals in leagues.values()]
        
        fig = go.Figure(data=[
            go.Bar(
                x=league_names,
                y=avg_goals,
                marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#10b981'],
                text=[f"{g:.2f}" for g in avg_goals],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Moyenne de buts par match par ligue",
            xaxis_title="Ligue",
            yaxis_title="Buts par match",
            template="plotly_white",
            height=400
        )
        
        return fig
    except:
        return None

# ============================================================================
# APPLICATION PRINCIPALE
# ============================================================================
def main():
    load_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">⚽ FOOTBALL ANALYTICS PRO</h1>
        <p style="color: white; text-align: center; margin: 10px 0 0 0; opacity: 0.9;">
            Données en temps réel • Analyse complète • Interface intuitive
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Source de données
        source = st.selectbox(
            "Source des données",
            ["demo", "worldfootball", "football_api"],
            format_func=lambda x: {
                "demo": "🎮 Mode Démo",
                "worldfootball": "🌐 WorldFootball",
                "football_api": "🔧 API Football"
            }[x],
            help="Mode Démo recommandé pour la stabilité"
        )
        
        # Sélection de ligue
        league = st.selectbox(
            "Ligue",
            ["Ligue 1", "Premier League", "La Liga", "Bundesliga", "Serie A", "Champions League"]
        )
        
        # Options
        auto_refresh = st.checkbox("🔄 Rafraîchissement automatique", value=False)
        show_details = st.checkbox("📊 Afficher les détails", value=True)
        
        # Bouton de rafraîchissement
        if st.button("🚀 Charger les données", type="primary", use_container_width=True):
            st.session_state.last_update = datetime.now()
            st.rerun()
        
        st.markdown("---")
        st.markdown(f"""
        ### 📊 Informations
        - **Dernière mise à jour:** {st.session_state.last_update.strftime('%H:%M:%S')}
        - **Mode:** {'Démo' if source == 'demo' else 'Web'}
        - **Ligue:** {league}
        """)
    
    # Initialiser le scraper
    if st.session_state.scraper is None:
        st.session_state.scraper = FootballScraper()
    
    scraper = st.session_state.scraper
    
    # Onglets principaux
    tab1, tab2, tab3 = st.tabs(["📊 Matchs en Direct", "🏆 Classements", "📈 Statistiques"])
    
    with tab1:
        st.header(f"⚽ Matchs - {league}")
        
        # Charger les données
        with st.spinner("Chargement des matchs..."):
            matches = scraper.get_matches(source, league)
            
            if not matches:
                st.warning("Aucun match trouvé. Utilisation des données de démo.")
                matches = scraper.get_demo_matches(league)
        
        # Calculer les métriques
        live_matches = [m for m in matches if m['status'] == 'LIVE']
        total_goals = 0
        for match in matches:
            if match['home_score'].isdigit() and match['away_score'].isdigit():
                total_goals += int(match['home_score']) + int(match['away_score'])
        
        # Afficher les métriques
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Matchs en direct", len(live_matches))
        with col2:
            st.metric("Total matchs", len(matches))
        with col3:
            st.metric("Buts totaux", total_goals)
        with col4:
            avg_goals = total_goals / len(matches) if matches else 0
            st.metric("Moyenne buts", f"{avg_goals:.1f}")
        
        # Afficher les matchs
        st.markdown("### 📅 Résultats des Matchs")
        
        for match in matches:
            display_match(match)
        
        # Graphique des buts si demandé
        if show_details and matches:
            goals_chart = create_goals_chart(matches)
            if goals_chart:
                st.plotly_chart(goals_chart, use_container_width=True)
    
    with tab2:
        st.header(f"🏆 Classement - {league}")
        
        with st.spinner("Chargement du classement..."):
            standings = scraper.get_standings(league)
        
        if standings:
            # Convertir en DataFrame
            df_standings = pd.DataFrame(standings)
            
            # Afficher le tableau
            st.dataframe(
                df_standings,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "position": st.column_config.NumberColumn("Pos", width="small"),
                    "team": st.column_config.TextColumn("Équipe"),
                    "matches": st.column_config.NumberColumn("MJ", width="small"),
                    "wins": st.column_config.NumberColumn("G", width="small"),
                    "draws": st.column_config.NumberColumn("N", width="small"),
                    "losses": st.column_config.NumberColumn("P", width="small"),
                    "goals_for": st.column_config.NumberColumn("BP", width="small"),
                    "goals_against": st.column_config.NumberColumn("BC", width="small"),
                    "goal_diff": st.column_config.TextColumn("+/-", width="small"),
                    "points": st.column_config.ProgressColumn(
                        "PTS",
                        format="%d",
                        min_value=0,
                        max_value=100,
                        width="medium"
                    )
                }
            )
            
            # Graphique du classement
            chart = create_standings_chart(df_standings)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("Aucun classement disponible pour cette ligue.")
    
    with tab3:
        st.header("📈 Statistiques Avancées")
        
        if 'matches' in locals() and matches:
            # Calculer diverses statistiques
            home_wins = len([m for m in matches if m['home_score'] > m['away_score']])
            draws = len([m for m in matches if m['home_score'] == m['away_score']])
            away_wins = len([m for m in matches if m['home_score'] < m['away_score']])
            
            high_scoring = len([m for m in matches if 
                               m['home_score'].isdigit() and m['away_score'].isdigit() and
                               int(m['home_score']) + int(m['away_score']) > 2.5])
            
            clean_sheets = len([m for m in matches if m['away_score'] == '0'])
            
            # Afficher les statistiques
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Distribution des résultats")
                results_data = {
                    "Résultat": ["Victoire domicile", "Match nul", "Victoire extérieur"],
                    "Pourcentage": [
                        f"{(home_wins/len(matches))*100:.1f}%" if matches else "0%",
                        f"{(draws/len(matches))*100:.1f}%" if matches else "0%",
                        f"{(away_wins/len(matches))*100:.1f}%" if matches else "0%"
                    ]
                }
                st.dataframe(pd.DataFrame(results_data), use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Autres statistiques")
                stats_data = {
                    "Statistique": ["Matchs > 2.5 buts", "Clean sheets", "Buts/match"],
                    "Valeur": [
                        f"{(high_scoring/len(matches))*100:.1f}%" if matches else "0%",
                        f"{(clean_sheets/len(matches))*100:.1f}%" if matches else "0%",
                        f"{total_goals/len(matches):.2f}" if matches else "0"
                    ]
                }
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
            
            # Graphique circulaire des résultats
            if home_wins + draws + away_wins > 0:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Victoire domicile', 'Match nul', 'Victoire extérieur'],
                    values=[home_wins, draws, away_wins],
                    hole=.3,
                    marker_colors=['#667eea', '#764ba2', '#f093fb']
                )])
                
                fig_pie.update_layout(
                    title="Distribution des résultats",
                    height=400
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Chargez d'abord des matchs dans l'onglet 'Matchs en Direct'")
    
    # Rafraîchissement automatique
    if auto_refresh:
        time.sleep(30)  # Rafraîchir toutes les 30 secondes
        st.session_state.last_update = datetime.now()
        st.rerun()

# Point d'entrée
if __name__ == "__main__":
    main()
