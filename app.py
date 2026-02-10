import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import requests
import json
from pytz import timezone
import warnings
import hashlib
from bs4 import BeautifulSoup
import re
import random

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="⚽ Football Betting Analytics Live",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation de session
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'timezone' not in st.session_state:
    st.session_state.timezone = 'Europe/Paris'
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'button_counter' not in st.session_state:
    st.session_state.button_counter = 0
if 'scraping_active' not in st.session_state:
    st.session_state.scraping_active = False

# CSS personnalisé
def load_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
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
    
    .match-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
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
    
    .scraping-badge {
        background: #10b981;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .dataframe {
        font-size: 0.9rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f3f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1rem;
        padding: 10px 20px;
    }
    
    .scraping-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Fonction pour générer des clés uniques
def generate_unique_key(base_string):
    """Génère une clé unique basée sur une chaîne et un compteur"""
    st.session_state.button_counter += 1
    return f"{base_string}_{st.session_state.button_counter}_{hashlib.md5(base_string.encode()).hexdigest()[:8]}"

# ============================================================================
# CLASSE DE SCRAPING ALTERNATIVE
# ============================================================================
class FootballScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
    
    def scrape_matches(self, source="worldfootball"):
        """
        Scrape les matchs depuis différentes sources
        """
        if source == "worldfootball":
            return self.scrape_worldfootball()
        elif source == "sofascore":
            return self.scrape_sofascore_demo()
        else:
            return self.get_demo_matches()
    
    def scrape_worldfootball(self):
        """
        Scrape les matchs depuis WorldFootball.net (plus accessible)
        """
        try:
            # Utiliser plusieurs URLs potentielles
            urls = [
                "https://www.worldfootball.net/live_commentary/",
                "https://www.worldfootball.net/schedule/eng-premier-league-2024-2025-spieltag/1/",
                "https://www.worldfootball.net/schedule/fra-ligue-1-2024-2025-spieltag/1/"
            ]
            
            all_matches = []
            
            for url in urls[:1]:  # Essayer seulement la première pour l'instant
                try:
                    response = requests.get(url, headers=self.headers, timeout=15)
                    
                    if response.status_code != 200:
                        continue
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Chercher les matchs dans différentes structures HTML
                    match_elements = []
                    
                    # Essayer plusieurs sélecteurs
                    selectors = [
                        'table.standard_tabelle tr',
                        'div.live div.match',
                        'table.matches tbody tr'
                    ]
                    
                    for selector in selectors:
                        elements = soup.select(selector)
                        if elements and len(elements) > 1:
                            match_elements = elements
                            break
                    
                    if not match_elements:
                        # Fallback: chercher des tables avec des scores
                        tables = soup.find_all('table')
                        for table in tables:
                            if any(x in table.text.lower() for x in ['score', 'result', 'match']):
                                match_elements = table.find_all('tr')[1:]
                                break
                    
                    for element in match_elements[:15]:  # Limiter à 15 matchs
                        try:
                            match_data = self.parse_match_element(element)
                            if match_data:
                                all_matches.append(match_data)
                        except:
                            continue
                            
                except requests.exceptions.Timeout:
                    st.warning(f"Timeout pour {url}")
                    continue
                except Exception as e:
                    st.warning(f"Erreur pour {url}: {str(e)}")
                    continue
            
            # Si on a des matchs, les retourner
            if all_matches:
                return all_matches[:10]  # Retourner max 10 matchs
            
            # Sinon, retourner les données de démo
            return self.get_demo_matches()
            
        except Exception as e:
            st.error(f"Erreur scraping générale: {str(e)}")
            return self.get_demo_matches()
    
    def parse_match_element(self, element):
        """Parse un élément HTML pour en extraire les données du match"""
        try:
            # Essayer plusieurs méthodes d'extraction
            text = element.get_text(separator='|', strip=True)
            parts = text.split('|')
            
            if len(parts) >= 4:
                # Méthode 1: Format standard
                time_info = parts[0] if parts[0] else "20:00"
                teams_part = parts[1] if len(parts) > 1 else "Team A - Team B"
                score_part = parts[2] if len(parts) > 2 else "0-0"
                
                # Nettoyer et extraire
                teams = teams_part.split('-')
                home_team = teams[0].strip() if len(teams) > 0 else "Home Team"
                away_team = teams[1].strip() if len(teams) > 1 else "Away Team"
                
                # Extraire le score
                score_match = re.search(r'(\d+)\s*-\s*(\d+)', score_part)
                if score_match:
                    home_score = score_match.group(1)
                    away_score = score_match.group(2)
                else:
                    home_score = "0"
                    away_score = "0"
                
                # Déterminer l'état
                if ':' in time_info and len(time_info) <= 5:
                    if time_info.endswith("'"):
                        status = 'LIVE'
                        elapsed = time_info.replace("'", "")
                    else:
                        status = 'NS'
                        elapsed = None
                elif time_info.upper() in ['HT', 'FT']:
                    status = time_info.upper()
                    elapsed = None
                else:
                    status = 'LIVE' if 'live' in text.lower() else 'NS'
                    elapsed = str(random.randint(30, 90)) if status == 'LIVE' else None
                
                # Déterminer la ligue
                league = "Ligue 1" if any(x in home_team.lower() for x in ['psg', 'marseille', 'lyon']) else \
                        "Premier League" if any(x in home_team.lower() for x in ['manchester', 'liverpool', 'arsenal']) else \
                        "La Liga" if any(x in home_team.lower() for x in ['real', 'barcelona', 'atletico']) else \
                        "Champions League"
                
                return {
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                    'status': status,
                    'elapsed': elapsed,
                    'league': league,
                    'match_time': time_info,
                    'source': 'worldfootball'
                }
            
        except Exception as e:
            pass
        
        return None
    
    def scrape_sofascore_demo(self):
        """Version démo pour SofaScore"""
        return self.get_demo_matches()
    
    def get_demo_matches(self):
        """Retourne des données de démo"""
        leagues = ['Ligue 1', 'Premier League', 'La Liga', 'Serie A', 'Bundesliga']
        teams_fr = ['Paris SG', 'Marseille', 'Lyon', 'Monaco', 'Lille', 'Nice', 'Rennes', 'Lens']
        teams_en = ['Manchester City', 'Liverpool', 'Arsenal', 'Chelsea', 'Tottenham', 'Man United', 'Newcastle', 'Aston Villa']
        teams_es = ['Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla', 'Valencia', 'Real Betis', 'Villarreal']
        teams_it = ['Juventus', 'AC Milan', 'Inter Milan', 'Napoli', 'Roma', 'Lazio', 'Atalanta']
        teams_de = ['Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen', 'Eintracht']
        
        matches = []
        status_options = ['NS', 'LIVE', 'HT', 'FT']
        
        for i in range(8):
            league = random.choice(leagues)
            
            if league == 'Ligue 1':
                teams = teams_fr
            elif league == 'Premier League':
                teams = teams_en
            elif league == 'La Liga':
                teams = teams_es
            elif league == 'Serie A':
                teams = teams_it
            else:
                teams = teams_de
            
            home = random.choice(teams)
            away = random.choice([t for t in teams if t != home])
            status = random.choice(status_options)
            
            if status == 'LIVE':
                home_score = random.randint(0, 3)
                away_score = random.randint(0, 3)
                elapsed = random.randint(1, 90)
                match_time = f"{elapsed}'"
            elif status == 'FT':
                home_score = random.randint(0, 4)
                away_score = random.randint(0, 4)
                elapsed = None
                match_time = "FT"
            elif status == 'HT':
                home_score = random.randint(0, 2)
                away_score = random.randint(0, 2)
                elapsed = 45
                match_time = "HT"
            else:
                home_score = 0
                away_score = 0
                elapsed = None
                match_time = f"{random.randint(14, 22)}:{random.choice(['00', '15', '30', '45'])}"
            
            matches.append({
                'home_team': home,
                'away_team': away,
                'home_score': str(home_score),
                'away_score': str(away_score),
                'status': status,
                'elapsed': str(elapsed) if elapsed else None,
                'league': league,
                'match_time': match_time,
                'source': 'demo'
            })
        
        return matches
    
    def scrape_standings(self, league="Ligue 1"):
        """Scrape les classements"""
        try:
            # Mapping des URLs pour les classements
            league_urls = {
                "Ligue 1": "https://www.worldfootball.net/table/fra-ligue-1-2024-2025/",
                "Premier League": "https://www.worldfootball.net/table/eng-premier-league-2024-2025/",
                "La Liga": "https://www.worldfootball.net/table/esp-primera-division-2024-2025/",
                "Bundesliga": "https://www.worldfootball.net/table/bundesliga-2024-2025/",
                "Serie A": "https://www.worldfootball.net/table/ita-serie-a-2024-2025/"
            }
            
            if league not in league_urls:
                return self.get_demo_standings(league)
            
            url = league_urls[league]
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                return self.get_demo_standings(league)
            
            soup = BeautifulSoup(response.content, 'html.parser')
            standings = []
            
            # Chercher la table des classements
            table = soup.find('table', class_='standard_tabelle')
            
            if table:
                rows = table.find_all('tr')[1:9]  # 8 premières équipes
                for i, row in enumerate(rows, 1):
                    cols = row.find_all('td')
                    if len(cols) >= 10:
                        standings.append({
                            'position': i,
                            'team': cols[2].text.strip() if len(cols) > 2 else f"Team {i}",
                            'matches': cols[3].text.strip() if len(cols) > 3 else str(random.randint(20, 30)),
                            'wins': cols[4].text.strip() if len(cols) > 4 else str(random.randint(10, 20)),
                            'draws': cols[5].text.strip() if len(cols) > 5 else str(random.randint(3, 8)),
                            'losses': cols[6].text.strip() if len(cols) > 6 else str(random.randint(2, 7)),
                            'goals_for': cols[7].text.strip() if len(cols) > 7 else str(random.randint(30, 50)),
                            'goals_against': cols[8].text.strip() if len(cols) > 8 else str(random.randint(15, 35)),
                            'goal_diff': cols[9].text.strip() if len(cols) > 9 else str(random.randint(5, 25)),
                            'points': cols[10].text.strip() if len(cols) > 10 else str(random.randint(30, 55))
                        })
            
            if standings:
                return standings
            else:
                return self.get_demo_standings(league)
                
        except Exception as e:
            st.warning(f"Erreur scraping classement: {str(e)}")
            return self.get_demo_standings(league)
    
    def get_demo_standings(self, league="Ligue 1"):
        """Retourne un classement de démo"""
        teams = {
            "Ligue 1": ["Paris SG", "Marseille", "Lyon", "Monaco", "Lille", "Nice", "Rennes", "Lens"],
            "Premier League": ["Manchester City", "Liverpool", "Arsenal", "Chelsea", "Tottenham", "Man United", "Newcastle", "Aston Villa"],
            "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Valencia", "Real Betis", "Villarreal", "Athletic Bilbao"],
            "Bundesliga": ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen", "Eintracht Frankfurt", "Wolfsburg", "M'gladbach", "Freiburg"],
            "Serie A": ["Juventus", "AC Milan", "Inter Milan", "Napoli", "Roma", "Lazio", "Atalanta", "Fiorentina"]
        }
        
        standings = []
        league_teams = teams.get(league, teams["Ligue 1"])
        
        for i, team in enumerate(league_teams, 1):
            matches = random.randint(25, 30)
            wins = random.randint(15, 20)
            draws = random.randint(5, 10)
            losses = matches - wins - draws
            goals_for = random.randint(40, 60)
            goals_against = random.randint(20, 40)
            goal_diff = goals_for - goals_against
            points = wins * 3 + draws
            
            standings.append({
                'position': i,
                'team': team,
                'matches': str(matches),
                'wins': str(wins),
                'draws': str(draws),
                'losses': str(losses),
                'goals_for': str(goals_for),
                'goals_against': str(goals_against),
                'goal_diff': f"+{goal_diff}" if goal_diff > 0 else str(goal_diff),
                'points': str(points)
            })
        
        return standings

# ============================================================================
# API MANAGER (EXISTANT)
# ============================================================================
class FootballAPIManager:
    def __init__(self, api_key=None):
        self.api_key = api_key or st.session_state.api_key
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            'x-apisports-key': self.api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }
    
    def make_request(self, endpoint, params=None):
        """Fait une requête à l'API Football"""
        try:
            if not self.api_key:
                return None
                
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.warning(f"Erreur API: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Erreur de connexion: {str(e)}")
            return None
    
    def get_live_matches(self):
        """Récupère les matchs en direct"""
        today = datetime.now().strftime('%Y-%m-%d')
        params = {
            'date': today,
            'status': 'LIVE'
        }
        return self.make_request('fixtures', params)
    
    def get_today_matches(self):
        """Récupère les matchs du jour"""
        today = datetime.now().strftime('%Y-%m-%d')
        params = {'date': today}
        return self.make_request('fixtures', params)

# ============================================================================
# DONNÉES DE DÉMO
# ============================================================================
DEMO_DATA = {
    'live_matches': [
        {
            'fixture': {'id': 1, 'status': {'short': 'LIVE', 'elapsed': 65}},
            'teams': {'home': {'name': 'Paris SG', 'logo': 'https://media.api-sports.io/football/teams/85.png'},
                     'away': {'name': 'Marseille', 'logo': 'https://media.api-sports.io/football/teams/81.png'}},
            'goals': {'home': 2, 'away': 1},
            'league': {'name': 'Ligue 1', 'country': 'France'}
        },
        {
            'fixture': {'id': 2, 'status': {'short': 'LIVE', 'elapsed': 30}},
            'teams': {'home': {'name': 'Liverpool', 'logo': 'https://media.api-sports.io/football/teams/40.png'},
                     'away': {'name': 'Manchester City', 'logo': 'https://media.api-sports.io/football/teams/50.png'}},
            'goals': {'home': 0, 'away': 0},
            'league': {'name': 'Premier League', 'country': 'England'}
        }
    ],
    'today_matches': [
        {
            'fixture': {'id': 3, 'date': (datetime.now() + timedelta(hours=2)).isoformat()},
            'teams': {'home': {'name': 'Real Madrid'}, 'away': {'name': 'Barcelona'}},
            'league': {'name': 'La Liga'}
        }
    ]
}

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================
def format_time(timestamp):
    """Formate un timestamp en heure locale"""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        local_tz = timezone(st.session_state.timezone)
        local_dt = dt.astimezone(local_tz)
        return local_dt.strftime('%H:%M')
    except:
        return "N/A"

def calculate_probability(home_stats, away_stats):
    """Calcule les probabilités basiques"""
    if not home_stats or not away_stats:
        return {'home': 45, 'draw': 30, 'away': 25}
    
    home_strength = home_stats.get('strength', 50)
    away_strength = away_stats.get('strength', 50)
    
    total = home_strength + away_strength
    home_prob = int((home_strength / total) * 45) + 40
    away_prob = int((away_strength / total) * 45) + 40
    draw_prob = 100 - home_prob - away_prob
    
    return {
        'home': min(95, max(5, home_prob)),
        'draw': min(95, max(5, draw_prob)),
        'away': min(95, max(5, away_prob))
    }

def generate_prediction_analysis(fixture):
    """Génère une analyse de prédiction"""
    analysis = {
        'confidence': np.random.randint(65, 90),
        'recommendation': np.random.choice(['1X', 'X2', 'Over 1.5', 'Under 3.5', 'BTTS Yes', 'BTTS No']),
        'risk_level': np.random.choice(['Low', 'Medium', 'High']),
        'key_factors': []
    }
    
    factors = [
        "Forme récente favorable",
        "Blessures importantes",
        "Motivation élevée",
        "Historique favorable",
        "Conditions météo optimales",
        "Suspensions clés"
    ]
    
    analysis['key_factors'] = np.random.choice(factors, size=3, replace=False).tolist()
    return analysis

# ============================================================================
# FONCTIONS D'AFFICHAGE
# ============================================================================
def display_scraped_match(match, index):
    """Affiche un match scrapé avec une clé unique"""
    try:
        col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 2])
        
        with col1:
            if match['status'] == 'LIVE':
                st.markdown(f"<span class='live-badge'>{match['elapsed'] or '45'}'</span>", 
                           unsafe_allow_html=True)
            elif match['status'] == 'HT':
                st.markdown(f"<span style='color: orange; font-weight: bold;'>HT</span>", 
                           unsafe_allow_html=True)
            elif match['status'] == 'FT':
                st.markdown(f"<span style='color: green; font-weight: bold;'>FT</span>", 
                           unsafe_allow_html=True)
            else:
                st.markdown(f"**{match['match_time']}**")
        
        with col2:
            st.markdown(f"**{match['home_team']}**")
        
        with col3:
            score_color = "#ff4757" if match['status'] == 'LIVE' else "inherit"
            st.markdown(
                f"<h3 style='text-align: center; color: {score_color}; margin: 0;'>{match['home_score']} - {match['away_score']}</h3>",
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(f"**{match['away_team']}**")
        
        with col5:
            source_badge = "🌐" if match['source'] == 'worldfootball' else "📱"
            st.caption(f"{source_badge} {match['league']}")
        
        st.divider()
        
    except Exception as e:
        st.warning(f"Erreur d'affichage: {str(e)}")

def display_live_match(match, index):
    """Affiche un match en direct avec une clé unique"""
    try:
        fixture = match.get('fixture', {})
        teams = match.get('teams', {})
        goals = match.get('goals', {})
        league = match.get('league', {})
        
        home_team = teams.get('home', {}).get('name', 'Home')
        away_team = teams.get('away', {}).get('name', 'Away')
        home_logo = teams.get('home', {}).get('logo', '')
        away_logo = teams.get('away', {}).get('logo', '')
        home_score = goals.get('home', 0)
        away_score = goals.get('away', 0)
        status = fixture.get('status', {}).get('short', 'NS')
        elapsed = fixture.get('status', {}).get('elapsed', 0)
        league_name = league.get('name', '')
        
        col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])
        
        with col1:
            if home_logo:
                st.image(home_logo, width=40)
        
        with col2:
            st.markdown(f"**{home_team}**")
        
        with col3:
            st.markdown(f"<h3 style='text-align: center; margin: 0;'>{home_score} - {away_score}</h3>", 
                       unsafe_allow_html=True)
            if status == 'LIVE':
                st.markdown(f'<span class="live-badge">{elapsed}\'</span>', unsafe_allow_html=True)
            else:
                st.caption(status)
        
        with col4:
            st.markdown(f"**{away_team}**")
        
        with col5:
            if away_logo:
                st.image(away_logo, width=40)
        
        st.caption(f"{league_name}")
        st.divider()
        
    except Exception as e:
        st.warning(f"Erreur d'affichage du match: {str(e)}")

def show_detailed_analysis(match, analysis, odds, index):
    """Affiche une analyse détaillée avec clé unique"""
    st.markdown("### 📊 Analyse détaillée")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique radar
        categories = ['Attaque', 'Défense', 'Forme', 'Motivation', 'Historique', 'Conditions']
        home_values = [np.random.randint(60, 95) for _ in categories]
        away_values = [np.random.randint(60, 95) for _ in categories]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=home_values,
            theta=categories,
            fill='toself',
            name='Domicile',
            line_color='#667eea'
        ))
        fig.add_trace(go.Scatterpolar(
            r=away_values,
            theta=categories,
            fill='toself',
            name='Extérieur',
            line_color='#f093fb'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Analyse comparative"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Recommandations
        st.markdown("### 💎 Recommandation")
        st.markdown(f"""
        <div class="prediction-card">
            <h3 style="margin:0;">{analysis['recommendation']}</h3>
            <p style="margin:10px 0 0 0;">
                <strong>Confiance:</strong> {analysis['confidence']}%<br>
                <strong>Cote conseillée:</strong> {odds}<br>
                <strong>Mise suggérée:</strong> 2-4% de bankroll
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Statistiques
        st.markdown("### 📈 Statistiques clés")
        stats = {
            "Buts/moyenne domicile": f"{np.random.randint(1, 3)}.{np.random.randint(0, 9)}",
            "Buts/moyenne extérieur": f"{np.random.randint(1, 3)}.{np.random.randint(0, 9)}",
            "Clean sheets dernière 5": f"{np.random.randint(1, 4)}/5",
            "BTTS dernière 5": f"{np.random.randint(2, 5)}/5",
            "+2.5 buts dernière 5": f"{np.random.randint(2, 5)}/5"
        }
        
        for key, value in stats.items():
            st.markdown(f"**{key}:** {value}")

# ============================================================================
# PAGES DE RENDU
# ============================================================================
def render_dashboard(api_manager, show_live):
    """Affiche le dashboard en direct"""
    st.markdown("### 📊 Tableau de bord Live")
    
    # KPI en temps réel
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Matchs en direct", "12", "+3")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Prédictions actives", "24", "78%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Cote moyenne", "2.15", "-0.10")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Taux de succès", "72.5%", "+1.8%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Section matchs en direct
    if show_live:
        st.markdown("### 🎯 Matchs en Direct")
        
        # Récupérer les matchs en direct
        live_data = api_manager.get_live_matches() if api_manager.api_key else DEMO_DATA
        
        if live_data and 'response' in live_data and live_data['response']:
            matches = live_data['response']
            for i, match in enumerate(matches[:5]):
                display_live_match(match, i)
        else:
            # Mode démo
            for i, match in enumerate(DEMO_DATA['live_matches']):
                display_live_match(match, i)
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des résultats
        results = ['Victoire Domicile', 'Match Nul', 'Victoire Extérieur']
        percentages = [45.3, 27.8, 26.9]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=results,
                values=percentages,
                hole=.4,
                marker_colors=['#667eea', '#764ba2', '#f093fb']
            )
        ])
        fig.update_layout(
            title="Distribution des résultats récents",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tendances des buts
        days = list(range(1, 31))
        goals = [2.1 + 0.1 * np.sin(i/3) + np.random.normal(0, 0.1) for i in days]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=days, y=goals,
            mode='lines+markers',
            name='Buts/match',
            line=dict(color='#f5576c', width=3),
            marker=dict(size=6)
        ))
        fig.update_layout(
            title="Évolution moyenne des buts (30 jours)",
            xaxis_title="Jours",
            yaxis_title="Buts par match",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

def render_predictions(api_manager):
    """Affiche les prédictions"""
    st.markdown("### 🔮 Prédictions Intelligentes")
    
    # Récupérer les matchs du jour
    today_data = api_manager.get_today_matches() if api_manager.api_key else None
    
    if today_data and 'response' in today_data:
        matches = today_data['response']
    else:
        matches = DEMO_DATA['today_matches']
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    with col1:
        min_confidence = st.slider("Confiance minimale", 50, 95, 70, key="min_confidence_slider")
    with col2:
        max_odds = st.slider("Cote maximale", 1.1, 5.0, 3.0, 0.1, key="max_odds_slider")
    with col3:
        bet_type = st.selectbox("Type de pari", ["Tous", "1X2", "Over/Under", "BTTS", "Double Chance"], key="bet_type_select")
    
    # Prédictions
    for i, match in enumerate(matches[:10]):
        try:
            fixture = match.get('fixture', {})
            teams = match.get('teams', {})
            
            home_team = teams.get('home', {}).get('name', f'Team H{i}')
            away_team = teams.get('away', {}).get('name', f'Team A{i}')
            match_time = format_time(fixture.get('date', ''))
            
            # Générer une prédiction
            analysis = generate_prediction_analysis(match)
            
            if analysis['confidence'] >= min_confidence:
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 2, 1, 2])
                    
                    with col1:
                        st.markdown(f"**{home_team} vs {away_team}**")
                        st.caption(f"⏰ {match_time}")
                    
                    with col2:
                        risk_color = {
                            'Low': 'green',
                            'Medium': 'orange',
                            'High': 'red'
                        }.get(analysis['risk_level'], 'gray')
                        
                        st.markdown(f"""
                        🎯 **{analysis['recommendation']}**
                        ⚡ **Niveau de risque:** <span style='color:{risk_color}'>{analysis['risk_level']}</span>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        confidence_color = 'green' if analysis['confidence'] > 75 else 'orange' if analysis['confidence'] > 65 else 'red'
                        st.markdown(f"""
                        <div style='text-align: center;'>
                            <h3 style='color:{confidence_color}; margin:0;'>{analysis['confidence']}%</h3>
                            <small>Confiance</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        odds = round(1.5 + np.random.random() * 1.5, 2)
                        if odds <= max_odds:
                            st.metric("Meilleure cote", f"{odds}", 
                                     delta=f"+{round((odds-1)*100, 1)}%")
                            
                            analyze_key = generate_unique_key(f"analyze_{home_team}_{away_team}")
                            if st.button("📊 Analyser", key=analyze_key):
                                show_detailed_analysis(match, analysis, odds, i)
                    
                    expander_key = generate_unique_key(f"expander_{home_team}_{away_team}")
                    with st.expander("📋 Facteurs déterminants", expanded=False, key=expander_key):
                        for factor in analysis['key_factors']:
                            st.markdown(f"✅ {factor}")
                    
                    st.divider()
                    
        except Exception as e:
            st.error(f"Erreur dans la prédiction: {str(e)}")

def render_statistics(api_manager):
    """Affiche les statistiques"""
    st.markdown("### 📈 Statistiques Avancées")
    
    # Sélection de l'équipe/ligue
    col1, col2 = st.columns(2)
    with col1:
        league_options = ["Ligue 1", "Premier League", "La Liga", "Bundesliga", "Serie A"]
        selected_league = st.selectbox("Sélectionner une ligue", league_options, key="league_select")
    
    with col2:
        team_options = {
            "Ligue 1": ["Paris SG", "Marseille", "Lyon", "Monaco", "Lille"],
            "Premier League": ["Manchester City", "Liverpool", "Arsenal", "Chelsea", "Tottenham"],
            "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Valencia"],
            "Bundesliga": ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen", "Eintracht Frankfurt"],
            "Serie A": ["Juventus", "AC Milan", "Inter Milan", "Napoli", "Roma"]
        }
        selected_team = st.selectbox("Sélectionner une équipe", team_options[selected_league], key="team_select")
    
    # Graphiques
    tab1, tab2, tab3 = st.tabs(["Performance", "Tendances", "Comparaisons"])
    
    with tab1:
        months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin']
        home_perf = [np.random.randint(60, 90) for _ in months]
        away_perf = [np.random.randint(50, 85) for _ in months]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=months, y=home_perf,
            name='Performance domicile',
            marker_color='#667eea'
        ))
        fig.add_trace(go.Bar(
            x=months, y=away_perf,
            name='Performance extérieur',
            marker_color='#f093fb'
        ))
        
        fig.update_layout(
            title=f"Performance {selected_team} - Saison en cours",
            barmode='group',
            xaxis_title="Mois",
            yaxis_title="Performance (%)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        weeks = list(range(1, 21))
        form_trend = [50 + np.random.normal(0, 10) + i*2 for i in weeks]
        goals_trend = [1.5 + np.random.normal(0, 0.3) + i*0.05 for i in weeks]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weeks, y=form_trend,
            mode='lines',
            name='Indice de forme',
            line=dict(color='#667eea', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=weeks, y=goals_trend,
            mode='lines',
            name='Buts/match',
            line=dict(color='#f093fb', width=3),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Évolution des performances",
            xaxis_title="Semaines",
            yaxis_title="Indice de forme",
            yaxis2=dict(
                title="Buts/match",
                overlaying='y',
                side='right'
            ),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        teams_comparison = {
            'Équipe': [selected_team] + np.random.choice(team_options[selected_league], 4, replace=False).tolist(),
            'Points': np.random.randint(20, 60, 5),
            'Buts pour': np.random.randint(20, 50, 5),
            'Buts contre': np.random.randint(10, 40, 5),
            'Différence': np.random.randint(-10, 20, 5)
        }
        
        df_comparison = pd.DataFrame(teams_comparison)
        df_comparison = df_comparison.sort_values('Points', ascending=False)
        
        st.dataframe(
            df_comparison,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Points': st.column_config.ProgressColumn(
                    format="%d",
                    min_value=0,
                    max_value=60
                ),
                'Buts pour': st.column_config.NumberColumn(format="%d"),
                'Buts contre': st.column_config.NumberColumn(format="%d"),
                'Différence': st.column_config.NumberColumn(format="%+d")
            }
        )

def render_matches(api_manager, show_live):
    """Affiche les matchs avec des clés uniques"""
    st.markdown("### ⚽ Calendrier des Matchs")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        date_range = st.selectbox(
            "Période",
            ["Aujourd'hui", "Demain", "Week-end", "7 prochains jours", "Tous"],
            key="date_range_select"
        )
    
    with col2:
        league_filter = st.multiselect(
            "Ligues",
            ["Ligue 1", "Premier League", "La Liga", "Bundesliga", "Serie A", "Champions League"],
            default=["Ligue 1", "Premier League"],
            key="league_multiselect"
        )
    
    with col3:
        only_live = st.checkbox("En direct seulement", value=False, key="only_live_checkbox")
    
    matches = []
    match_counter = 0
    for league in league_filter:
        for i in range(5):
            match_counter += 1
            matches.append({
                'id': match_counter,
                'date': (datetime.now() + timedelta(days=np.random.randint(0, 7))).strftime('%Y-%m-%d'),
                'time': f"{np.random.randint(12, 22)}:{np.random.choice(['00', '15', '30', '45'])}",
                'league': league,
                'home': f"Équipe H{i}",
                'away': f"Équipe A{i}",
                'status': np.random.choice(['NS', 'LIVE', 'HT', 'FT'], p=[0.6, 0.1, 0.1, 0.2])
            })
    
    for match in matches:
        if only_live and match['status'] != 'LIVE':
            continue
            
        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 1, 2, 2, 1])
            
            with col1:
                status_color = {
                    'NS': 'gray',
                    'LIVE': 'red',
                    'HT': 'orange',
                    'FT': 'green'
                }.get(match['status'], 'gray')
                st.markdown(f"<span style='color:{status_color}; font-weight:bold'>{match['status']}</span>", 
                           unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**{match['home']}**")
            
            with col3:
                st.markdown("**vs**")
            
            with col4:
                st.markdown(f"**{match['away']}**")
            
            with col5:
                st.markdown(f"{match['league']} • {match['time']}")
            
            with col6:
                button_key = generate_unique_key(f"stats_{match['id']}_{match['home']}_{match['away']}")
                if st.button("📊", key=button_key):
                    st.info(f"Analyse détaillée pour {match['home']} vs {match['away']}")
            
            st.divider()

def render_standings(api_manager):
    """Affiche les classements"""
    st.markdown("### 🏆 Classements des Ligues")
    
    selected_league = st.selectbox(
        "Choisir une ligue",
        ["Ligue 1", "Premier League", "La Liga", "Bundesliga", "Serie A", "Champions League"],
        key="standings_league_select"
    )
    
    teams = {
        "Ligue 1": ["Paris SG", "Marseille", "Lyon", "Monaco", "Lille", "Nice", "Rennes", "Lens"],
        "Premier League": ["Manchester City", "Liverpool", "Arsenal", "Chelsea", "Tottenham", "Manchester United", "Newcastle", "Aston Villa"],
        "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Valencia", "Real Betis", "Villarreal", "Athletic Bilbao"],
        "Bundesliga": ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen", "Eintracht Frankfurt", "Wolfsburg", "Borussia M'gladbach", "Freiburg"],
        "Serie A": ["Juventus", "AC Milan", "Inter Milan", "Napoli", "Roma", "Lazio", "Atalanta", "Fiorentina"],
        "Champions League": ["Real Madrid", "Manchester City", "Bayern Munich", "Paris SG", "Liverpool", "Barcelona", "Chelsea", "AC Milan"]
    }
    
    standings_data = []
    for i, team in enumerate(teams[selected_league], 1):
        standings_data.append({
            'Position': i,
            'Équipe': team,
            'MJ': np.random.randint(20, 30),
            'G': np.random.randint(10, 20),
            'N': np.random.randint(3, 10),
            'P': np.random.randint(2, 8),
            'BP': np.random.randint(30, 60),
            'BC': np.random.randint(15, 40),
            'Diff': np.random.randint(-5, 30),
            'PTS': np.random.randint(30, 60)
        })
    
    df_standings = pd.DataFrame(standings_data)
    
    st.dataframe(
        df_standings,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Position': st.column_config.NumberColumn(format="%d"),
            'PTS': st.column_config.ProgressColumn(
                format="%d",
                min_value=0,
                max_value=60
            ),
            'Diff': st.column_config.NumberColumn(format="%+d")
        }
    )
    
    fig = go.Figure(data=[
        go.Bar(
            x=df_standings['Équipe'],
            y=df_standings['PTS'],
            marker_color=['#667eea' if i < 4 else '#f093fb' if i < 7 else '#764ba2' for i in range(len(df_standings))]
        )
    ])
    
    fig.update_layout(
        title=f"Points - {selected_league}",
        xaxis_title="Équipes",
        yaxis_title="Points",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE DE SCRAPING
# ============================================================================
def render_scraping_page():
    """Page dédiée au scraping de données football"""
    st.markdown("### 🌐 Données Football en Direct (Scrapé)")
    
    # Avertissement
    st.markdown("""
    <div class="scraping-warning">
        ⚠️ <strong>Attention:</strong> Cette page utilise le web scraping pour récupérer des données en temps réel.
        Les données peuvent être limitées et dépendent de la disponibilité des sites sources.
        Utilisez avec modération pour éviter de surcharger les serveurs.
    </div>
    """, unsafe_allow_html=True)
    
    # Options de scraping
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source = st.selectbox(
            "Source de données",
            ["worldfootball", "sofascore", "demo"],
            key="scraping_source",
            help="worldfootball.net est généralement plus accessible"
        )
    
    with col2:
        scrape_action = st.button(
            "🔄 Scraper maintenant",
            key="scrape_now",
            type="primary",
            use_container_width=True
        )
    
    with col3:
        auto_scrape = st.checkbox("Auto-scrape", value=False, key="auto_scrape")
        if auto_scrape:
            refresh_interval = st.slider("Intervalle (secondes)", 30, 300, 60, key="scrape_interval")
    
    # Initialiser le scraper
    scraper = FootballScraper()
    
    # Lancer le scraping si demandé
    if scrape_action or auto_scrape or st.session_state.scraping_active:
        st.session_state.scraping_active = True
        
        with st.spinner(f"Scraping des données depuis {source}..."):
            matches = scraper.scrape_matches(source)
        
        if matches:
            st.success(f"✅ {len(matches)} matchs trouvés")
            
            # Afficher les matchs
            st.markdown("### ⚽ Matchs Scrapés")
            
            live_count = sum(1 for m in matches if m['status'] == 'LIVE')
            upcoming_count = sum(1 for m in matches if m['status'] == 'NS')
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Matchs en direct", live_count)
            with col2:
                st.metric("Matchs à venir", upcoming_count)
            
            # Filtrer par ligue
            all_leagues = list(set(m['league'] for m in matches))
            selected_leagues = st.multiselect(
                "Filtrer par ligue",
                all_leagues,
                default=all_leagues[:3] if len(all_leagues) > 3 else all_leagues,
                key="scraped_leagues_filter"
            )
            
            filtered_matches = [m for m in matches if m['league'] in selected_leagues] if selected_leagues else matches
            
            # Trier par statut
            status_order = {'LIVE': 0, 'HT': 1, 'NS': 2, 'FT': 3}
            filtered_matches.sort(key=lambda x: status_order.get(x['status'], 4))
            
            # Afficher chaque match
            for i, match in enumerate(filtered_matches):
                display_scraped_match(match, i)
            
            # Afficher les classements scrapés
            st.markdown("### 🏆 Classements Scrapés")
            
            league_for_standings = st.selectbox(
                "Choisir une ligue pour le classement",
                ["Ligue 1", "Premier League", "La Liga", "Bundesliga", "Serie A"],
                key="scraped_standings_league"
            )
            
            with st.spinner(f"Scraping du classement {league_for_standings}..."):
                standings = scraper.scrape_standings(league_for_standings)
            
            if standings:
                df_standings = pd.DataFrame(standings)
                
                # Mise en forme du dataframe
                st.dataframe(
                    df_standings,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'position': st.column_config.NumberColumn("Pos", format="%d"),
                        'team': st.column_config.TextColumn("Équipe"),
                        'matches': st.column_config.NumberColumn("MJ", format="%d"),
                        'wins': st.column_config.NumberColumn("G", format="%d"),
                        'draws': st.column_config.NumberColumn("N", format="%d"),
                        'losses': st.column_config.NumberColumn("P", format="%d"),
                        'goals_for': st.column_config.NumberColumn("BP", format="%d"),
                        'goals_against': st.column_config.NumberColumn("BC", format="%d"),
                        'goal_diff': st.column_config.TextColumn("+/-"),
                        'points': st.column_config.ProgressColumn(
                            "PTS",
                            format="%d",
                            min_value=0,
                            max_value=100
                        )
                    }
                )
                
                # Graphique du classement
                fig = go.Figure(data=[
                    go.Bar(
                        x=df_standings['team'],
                        y=df_standings['points'].astype(int),
                        marker_color=['#667eea' if int(p) >= 50 else '#f093fb' if int(p) >= 40 else '#764ba2' 
                                     for p in df_standings['points']],
                        text=df_standings['points'],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title=f"Classement {league_for_standings} - Points",
                    xaxis_title="Équipes",
                    yaxis_title="Points",
                    template="plotly_white",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques des données scrapées
            st.markdown("### 📊 Statistiques des Données")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                leagues_count = len(set(m['league'] for m in matches))
                st.metric("Ligues couvertes", leagues_count)
            
            with col2:
                total_goals = sum(int(m['home_score']) + int(m['away_score']) for m in matches 
                                 if m['home_score'].isdigit() and m['away_score'].isdigit())
                st.metric("Buts totaux", total_goals)
            
            with col3:
                avg_goals = total_goals / len(matches) if matches else 0
                st.metric("Moyenne buts/match", f"{avg_goals:.2f}")
        
        else:
            st.warning("Aucun match trouvé. Affichage des données de démo.")
            demo_matches = scraper.get_demo_matches()
            for i, match in enumerate(demo_matches):
                display_scraped_match(match, i)
    
    # Section d'information
    with st.expander("ℹ️ Informations sur le scraping", expanded=False):
        st.markdown("""
        **Sources utilisées :**
        - 🌐 **worldfootball.net** : Site de statistiques footballistiques
        - 📱 **SofaScore** : Données de démo uniquement
        - 🎮 **Démo** : Données générées aléatoirement
        
        **Fonctionnalités :**
        - Récupération des scores en direct
        - Classements des ligues
        - Filtrage par ligue
        - Mise à jour manuelle/automatique
        
        **Limitations :**
        - Dépend de la disponibilité des sites sources
        - Peut être bloqué par certaines protections anti-bot
        - Données parfois incomplètes
        
        **Recommandations :**
        - Utilisez le mode "Auto-scrape" avec modération
        - Privilégiez les API officielles pour un usage intensif
        - Respectez les conditions d'utilisation des sites
        """)
    
    # Auto-refresh si activé
    if auto_scrape:
        time.sleep(refresh_interval)
        st.rerun()

# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================
def main():
    load_css()
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="main-header">
            <h1 style="color: white; text-align: center; margin: 0;">⚽ FOOTBALL BETTING ANALYTICS LIVE</h1>
            <p style="color: white; text-align: center; margin: 10px 0 0 0; opacity: 0.9;">
                Analyse en temps réel • Données live • Prédictions intelligentes
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        # Configuration API
        api_key = st.text_input(
            "Clé API Football",
            type="password",
            help="Obtenez une clé sur https://dashboard.api-football.com/",
            value=st.session_state.get('api_key', '')
        )
        
        if api_key != st.session_state.get('api_key', ''):
            st.session_state.api_key = api_key
            st.success("Clé API mise à jour!")
        
        st.session_state.timezone = st.selectbox(
            "Fuseau horaire",
            ["Europe/Paris", "Europe/London", "America/New_York", "Asia/Tokyo", "Australia/Sydney"],
            index=0
        )
        
        # Options d'affichage
        st.markdown("### 👁️ Options d'affichage")
        show_live = st.checkbox("Matchs en direct", value=True)
        show_odds = st.checkbox("Afficher les cotes", value=True)
        auto_refresh = st.checkbox("Rafraîchissement auto", value=True)
        
        if auto_refresh:
            refresh_rate = st.slider("Intervalle (secondes)", 30, 300, 60)
            if st.button("🔄 Rafraîchir maintenant", key="refresh_button"):
                st.rerun()
        
        # Options de scraping
        st.markdown("### 🌐 Options de scraping")
        enable_scraping = st.checkbox("Activer le scraping", value=False, 
                                     help="Active la récupération de données via web scraping")
        
        st.markdown("---")
        st.markdown("""
        ### 📊 Statut
        - **Dernière mise à jour:** {}
        - **Mode:** {}
        - **Fuseau:** {}
        - **Scraping:** {}
        """.format(
            st.session_state.last_update.strftime('%H:%M:%S'),
            "API" if st.session_state.api_key else "Démo",
            st.session_state.timezone,
            "Activé" if enable_scraping else "Désactivé"
        ))
    
    # Contenu principal avec onglets
    if enable_scraping:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dashboard Live", 
            "🔮 Prédictions", 
            "📈 Statistiques", 
            "⚽ Matchs", 
            "🏆 Classements",
            "🌐 Données Scrapées"  # NOUVEL ONGLET
        ])
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard Live", 
            "🔮 Prédictions", 
            "📈 Statistiques", 
            "⚽ Matchs", 
            "🏆 Classements"
        ])
    
    # Initialiser le manager API
    api_manager = FootballAPIManager()
    
    with tab1:
        render_dashboard(api_manager, show_live)
    
    with tab2:
        render_predictions(api_manager)
    
    with tab3:
        render_statistics(api_manager)
    
    with tab4:
        render_matches(api_manager, show_live)
    
    with tab5:
        render_standings(api_manager)
    
    # Onglet scraping si activé
    if enable_scraping:
        with tab6:
            render_scraping_page()
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_rate)
        st.session_state.last_update = datetime.now()
        st.rerun()

# Footer
def render_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **ℹ️ À propos**  
        Plateforme d'analyse footballistique en temps réel
        """)
    
    with col2:
        st.markdown("""
        **⚖️ Disclaimer**  
        À but informatif seulement.  
        Paris sportifs = risque financier.
        """)
    
    with col3:
        st.markdown("""
        **🔄 Dernière mise à jour**  
        {}
        """.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

if __name__ == "__main__":
    main()
    render_footer()
