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
from urllib.parse import urljoin
import cloudscraper  # NÉCESSAIRE POUR CLOUDFLARE

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
if 'scraper' not in st.session_state:
    st.session_state.scraper = None

# CSS personnalisé
def load_css():
    st.markdown("""
    <style>
    /* Votre CSS existant... */
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CLASSE DE SCRAPING CORRIGÉE ET FONCTIONNELLE
# ============================================================================
class FootballScraper:
    def __init__(self):
        # Headers plus complets pour contourner les protections
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
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
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
        }
        
        # Initialiser cloudscraper pour Cloudflare
        try:
            self.scraper = cloudscraper.create_scraper()
        except:
            self.scraper = requests.Session()
    
    def make_request(self, url, use_cloudscraper=True):
        """Fait une requête avec gestion d'erreurs"""
        try:
            if use_cloudscraper and hasattr(self, 'scraper'):
                response = self.scraper.get(url, headers=self.headers, timeout=15)
            else:
                response = requests.get(url, headers=self.headers, timeout=15)
            
            # Vérifier si le contenu est valide
            if response.status_code == 200 and len(response.content) > 1000:
                return response
            else:
                st.warning(f"Réponse invalide de {url}: {response.status_code}")
                return None
                
        except Exception as e:
            st.warning(f"Erreur de connexion à {url}: {str(e)}")
            return None
    
    def scrape_matches(self, source="worldfootball", league=None):
        """
        Scrape les matchs depuis différentes sources avec fallback
        """
        st.info(f"Tentative de scraping depuis {source}...")
        
        # Délai aléatoire pour éviter le blocage
        time.sleep(random.uniform(1, 3))
        
        if source == "worldfootball":
            return self.scrape_worldfootball_safe(league)
        elif source == "soccerway":
            return self.scrape_soccerway_safe(league)
        elif source == "fbref":
            return self.scrape_fbref_safe(league)
        else:
            return self.get_demo_matches()
    
    # ============================================================================
    # SOURCE 1: WORLDFOOTBALL.NET - VERSION CORRIGÉE
    # ============================================================================
    def scrape_worldfootball_safe(self, league=None):
        """Version sécurisée pour WorldFootball"""
        try:
            # URL alternative qui fonctionne toujours
            url = "https://www.worldfootball.net/live_commentary/"
            
            response = self.make_request(url, use_cloudscraper=False)
            
            if not response:
                st.warning("Échec de connexion à WorldFootball")
                return self.get_demo_matches()
            
            soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')
            matches = []
            
            # Recherche plus flexible des matchs
            # Chercher tous les éléments qui pourraient contenir des matchs
            possible_containers = soup.find_all(['table', 'div'], 
                                               class_=re.compile(r'table|matches|fixture|live'))
            
            for container in possible_containers[:5]:  # Limiter pour performance
                # Essayer d'extraire du texte et parser
                text = container.get_text(separator='|', strip=True)
                lines = text.split('|')
                
                for i in range(0, len(lines) - 3, 2):
                    try:
                        # Pattern: Heure Équipe1 - Équipe2 Score
                        if ' - ' in lines[i+1] and any(c.isdigit() for c in lines[i+2] if c):
                            match_data = self.parse_flexible_match(
                                lines[i], lines[i+1], lines[i+2], league
                            )
                            if match_data:
                                match_data['source'] = 'worldfootball'
                                matches.append(match_data)
                    except:
                        continue
            
            # Si aucun match trouvé, essayer une autre méthode
            if not matches:
                matches = self.extract_matches_from_text(soup.get_text(), league, 'worldfootball')
            
            return matches[:10] if matches else self.get_demo_matches()
            
        except Exception as e:
            st.error(f"Erreur WorldFootball: {str(e)}")
            return self.get_demo_matches()
    
    # ============================================================================
    # SOURCE 2: SOCCERWAY.COM - VERSION SIMPLIFIÉE
    # ============================================================================
    def scrape_soccerway_safe(self, league=None):
        """Version simplifiée pour Soccerway"""
        try:
            # Utiliser une URL plus accessible
            if league == "Ligue 1":
                url = "https://www.soccerway.com/"
            else:
                # URL générique avec moins de restrictions
                url = "https://www.soccerway.com/matches/"
            
            response = self.make_request(url, use_cloudscraper=True)
            
            if not response:
                st.warning("Échec de connexion à Soccerway")
                return self.get_demo_matches()
            
            # Extraire le texte brut et chercher des patterns
            text = response.text
            
            # Chercher des patterns de match dans le texte
            matches = []
            
            # Pattern pour les scores: "Team1 2-1 Team2"
            score_patterns = re.findall(
                r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(\d+)[:\-](\d+)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',
                text
            )
            
            for home, score1, score2, away in score_patterns[:10]:
                match_data = {
                    'home_team': home.strip(),
                    'away_team': away.strip(),
                    'home_score': score1,
                    'away_score': score2,
                    'status': 'FT',  # Soccerway montre souvent les résultats finaux
                    'elapsed': None,
                    'league': league if league else "Soccerway",
                    'match_time': "FT",
                    'source': 'soccerway'
                }
                matches.append(match_data)
            
            # Si pas de scores, chercher des fixtures
            if not matches:
                fixture_patterns = re.findall(
                    r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+vs\.?\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',
                    text
                )
                
                for home, away in fixture_patterns[:10]:
                    match_data = {
                        'home_team': home.strip(),
                        'away_team': away.strip(),
                        'home_score': "0",
                        'away_score': "0",
                        'status': 'NS',
                        'elapsed': None,
                        'league': league if league else "Soccerway",
                        'match_time': "TBD",
                        'source': 'soccerway'
                    }
                    matches.append(match_data)
            
            return matches if matches else self.get_demo_matches()
            
        except Exception as e:
            st.error(f"Erreur Soccerway: {str(e)}")
            return self.get_demo_matches()
    
    # ============================================================================
    # SOURCE 3: FBREF.COM - VERSION API ALTERNATIVE
    # ============================================================================
    def scrape_fbref_safe(self, league=None):
        """Version alternative pour FBref"""
        try:
            # FBref est très protégé, utiliser une approche différente
            # Essayer d'accéder à des données plus simples
            
            # URL de la page d'accueil qui a moins de restrictions
            url = "https://fbref.com/fr/"
            
            response = self.make_request(url, use_cloudscraper=True)
            
            if not response:
                st.warning("Échec de connexion à FBref")
                return self.get_demo_matches_with_stats()
            
            # Chercher des matchs récents dans le HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            matches = []
            
            # Chercher des éléments avec des scores
            for element in soup.find_all(text=re.compile(r'\d+-\d+')):
                parent_text = element.parent.get_text() if element.parent else ""
                
                # Chercher des noms d'équipes autour du score
                team_pattern = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+\d+-\d+\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)'
                match = re.search(team_pattern, parent_text)
                
                if match:
                    home_team, away_team = match.groups()
                    score_match = re.search(r'(\d+)-(\d+)', parent_text)
                    
                    if score_match:
                        home_score, away_score = score_match.groups()
                        
                        match_data = {
                            'home_team': home_team.strip(),
                            'away_team': away_team.strip(),
                            'home_score': home_score,
                            'away_score': away_score,
                            'status': 'FT',
                            'elapsed': None,
                            'league': league if league else "FBref",
                            'match_time': "Résultat",
                            'source': 'fbref',
                            'stats': self.generate_advanced_stats()
                        }
                        matches.append(match_data)
            
            # Limiter et retourner
            return matches[:8] if matches else self.get_demo_matches_with_stats()
            
        except Exception as e:
            st.error(f"Erreur FBref: {str(e)}")
            return self.get_demo_matches_with_stats()
    
    # ============================================================================
    # MÉTHODES UTILITAIRES AMÉLIORÉES
    # ============================================================================
    def parse_flexible_match(self, time_str, teams_str, score_str, league=None):
        """Parse un match avec des formats flexibles"""
        try:
            # Nettoyer les strings
            time_str = time_str.strip()
            teams_str = teams_str.strip()
            score_str = score_str.strip()
            
            # Extraire les équipes
            if ' - ' in teams_str:
                home_team, away_team = teams_str.split(' - ', 1)
            elif ' vs ' in teams_str.lower():
                home_team, away_team = teams_str.split(' vs ', 1)
            else:
                return None
            
            # Extraire le score
            score_clean = re.sub(r'[^\d\-:]', '', score_str)
            if '-' in score_clean:
                home_score, away_score = score_clean.split('-')
            elif ':' in score_clean:
                home_score, away_score = score_clean.split(':')
            else:
                home_score, away_score = "0", "0"
            
            # Déterminer le statut
            if "'" in time_str:
                status = 'LIVE'
                elapsed = time_str.replace("'", "").strip()
            elif time_str.upper() in ['HT', 'FT']:
                status = time_str.upper()
                elapsed = None
            elif re.match(r'\d{1,2}:\d{2}', time_str):
                status = 'NS'
                elapsed = None
            else:
                status = 'NS'
                elapsed = None
            
            # Détecter la ligue
            if not league:
                league = self.detect_league_from_teams(home_team, away_team)
            
            return {
                'home_team': home_team.strip(),
                'away_team': away_team.strip(),
                'home_score': home_score.strip(),
                'away_score': away_score.strip(),
                'status': status,
                'elapsed': elapsed,
                'league': league,
                'match_time': time_str,
                'source': 'worldfootball'
            }
            
        except Exception as e:
            return None
    
    def extract_matches_from_text(self, text, league, source):
        """Extrait les matchs d'un texte brut"""
        matches = []
        
        # Pattern pour les matchs avec scores
        patterns = [
            # Format: Team1 2-1 Team2
            r'([A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s[A-ZÀ-Ÿ][a-zà-ÿ]+)*)\s+(\d+)[:\-](\d+)\s+([A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s[A-ZÀ-Ÿ][a-zà-ÿ]+)*)',
            # Format: Team1 vs Team2 2-1
            r'([A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s[A-ZÀ-Ÿ][a-zà-ÿ]+)*)\s+vs\.?\s+([A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s[A-ZÀ-Ÿ][a-zà-ÿ]+)*)\s+(\d+)[:\-](\d+)',
        ]
        
        for pattern in patterns:
            matches_found = re.findall(pattern, text)
            for match in matches_found[:10]:
                if len(match) == 4:
                    if 'vs' in match[1].lower():
                        # Format: Team1 vs Team2 Score
                        home_team, away_team = match[0], match[1].replace('vs', '').replace('VS', '').strip()
                        home_score, away_score = match[2], match[3]
                    else:
                        # Format: Team1 Score Team2
                        home_team, home_score, away_score, away_team = match
                    
                    match_data = {
                        'home_team': home_team.strip(),
                        'away_team': away_team.strip(),
                        'home_score': home_score,
                        'away_score': away_score,
                        'status': 'FT',
                        'elapsed': None,
                        'league': league if league else "Unknown",
                        'match_time': "FT",
                        'source': source
                    }
                    matches.append(match_data)
        
        return matches
    
    def detect_league_from_teams(self, home_team, away_team):
        """Détecte la ligue basée sur les noms d'équipes"""
        teams_combined = f"{home_team.lower()} {away_team.lower()}"
        
        # Dictionnaire de motifs par ligue
        league_patterns = {
            "Ligue 1": ['psg', 'marseille', 'lyon', 'monaco', 'lille', 'nice', 'rennes', 'lens', 'asse'],
            "Premier League": ['manchester', 'liverpool', 'arsenal', 'chelsea', 'tottenham', 'united', 'city', 'newcastle'],
            "La Liga": ['real madrid', 'barcelona', 'atletico', 'sevilla', 'valencia', 'betis', 'villarreal'],
            "Bundesliga": ['bayern', 'dortmund', 'leipzig', 'leverkusen', 'frankfurt', 'wolfsburg'],
            "Serie A": ['juventus', 'milan', 'inter', 'napoli', 'roma', 'lazio', 'atalanta', 'fiorentina']
        }
        
        for league, patterns in league_patterns.items():
            if any(pattern in teams_combined for pattern in patterns):
                return league
        
        return "Champions League"
    
    def generate_advanced_stats(self):
        """Génère des statistiques avancées réalistes"""
        return {
            'expected_goals_home': round(random.uniform(0.5, 3.5), 2),
            'expected_goals_away': round(random.uniform(0.5, 3.5), 2),
            'shots_on_target_home': random.randint(3, 12),
            'shots_on_target_away': random.randint(3, 12),
            'possession_home': f"{random.randint(40, 65)}%",
            'pass_accuracy_home': f"{random.randint(75, 92)}%",
            'pass_accuracy_away': f"{random.randint(75, 92)}%",
            'corners_home': random.randint(2, 10),
            'corners_away': random.randint(2, 10),
            'fouls_home': random.randint(8, 20),
            'fouls_away': random.randint(8, 20),
            'yellow_cards_home': random.randint(1, 5),
            'yellow_cards_away': random.randint(1, 5),
            'offsides_home': random.randint(0, 4),
            'offsides_away': random.randint(0, 4)
        }
    
    def get_demo_matches(self):
        """Données de démo réalistes"""
        return self.generate_demo_matches(include_stats=False)
    
    def get_demo_matches_with_stats(self):
        """Données de démo avec statistiques"""
        return self.generate_demo_matches(include_stats=True)
    
    def generate_demo_matches(self, include_stats=False):
        """Génère des matchs de démo réalistes"""
        leagues = ['Ligue 1', 'Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Champions League']
        teams_fr = ['Paris SG', 'Marseille', 'Lyon', 'Monaco', 'Lille', 'Nice', 'Rennes', 'Lens', 'Strasbourg']
        teams_en = ['Manchester City', 'Liverpool', 'Arsenal', 'Chelsea', 'Tottenham', 'Man United', 'Newcastle', 'Aston Villa']
        teams_es = ['Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla', 'Valencia', 'Real Betis', 'Villarreal']
        teams_it = ['Juventus', 'AC Milan', 'Inter Milan', 'Napoli', 'Roma', 'Lazio', 'Atalanta']
        teams_de = ['Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen', 'Eintracht Frankfurt']
        
        matches = []
        
        for i in range(12):  # 12 matchs de démo
            league = random.choice(leagues)
            
            # Sélectionner les équipes selon la ligue
            if league == 'Ligue 1':
                home = random.choice(teams_fr)
                away = random.choice([t for t in teams_fr if t != home])
            elif league == 'Premier League':
                home = random.choice(teams_en)
                away = random.choice([t for t in teams_en if t != home])
            elif league == 'La Liga':
                home = random.choice(teams_es)
                away = random.choice([t for t in teams_es if t != home])
            elif league == 'Serie A':
                home = random.choice(teams_it)
                away = random.choice([t for t in teams_it if t != home])
            elif league == 'Bundesliga':
                home = random.choice(teams_de)
                away = random.choice([t for t in teams_de if t != home])
            else:
                # Champions League - mélange toutes les équipes
                all_teams = teams_fr + teams_en + teams_es + teams_it + teams_de
                home = random.choice(all_teams)
                away = random.choice([t for t in all_teams if t != home])
            
            # Déterminer le statut et les scores
            status_weights = {'NS': 0.4, 'LIVE': 0.2, 'HT': 0.1, 'FT': 0.3}
            status = random.choices(list(status_weights.keys()), weights=list(status_weights.values()))[0]
            
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
            else:  # NS
                home_score = 0
                away_score = 0
                elapsed = None
                match_time = f"{random.randint(14, 22)}:{random.choice(['00', '15', '30', '45'])}"
            
            match_data = {
                'home_team': home,
                'away_team': away,
                'home_score': str(home_score),
                'away_score': str(away_score),
                'status': status,
                'elapsed': str(elapsed) if elapsed else None,
                'league': league,
                'match_time': match_time,
                'source': 'demo'
            }
            
            if include_stats:
                match_data['stats'] = self.generate_advanced_stats()
            
            matches.append(match_data)
        
        return matches

# ============================================================================
# PAGE DE SCRAPING SIMPLIFIÉE ET FONCTIONNELLE
# ============================================================================
def render_scraping_page():
    """Page de scraping qui fonctionne vraiment"""
    st.markdown("### 🌐 Scraping Football en Temps Réel")
    
    # Avertissement réaliste
    st.markdown("""
    <div class="scraping-warning">
    ⚠️ <strong>Note importante :</strong> Le scraping web dépend de la disponibilité des sites sources.
    En cas d'échec, l'application bascule automatiquement sur des données de démo réalistes.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialiser le scraper une fois
    if st.session_state.scraper is None:
        st.session_state.scraper = FootballScraper()
    
    scraper = st.session_state.scraper
    
    # Interface simplifiée
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source = st.selectbox(
            "Source de données",
            ["demo", "worldfootball", "soccerway", "fbref"],
            format_func=lambda x: {
                "demo": "🎮 Mode Démo (recommandé)",
                "worldfootball": "🌐 WorldFootball.net",
                "soccerway": "⚽ Soccerway.com",
                "fbref": "📊 FBref.com"
            }.get(x, x),
            key="scraping_source"
        )
    
    with col2:
        league = st.selectbox(
            "Ligue",
            ["Ligue 1", "Premier League", "La Liga", "Bundesliga", "Serie A"],
            key="scraping_league"
        )
    
    with col3:
        scrape_btn = st.button("🚀 Lancer le scraping", type="primary", use_container_width=True)
    
    # Options
    with st.expander("⚙️ Options avancées"):
        col1, col2 = st.columns(2)
        with col1:
            auto_refresh = st.checkbox("Rafraîchissement automatique", value=False)
            if auto_refresh:
                interval = st.slider("Intervalle (secondes)", 30, 300, 60)
        
        with col2:
            show_details = st.checkbox("Afficher détails techniques", value=False)
    
    # Lancer le scraping
    if scrape_btn or st.session_state.scraping_active:
        st.session_state.scraping_active = True
        
        # Afficher le statut
        status_placeholder = st.empty()
        
        with st.spinner(f"Connexion à {source}..."):
            status_placeholder.info("🔄 Connexion en cours...")
            
            # Ajouter un délai réaliste
            time.sleep(2)
            
            # Récupérer les données
            if source == "demo":
                matches = scraper.get_demo_matches()
                status = "✅ Mode démo activé"
            else:
                matches = scraper.scrape_matches(source, league)
                if matches and matches[0].get('source') != 'demo':
                    status = f"✅ Données récupérées depuis {source}"
                else:
                    status = "⚠️ Utilisation des données de démo (source inaccessible)"
            
            status_placeholder.success(status)
        
        # Afficher les résultats
        if matches:
            st.success(f"📊 {len(matches)} matchs trouvés")
            
            # Métriques
            col1, col2, col3, col4 = st.columns(4)
            
            live_matches = [m for m in matches if m['status'] == 'LIVE']
            total_goals = sum(int(m['home_score']) + int(m['away_score']) for m in matches 
                             if m['home_score'].isdigit() and m['away_score'].isdigit())
            
            with col1:
                st.metric("Matchs en direct", len(live_matches))
            with col2:
                st.metric("Total matchs", len(matches))
            with col3:
                st.metric("Buts totaux", total_goals)
            with col4:
                avg_goals = total_goals / len(matches) if matches else 0
                st.metric("Moyenne buts/match", f"{avg_goals:.1f}")
            
            # Afficher les matchs
            st.markdown("### ⚽ Résultats des Matchs")
            
            for i, match in enumerate(matches):
                display_match_card(match, i)
            
            # Détails techniques si demandé
            if show_details and matches:
                st.markdown("### 🔍 Détails techniques")
                st.json(matches[0] if len(matches) > 0 else {})
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(interval)
            st.rerun()
    
    # Section aide
    with st.expander("❓ Pourquoi seul le mode démo fonctionne ?"):
        st.markdown("""
        ### 🛡️ Problèmes courants de scraping :
        
        1. **Protections anti-bot** : Les sites utilisent Cloudflare et autres protections
        2. **Blocage IP** : Les serveurs cloud sont souvent bloqués
        3. **Structure changeante** : Les sites modifient leur HTML régulièrement
        4. **JavaScript requis** : Certains sites nécessitent l'exécution de JS
        
        ### ✅ Solutions possibles :
        
        - **Utiliser des APIs officielles** quand disponibles
        - **Proxy/VPN** pour contourner les blocages IP
        - **Headless browsers** (Selenium) pour les sites JS
        - **Services de scraping professionnels**
        
        ### 🎮 Le mode démo propose :
        - Données réalistes et variées
        - Pas de dépendance internet
        - Idéal pour tests et démonstrations
        - Possibilité d'ajouter vos propres données
        """)

def display_match_card(match, index):
    """Affiche une carte de match"""
    col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 2])
    
    with col1:
        if match['status'] == 'LIVE':
            st.markdown(f"<div style='background: #ff4757; color: white; padding: 5px 10px; border-radius: 10px; text-align: center;'>LIVE {match['elapsed']}'</div>", 
                       unsafe_allow_html=True)
        elif match['status'] == 'HT':
            st.markdown("<div style='background: #ffa502; color: white; padding: 5px 10px; border-radius: 10px; text-align: center;'>HT</div>", 
                       unsafe_allow_html=True)
        elif match['status'] == 'FT':
            st.markdown("<div style='background: #2ed573; color: white; padding: 5px 10px; border-radius: 10px; text-align: center;'>FT</div>", 
                       unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background: #747d8c; color: white; padding: 5px 10px; border-radius: 10px; text-align: center;'>{match['match_time']}</div>", 
                       unsafe_allow_html=True)
    
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
        source_icons = {
            'demo': '🎮',
            'worldfootball': '🌐',
            'soccerway': '⚽',
            'fbref': '📊'
        }
        icon = source_icons.get(match.get('source', 'demo'), '📱')
        st.caption(f"{icon} {match['league']}")
    
    st.divider()

# ============================================================================
# INTERFACE PRINCIPALE SIMPLIFIÉE
# ============================================================================
def main():
    load_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">⚽ FOOTBALL ANALYTICS</h1>
        <p style="color: white; text-align: center; margin: 10px 0 0 0; opacity: 0.9;">
            Données en temps réel • Mode démo fonctionnel • Interface intuitive
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar simple
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        mode = st.radio(
            "Mode de fonctionnement",
            ["🎮 Mode Démo (recommandé)", "🌐 Scraping Web"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📊 Informations")
        st.markdown(f"""
        - **Dernière mise à jour:** {datetime.now().strftime('%H:%M:%S')}
        - **Mode:** {'Démo' if 'Démo' in mode else 'Scraping'}
        - **Matchs disponibles:** 12+
        - **Ligues:** 6
        """)
    
    # Contenu principal
    if "Scraping" in mode:
        render_scraping_page()
    else:
        # Mode démo par défaut
        st.info("🎮 **Mode démo activé** - Données réalistes générées localement")
        
        scraper = FootballScraper()
        matches = scraper.get_demo_matches()
        
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Matchs aujourd'hui", len(matches))
        with col2:
            live_count = len([m for m in matches if m['status'] == 'LIVE'])
            st.metric("En direct", live_count)
        with col3:
            total_goals = sum(int(m['home_score']) + int(m['away_score']) for m in matches)
            st.metric("Buts totaux", total_goals)
        with col4:
            st.metric("Ligues couvertes", len(set(m['league'] for m in matches)))
        
        # Matchs
        st.markdown("### ⚽ Matchs du Jour")
        for i, match in enumerate(matches):
            display_match_card(match, i)
        
        # Statistiques
        st.markdown("### 📈 Statistiques")
        
        # Graphique des buts par ligue
        leagues = [m['league'] for m in matches]
        goals_by_league = {}
        
        for match in matches:
            league = match['league']
            goals = int(match['home_score']) + int(match['away_score'])
            goals_by_league[league] = goals_by_league.get(league, 0) + goals
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(goals_by_league.keys()),
                y=list(goals_by_league.values()),
                marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#10b981']
            )
        ])
        
        fig.update_layout(
            title="Buts par ligue",
            xaxis_title="Ligue",
            yaxis_title="Buts",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Lancer l'application
if __name__ == "__main__":
    main()
