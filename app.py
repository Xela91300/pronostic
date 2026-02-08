# app.py - Système d'Analyse et Pronostics de Matchs Football
# Version avec API réelle et corrections Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import random
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION API
# =============================================================================

class APIConfig:
    """Configuration API Football"""
    API_FOOTBALL_KEY = "249b3051eCA063F0e381609128c00d7d"
    API_FOOTBALL_URL = "https://v3.football.api-sports.io"
    CACHE_DURATION = 1800  # 30 minutes

# =============================================================================
# CLIENT API SIMPLIFIÉ
# =============================================================================

class FootballAPIClient:
    """Client API Football avec données simulées en cas d'erreur"""
    
    def __init__(self):
        self.config = APIConfig()
        self.session = requests.Session()
        self.session.headers.update({
            'x-apisports-key': self.config.API_FOOTBALL_KEY,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.use_simulation = False
        self.test_connection()
    
    def test_connection(self):
        """Teste la connexion à l'API"""
        try:
            url = f"{self.config.API_FOOTBALL_URL}/status"
            response = self.session.get(url, timeout=5)
            self.use_simulation = response.status_code != 200
            return not self.use_simulation
        except:
            self.use_simulation = True
            return False
    
    def get_todays_fixtures(self) -> List[Dict]:
        """Récupère les matchs du jour"""
        if self.use_simulation:
            return self._simulate_todays_fixtures()
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures"
            params = {
                'date': date.today().strftime('%Y-%m-%d'),
                'timezone': 'Europe/Paris'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                fixtures = []
                
                for fixture in data:
                    fixture_data = fixture.get('fixture', {})
                    teams = fixture.get('teams', {})
                    league = fixture.get('league', {})
                    
                    # Ne prendre que les matchs à venir
                    status = fixture_data.get('status', {}).get('short')
                    if status in ['NS', 'TBD', 'PST']:
                        fixtures.append({
                            'fixture_id': fixture_data.get('id'),
                            'date': fixture_data.get('date'),
                            'home_id': teams.get('home', {}).get('id'),
                            'home_name': teams.get('home', {}).get('name'),
                            'away_id': teams.get('away', {}).get('id'),
                            'away_name': teams.get('away', {}).get('name'),
                            'league_id': league.get('id'),
                            'league_name': league.get('name'),
                            'league_country': league.get('country'),
                            'status': status
                        })
                
                return fixtures
            
            return self._simulate_todays_fixtures()
        except:
            return self._simulate_todays_fixtures()
    
    def get_upcoming_fixtures(self, days_ahead: int = 3) -> List[Dict]:
        """Récupère les matchs à venir"""
        if self.use_simulation:
            return self._simulate_upcoming_fixtures(days_ahead)
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures"
            params = {
                'from': date.today().strftime('%Y-%m-%d'),
                'to': (date.today() + timedelta(days=days_ahead)).strftime('%Y-%m-%d'),
                'timezone': 'Europe/Paris'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                fixtures = []
                
                for fixture in data:
                    fixture_data = fixture.get('fixture', {})
                    teams = fixture.get('teams', {})
                    league = fixture.get('league', {})
                    
                    # Ne prendre que les matchs à venir
                    status = fixture_data.get('status', {}).get('short')
                    if status in ['NS', 'TBD', 'PST']:
                        fixtures.append({
                            'fixture_id': fixture_data.get('id'),
                            'date': fixture_data.get('date'),
                            'home_id': teams.get('home', {}).get('id'),
                            'home_name': teams.get('home', {}).get('name'),
                            'away_id': teams.get('away', {}).get('id'),
                            'away_name': teams.get('away', {}).get('name'),
                            'league_id': league.get('id'),
                            'league_name': league.get('name'),
                            'league_country': league.get('country'),
                            'status': status
                        })
                
                return fixtures
            
            return self._simulate_upcoming_fixtures(days_ahead)
        except:
            return self._simulate_upcoming_fixtures(days_ahead)
    
    def get_fixture_odds(self, fixture_id: int) -> Dict:
        """Récupère les cotes pour un match"""
        if self.use_simulation:
            return self._simulate_odds()
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/odds"
            params = {'fixture': fixture_id}
            
            response = self.session.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                if data:
                    return data[0]
            
            return self._simulate_odds()
        except:
            return self._simulate_odds()
    
    def get_team_statistics(self, team_id: int, league_id: int) -> Dict:
        """Récupère les statistiques d'une équipe"""
        if self.use_simulation:
            return self._simulate_team_stats()
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/teams/statistics"
            params = {
                'team': team_id,
                'league': league_id,
                'season': 2024
            }
            
            response = self.session.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                return response.json().get('response', {})
            
            return self._simulate_team_stats()
        except:
            return self._simulate_team_stats()
    
    def get_head_to_head(self, team1_id: int, team2_id: int) -> List[Dict]:
        """Récupère l'historique des confrontations"""
        if self.use_simulation:
            return self._simulate_h2h()
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures/headtohead"
            params = {
                'h2h': f"{team1_id}-{team2_id}",
                'last': 3
            }
            
            response = self.session.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                return data[:3]
            
            return self._simulate_h2h()
        except:
            return self._simulate_h2h()
    
    # Méthodes de simulation
    def _simulate_todays_fixtures(self) -> List[Dict]:
        """Simule les matchs du jour"""
        teams = [
            ('PSG', 'Marseille'), ('Real Madrid', 'Barcelona'), 
            ('Manchester City', 'Liverpool'), ('Bayern Munich', 'Borussia Dortmund'),
            ('Juventus', 'Inter Milan'), ('Lille', 'Monaco'), ('Arsenal', 'Chelsea')
        ]
        
        fixtures = []
        today = date.today()
        
        for i, (home, away) in enumerate(teams[:5]):
            hour = random.randint(18, 22)
            minute = random.choice([0, 30])
            
            fixtures.append({
                'fixture_id': random.randint(1000, 9999),
                'date': f"{today.strftime('%Y-%m-%d')}T{hour:02d}:{minute:02d}:00+00:00",
                'home_name': home,
                'away_name': away,
                'league_name': random.choice(['Ligue 1', 'La Liga', 'Premier League', 'Bundesliga', 'Serie A']),
                'league_country': random.choice(['France', 'Spain', 'England', 'Germany', 'Italy']),
                'status': 'NS'
            })
        
        return fixtures
    
    def _simulate_upcoming_fixtures(self, days_ahead: int) -> List[Dict]:
        """Simule les matchs à venir"""
        teams = [
            'PSG', 'Marseille', 'Lyon', 'Monaco', 'Lille', 'Nice',
            'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla',
            'Manchester City', 'Liverpool', 'Arsenal', 'Chelsea',
            'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig',
            'Juventus', 'Inter Milan', 'AC Milan', 'Napoli', 'Roma'
        ]
        
        fixtures = []
        
        for day in range(days_ahead + 1):
            match_date = date.today() + timedelta(days=day)
            
            # Créer 3-5 matchs par jour
            for _ in range(random.randint(3, 5)):
                home, away = random.sample(teams, 2)
                hour = random.randint(16, 22)
                minute = random.choice([0, 30])
                
                fixtures.append({
                    'fixture_id': random.randint(1000, 9999),
                    'date': f"{match_date.strftime('%Y-%m-%d')}T{hour:02d}:{minute:02d}:00+00:00",
                    'home_name': home,
                    'away_name': away,
                    'league_name': random.choice(['Ligue 1', 'La Liga', 'Premier League', 'Bundesliga', 'Serie A']),
                    'league_country': random.choice(['France', 'Spain', 'England', 'Germany', 'Italy']),
                    'status': 'NS'
                })
        
        return fixtures[:30]  # Limiter à 30 matchs
    
    def _simulate_odds(self) -> Dict:
        """Simule les cotes"""
        return {
            'bookmakers': [{
                'name': 'Bet365',
                'bets': [{
                    'name': 'Match Winner',
                    'values': [
                        {'value': 'Home', 'odd': round(random.uniform(1.5, 3.0), 2)},
                        {'value': 'Draw', 'odd': round(random.uniform(3.0, 4.5), 2)},
                        {'value': 'Away', 'odd': round(random.uniform(2.0, 4.0), 2)}
                    ]
                }]
            }]
        }
    
    def _simulate_team_stats(self) -> Dict:
        """Simule les statistiques d'équipe"""
        return {
            'fixtures': {
                'played': {'total': random.randint(20, 30)},
                'wins': {'total': random.randint(10, 20)},
                'draws': {'total': random.randint(5, 10)},
                'loses': {'total': random.randint(5, 10)}
            },
            'goals': {
                'for': {'total': random.randint(30, 60)},
                'against': {'total': random.randint(20, 40)}
            }
        }
    
    def _simulate_h2h(self) -> List[Dict]:
        """Simule l'historique des confrontations"""
        matches = []
        
        for i in range(3):
            days_ago = random.randint(100, 500)
            match_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            
            matches.append({
                'fixture': {'date': match_date},
                'goals': {
                    'home': random.randint(0, 3),
                    'away': random.randint(0, 3)
                }
            })
        
        return matches

# =============================================================================
# SYSTÈME DE PRÉDICTION
# =============================================================================

class PredictionEngine:
    """Moteur de prédiction pour les matchs de football"""
    
    def __init__(self, api_client: FootballAPIClient):
        self.api_client = api_client
        self.team_ratings = {}
        self._initialize_ratings()
    
    def _initialize_ratings(self):
        """Initialise les ratings des équipes populaires"""
        popular_teams = {
            'PSG': 90, 'Marseille': 78, 'Lyon': 76, 'Monaco': 75, 'Lille': 77,
            'Real Madrid': 92, 'Barcelona': 89, 'Atletico Madrid': 85,
            'Manchester City': 93, 'Liverpool': 90, 'Arsenal': 87, 'Chelsea': 85,
            'Bayern Munich': 91, 'Borussia Dortmund': 84,
            'Juventus': 86, 'Inter Milan': 85, 'AC Milan': 83, 'Napoli': 84
        }
        
        self.team_ratings = popular_teams
    
    def get_team_rating(self, team_name: str) -> float:
        """Retourne le rating d'une équipe"""
        if team_name in self.team_ratings:
            return self.team_ratings[team_name]
        
        # Générer un rating pour les équipes inconnues
        rating = random.uniform(65, 85)
        self.team_ratings[team_name] = rating
        return rating
    
    def analyze_fixture(self, fixture: Dict) -> Dict:
        """Analyse un match et génère des prédictions"""
        
        home_team = fixture.get('home_name', 'Home')
        away_team = fixture.get('away_name', 'Away')
        league = fixture.get('league_name', 'Unknown')
        
        # Ratings des équipes
        home_rating = self.get_team_rating(home_team)
        away_rating = self.get_team_rating(away_team)
        
        # Avantage du terrain
        home_advantage = 1.15
        
        # Calcul des probabilités
        home_strength = home_rating * home_advantage
        away_strength = away_rating
        
        total_strength = home_strength + away_strength
        
        home_win_prob = (home_strength / total_strength) * 100 * 0.9
        away_win_prob = (away_strength / total_strength) * 100 * 0.9
        draw_prob = 100 - home_win_prob - away_win_prob
        
        # Ajustements basés sur la ligue
        if 'Ligue 1' in league:
            draw_prob += 5
        elif 'Premier League' in league:
            home_win_prob += 3
        elif 'La Liga' in league:
            draw_prob += 3
        
        # Normalisation
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob = (home_win_prob / total) * 100
        draw_prob = (draw_prob / total) * 100
        away_win_prob = (away_win_prob / total) * 100
        
        # Prédiction principale
        if home_win_prob > away_win_prob and home_win_prob > draw_prob:
            main_prediction = f"Victoire {home_team}"
            prediction_type = "1"
            confidence = home_win_prob
        elif away_win_prob > home_win_prob and away_win_prob > draw_prob:
            main_prediction = f"Victoire {away_team}"
            prediction_type = "2"
            confidence = away_win_prob
        else:
            main_prediction = "Match nul"
            prediction_type = "X"
            confidence = draw_prob
        
        # Score probable
        home_goals = self._predict_goals(home_rating, away_rating, is_home=True)
        away_goals = self._predict_goals(away_rating, home_rating, is_home=False)
        
        # Recommandation de pari
        bet_recommendation = self._get_bet_recommendation(
            home_win_prob, draw_prob, away_win_prob, home_rating, away_rating
        )
        
        return {
            'match': f"{home_team} vs {away_team}",
            'league': league,
            'date': fixture.get('date', ''),
            'time': fixture.get('date', '')[11:16] if len(fixture.get('date', '')) > 16 else '',
            'probabilities': {
                'home_win': round(home_win_prob, 1),
                'draw': round(draw_prob, 1),
                'away_win': round(away_win_prob, 1)
            },
            'main_prediction': main_prediction,
            'prediction_type': prediction_type,
            'confidence': round(confidence, 1),
            'score_prediction': f"{home_goals}-{away_goals}",
            'bet_recommendation': bet_recommendation,
            'home_rating': round(home_rating, 1),
            'away_rating': round(away_rating, 1),
            'analysis': self._generate_analysis(home_team, away_team, home_rating, away_rating)
        }
    
    def _predict_goals(self, attack_rating: float, defense_rating: float, is_home: bool = True) -> int:
        """Prédit le nombre de buts"""
        base_goals = (attack_rating / defense_rating) * 1.5
        
        if is_home:
            base_goals *= 1.2
        
        # Ajouter de l'aléatoire
        goals = max(0, int(round(base_goals + random.uniform(-0.5, 1.0))))
        
        # Limiter à 4 buts maximum
        return min(goals, 4)
    
    def _get_bet_recommendation(self, home_prob: float, draw_prob: float, 
                               away_prob: float, home_rating: float, away_rating: float) -> Dict:
        """Génère une recommandation de pari"""
        
        max_prob = max(home_prob, draw_prob, away_prob)
        
        if max_prob == home_prob:
            bet_type = "1"
            odd = round(1 / (home_prob / 100) * 0.95, 2)
        elif max_prob == away_prob:
            bet_type = "2"
            odd = round(1 / (away_prob / 100) * 0.95, 2)
        else:
            bet_type = "X"
            odd = round(1 / (draw_prob / 100) * 0.95, 2)
        
        # Évaluer la valeur
        value_score = (odd * (max_prob / 100) - 1) * 100
        
        if value_score > 8:
            value = "Excellente"
            color = "🟢"
        elif value_score > 4:
            value = "Bonne"
            color = "🟡"
        else:
            value = "Correcte"
            color = "🟠"
        
        return {
            'type': bet_type,
            'odd': odd,
            'value': value,
            'color': color,
            'value_score': round(value_score, 1)
        }
    
    def _generate_analysis(self, home_team: str, away_team: str, 
                          home_rating: float, away_rating: float) -> str:
        """Génère une analyse textuelle"""
        
        diff = home_rating - away_rating
        
        if diff > 15:
            return f"{home_team} est clairement favori à domicile avec un avantage significatif."
        elif diff > 5:
            return f"{home_team} a un léger avantage à domicile face à {away_team}."
        elif diff > -5:
            return f"Match équilibré entre {home_team} et {away_team}. Tout est possible."
        elif diff > -15:
            return f"{away_team} pourrait créer la surprise en déplacement."
        else:
            return f"{away_team} est le favori clair de cette rencontre."

# =============================================================================
# APPLICATION STREAMLIT
# =============================================================================

def main():
    """Application principale Streamlit"""
    
    # Configuration de la page
    st.set_page_config(
        page_title="Pronostics Football Pro",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
    }
    .confidence-high {
        background: #4CAF50;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .confidence-medium {
        background: #FF9800;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .confidence-low {
        background: #f44336;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-title">⚽ PRONOSTICS FOOTBALL PROFESSIONNELS</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">Analyse en temps réel • Prédictions précises • Meilleures opportunités</div>', unsafe_allow_html=True)
    
    # Initialisation des sessions
    if 'api_client' not in st.session_state:
        st.session_state.api_client = FootballAPIClient()
    
    if 'prediction_engine' not in st.session_state:
        st.session_state.prediction_engine = PredictionEngine(st.session_state.api_client)
    
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURATION")
        
        # Test de connexion
        api_status = st.session_state.api_client.test_connection()
        if api_status:
            st.success("✅ API Connectée")
        else:
            st.warning("⚠️ Mode simulation activé")
        
        st.divider()
        
        # Paramètres
        st.header("🎯 PARAMÈTRES")
        
        days_ahead = st.slider(
            "Jours à analyser",
            min_value=1,
            max_value=7,
            value=3,
            key="days_ahead_slider"
        )
        
        min_confidence = st.slider(
            "Confiance minimum (%)",
            min_value=50,
            max_value=95,
            value=60,
            key="min_confidence_slider"
        )
        
        # Bouton d'analyse
        if st.button("🚀 ANALYSER LES MATCHS", type="primary", use_container_width=True, key="analyze_button"):
            with st.spinner("Récupération des matchs..."):
                fixtures = st.session_state.api_client.get_upcoming_fixtures(days_ahead=days_ahead)
                
                if not fixtures:
                    st.error("Aucun match trouvé")
                else:
                    st.session_state.predictions = []
                    progress_bar = st.progress(0)
                    
                    for i, fixture in enumerate(fixtures):
                        try:
                            prediction = st.session_state.prediction_engine.analyze_fixture(fixture)
                            if prediction['confidence'] >= min_confidence:
                                st.session_state.predictions.append(prediction)
                        except:
                            pass
                        
                        progress_bar.progress((i + 1) / len(fixtures))
                    
                    progress_bar.empty()
                    
                    if st.session_state.predictions:
                        # Trier par confiance
                        st.session_state.predictions.sort(key=lambda x: x['confidence'], reverse=True)
                        st.success(f"✅ {len(st.session_state.predictions)} prédictions générées")
                    else:
                        st.warning("Aucune prédiction ne correspond aux critères")
                    
                    st.rerun()
        
        st.divider()
        
        # Statistiques
        st.header("📊 STATISTIQUES")
        
        if st.session_state.predictions:
            total = len(st.session_state.predictions)
            avg_conf = np.mean([p['confidence'] for p in st.session_state.predictions])
            
            st.metric("Pronostics", total)
            st.metric("Confiance moyenne", f"{avg_conf:.1f}%")
        
        st.divider()
        
        # Info
        st.header("ℹ️ GUIDE")
        st.info("""
        **Légende:**
        • 🟢 Excellente valeur
        • 🟡 Bonne valeur
        • 🟠 Valeur correcte
        
        **Confidence:**
        • >70%: Élevée
        • 60-70%: Moyenne
        • <60%: Faible
        """)
    
    # Onglets principaux
    tab1, tab2, tab3 = st.tabs(["🏆 PRONOSTICS", "📅 MATCHS", "📈 ANALYSE"])
    
    with tab1:
        display_predictions()
    
    with tab2:
        display_fixtures(days_ahead)
    
    with tab3:
        display_analysis()

def display_predictions():
    """Affiche les prédictions"""
    
    st.header("🏆 MEILLEURS PRONOSTICS")
    
    if not st.session_state.predictions:
        st.info("""
        👋 **Bienvenue !**
        
        Pour commencer :
        1. Configurez les paramètres dans la sidebar
        2. Cliquez sur "🚀 ANALYSER LES MATCHS"
        3. Les pronostics apparaîtront ici
        """)
        return
    
    # Filtres
    col1, col2 = st.columns(2)
    
    with col1:
        filter_confidence = st.slider(
            "Filtrer par confiance", 
            50, 100, 60,
            key="filter_confidence_slider"
        )
    
    with col2:
        filter_league = st.selectbox(
            "Filtrer par ligue",
            ["Toutes"] + list(set([p['league'] for p in st.session_state.predictions])),
            key="filter_league_select"
        )
    
    # Appliquer les filtres
    filtered_predictions = [
        p for p in st.session_state.predictions 
        if p['confidence'] >= filter_confidence
    ]
    
    if filter_league != "Toutes":
        filtered_predictions = [p for p in filtered_predictions if p['league'] == filter_league]
    
    st.success(f"📊 **{len(filtered_predictions)} pronostics filtrés**")
    
    if not filtered_predictions:
        st.warning("Aucun pronostic ne correspond aux filtres")
        return
    
    # Afficher les prédictions
    for idx, pred in enumerate(filtered_predictions[:20]):  # Limiter à 20
        with st.container():
            col_pred1, col_pred2 = st.columns([3, 2])
            
            with col_pred1:
                # Informations du match
                st.markdown(f"### {pred['match']}")
                st.write(f"**{pred['league']}** • {pred['date'][:10]} {pred['time']}")
                
                # Prédiction principale
                st.markdown(f"**🎯 PRONOSTIC:** {pred['main_prediction']}")
                
                # Score prédit
                st.markdown(f"**⚽ SCORE:** {pred['score_prediction']}")
                
                # Analyse
                with st.expander("📝 Analyse détaillée", key=f"analysis_{idx}"):
                    st.write(pred['analysis'])
                    st.write(f"**Ratings:** {pred['home_rating']} vs {pred['away_rating']}")
            
            with col_pred2:
                # Confiance
                confidence = pred['confidence']
                if confidence >= 75:
                    confidence_class = "confidence-high"
                    confidence_text = "ÉLEVÉE"
                elif confidence >= 65:
                    confidence_class = "confidence-medium"
                    confidence_text = "MOYENNE"
                else:
                    confidence_class = "confidence-low"
                    confidence_text = "FAIBLE"
                
                st.markdown(f'<div class="{confidence_class}" style="text-align: center; padding: 15px; border-radius: 10px;">'
                          f'<h3>{confidence_text}</h3>'
                          f'<h2>{confidence}%</h2>'
                          f'</div>', unsafe_allow_html=True)
                
                # Probabilités
                probs = pred['probabilities']
                st.write("**Probabilités:**")
                st.write(f"• 1: {probs['home_win']}%")
                st.write(f"• X: {probs['draw']}%")
                st.write(f"• 2: {probs['away_win']}%")
                
                # Recommandation de pari
                bet = pred['bet_recommendation']
                st.markdown(f"**💰 PARI:** {bet['type']} @ {bet['odd']}")
                st.markdown(f"**Valeur:** {bet['value']} {bet['color']}")
            
            st.divider()

def display_fixtures(days_ahead: int):
    """Affiche les matchs disponibles"""
    
    st.header("📅 MATCHS DISPONIBLES")
    
    # Récupérer les matchs
    fixtures = st.session_state.api_client.get_upcoming_fixtures(days_ahead=days_ahead)
    
    if not fixtures:
        st.info("Aucun match trouvé pour cette période")
        return
    
    # Regrouper par date
    fixtures_by_date = {}
    for fixture in fixtures:
        date_str = fixture['date'][:10]
        if date_str not in fixtures_by_date:
            fixtures_by_date[date_str] = []
        fixtures_by_date[date_str].append(fixture)
    
    # Afficher par date
    for date_str, date_fixtures in sorted(fixtures_by_date.items()):
        st.subheader(f"📅 {date_str}")
        
        for fixture in date_fixtures:
            col_fix1, col_fix2, col_fix3 = st.columns([3, 1, 3])
            
            with col_fix1:
                st.write(f"**{fixture['home_name']}**")
            
            with col_fix2:
                st.write("**VS**")
                st.write(f"{fixture['date'][11:16]}")
            
            with col_fix3:
                st.write(f"**{fixture['away_name']}**")
            
            st.write(f"📍 {fixture['league_name']}")
            
            # Bouton pour analyser ce match
            if st.button(f"🔍 Analyser", key=f"analyze_{fixture.get('fixture_id', random.randint(1000, 9999))}"):
                with st.spinner("Analyse en cours..."):
                    try:
                        prediction = st.session_state.prediction_engine.analyze_fixture(fixture)
                        
                        # Afficher l'analyse rapide
                        st.markdown("---")
                        st.subheader(f"⚡ ANALYSE RAPIDE: {prediction['match']}")
                        
                        col_quick1, col_quick2 = st.columns(2)
                        
                        with col_quick1:
                            st.markdown(f"**🎯 Pronostic:** {prediction['main_prediction']}")
                            st.markdown(f"**📊 Confiance:** {prediction['confidence']}%")
                            st.markdown(f"**⚽ Score:** {prediction['score_prediction']}")
                        
                        with col_quick2:
                            bet = prediction['bet_recommendation']
                            st.markdown(f"**💰 Pari recommandé:** {bet['type']} @ {bet['odd']}")
                            st.markdown(f"**📈 Valeur:** {bet['value']} {bet['color']}")
                        
                        st.markdown("---")
                    except:
                        st.error("Erreur lors de l'analyse")
            
            st.divider()

def display_analysis():
    """Affiche l'analyse détaillée"""
    
    st.header("📈 ANALYSE STATISTIQUE")
    
    if not st.session_state.predictions:
        st.info("Générez d'abord des prédictions pour voir les statistiques")
        return
    
    # Statistiques générales
    predictions = st.session_state.predictions
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric("Total pronostics", len(predictions))
    
    with col_stat2:
        avg_conf = np.mean([p['confidence'] for p in predictions])
        st.metric("Confiance moyenne", f"{avg_conf:.1f}%")
    
    with col_stat3:
        # Distribution des types de paris
        bet_types = [p['prediction_type'] for p in predictions]
        type_counts = {t: bet_types.count(t) for t in set(bet_types)}
        most_common = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "N/A"
        st.metric("Pari le plus fréquent", most_common)
    
    with col_stat4:
        avg_odd = np.mean([p['bet_recommendation']['odd'] for p in predictions])
        st.metric("Cote moyenne", f"{avg_odd:.2f}")
    
    st.divider()
    
    # Graphiques
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Distribution des confiances
        confidences = [p['confidence'] for p in predictions]
        
        if confidences:
            st.subheader("📊 Distribution des confiances")
            hist_data = pd.DataFrame({'Confiance (%)': confidences})
            st.bar_chart(hist_data)
    
    with col_chart2:
        # Distribution des types de paris
        if predictions:
            st.subheader("🎯 Types de pronostics")
            bet_types = [p['prediction_type'] for p in predictions]
            type_df = pd.DataFrame({'Type': bet_types})
            type_counts = type_df['Type'].value_counts()
            st.bar_chart(type_counts)
    
    # Tableau des meilleurs paris
    st.subheader("💰 MEILLEURS PARIS PAR VALEUR")
    
    # Trier par valeur du pari
    best_bets = sorted(
        predictions,
        key=lambda x: x['bet_recommendation']['value_score'],
        reverse=True
    )[:10]
    
    if best_bets:
        bet_data = []
        for pred in best_bets:
            bet = pred['bet_recommendation']
            bet_data.append({
                'Match': pred['match'][:30],
                'Type': bet['type'],
                'Cote': bet['odd'],
                'Valeur': f"{bet['value']} {bet['color']}",
                'Score Valeur': f"{bet['value_score']}%",
                'Confiance': f"{pred['confidence']}%"
            })
        
        df_bets = pd.DataFrame(bet_data)
        st.dataframe(df_bets, use_container_width=True, hide_index=True)
    
    # Analyse des ligues
    st.subheader("🏆 ANALYSE PAR LIGUE")
    
    league_stats = {}
    for pred in predictions:
        league = pred['league']
        if league not in league_stats:
            league_stats[league] = {'count': 0, 'total_conf': 0, 'bets': []}
        
        league_stats[league]['count'] += 1
        league_stats[league]['total_conf'] += pred['confidence']
        league_stats[league]['bets'].append(pred['prediction_type'])
    
    if league_stats:
        league_data = []
        for league, stats in league_stats.items():
            avg_conf = stats['total_conf'] / stats['count']
            most_common_bet = max(set(stats['bets']), key=stats['bets'].count) if stats['bets'] else "N/A"
            
            league_data.append({
                'Ligue': league,
                'Pronostics': stats['count'],
                'Confiance moyenne': f"{avg_conf:.1f}%",
                'Pari fréquent': most_common_bet
            })
        
        df_leagues = pd.DataFrame(league_data)
        st.dataframe(df_leagues, use_container_width=True, hide_index=True)

# =============================================================================
# EXÉCUTION
# =============================================================================

if __name__ == "__main__":
    main()
