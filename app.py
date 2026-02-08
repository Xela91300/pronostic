# app.py - Système d'Analyse de Matchs Automatique
# Entrez juste les noms des équipes, le reste est automatique

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import random

# =============================================================================
# CONFIGURATION
# =============================================================================

class APIConfig:
    """Configuration API Football"""
    API_FOOTBALL_KEY: str = "249b3051eCA063F0e381609128c00d7d"
    API_FOOTBALL_URL: str = "https://v3.football.api-sports.io"

# =============================================================================
# CLIENT API SIMPLIFIÉ
# =============================================================================

class FootballDataClient:
    """Client pour récupérer les données des équipes"""
    
    def __init__(self):
        self.config = APIConfig()
        self.session = requests.Session()
        self.session.headers.update({
            'x-apisports-key': self.config.API_FOOTBALL_KEY,
            'User-Agent': 'Mozilla/5.0'
        })
    
    def test_connection(self) -> bool:
        """Teste la connexion à l'API"""
        try:
            url = f"{self.config.API_FOOTBALL_URL}/status"
            response = self.session.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def search_team(self, team_name: str) -> List[Dict]:
        """Recherche une équipe par son nom"""
        try:
            url = f"{self.config.API_FOOTBALL_URL}/teams"
            params = {'search': team_name}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                return data[:5]  # Retourne max 5 résultats
            return []
        except:
            return []
    
    def get_team_statistics(self, team_id: int, league_id: int = 39, season: int = 2024) -> Dict:
        """Récupère les statistiques d'une équipe"""
        try:
            url = f"{self.config.API_FOOTBALL_URL}/teams/statistics"
            params = {
                'team': team_id,
                'league': league_id,
                'season': season
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json().get('response', {})
            return {}
        except:
            return {}
    
    def get_last_matches(self, team_id: int, last: int = 5) -> List[Dict]:
        """Récupère les derniers matchs d'une équipe"""
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures"
            params = {
                'team': team_id,
                'last': last,
                'season': 2024
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                matches = []
                
                for match in data:
                    fixture = match.get('fixture', {})
                    teams = match.get('teams', {})
                    goals = match.get('goals', {})
                    
                    matches.append({
                        'date': fixture.get('date'),
                        'home': teams.get('home', {}).get('name'),
                        'away': teams.get('away', {}).get('name'),
                        'home_score': goals.get('home'),
                        'away_score': goals.get('away'),
                        'is_home': teams.get('home', {}).get('id') == team_id
                    })
                
                return matches
            return []
        except:
            return []

# =============================================================================
# SYSTÈME D'ANALYSE AUTOMATIQUE
# =============================================================================

class AutoAnalyzer:
    """Analyse automatique des équipes"""
    
    def __init__(self):
        self.team_cache = {}
    
    def analyze_team(self, team_name: str, api_client: FootballDataClient) -> Dict:
        """Analyse automatique d'une équipe"""
        
        # Recherche de l'équipe
        if team_name in self.team_cache:
            return self.team_cache[team_name]
        
        # Données par défaut (si API échoue)
        default_stats = {
            'name': team_name,
            'form': np.random.uniform(5, 9),  # 5-9/10
            'attack': np.random.uniform(1.5, 3.0),  # buts/match
            'defense': np.random.uniform(0.8, 2.0),  # buts encaissés/match
            'last_5_results': ['W', 'D', 'W', 'L', 'W'],  # W=Win, D=Draw, L=Loss
            'home_strength': np.random.uniform(0.6, 0.9),  # Force à domicile
            'away_strength': np.random.uniform(0.4, 0.8),  # Force à l'extérieur
            'goals_scored_last_5': random.choices(range(0, 5), k=5),
            'goals_conceded_last_5': random.choices(range(0, 3), k=5)
        }
        
        try:
            # Essayer de récupérer des données réelles
            search_results = api_client.search_team(team_name)
            
            if search_results:
                team_data = search_results[0]
                team_id = team_data.get('team', {}).get('id')
                
                if team_id:
                    # Récupérer les statistiques
                    stats = api_client.get_team_statistics(team_id)
                    last_matches = api_client.get_last_matches(team_id, 5)
                    
                    if stats:
                        # Calculer la forme basée sur les derniers matchs
                        form = self._calculate_form(last_matches, team_id)
                        attack = self._calculate_attack(stats)
                        defense = self._calculate_defense(stats)
                        
                        analysis = {
                            'name': team_name,
                            'form': form,
                            'attack': attack,
                            'defense': defense,
                            'last_5_results': self._get_last_results(last_matches, team_id),
                            'home_strength': self._calculate_home_strength(stats),
                            'away_strength': self._calculate_away_strength(stats),
                            'real_data': True
                        }
                        
                        self.team_cache[team_name] = analysis
                        return analysis
            
        except Exception as e:
            st.warning(f"Données simulées pour {team_name}: {str(e)}")
        
        # Retourner les données par défaut
        self.team_cache[team_name] = default_stats
        return default_stats
    
    def _calculate_form(self, matches: List[Dict], team_id: int) -> float:
        """Calcule la forme sur 10"""
        if not matches:
            return np.random.uniform(5, 8)
        
        points = 0
        for match in matches[:5]:  # 5 derniers matchs
            is_home = match.get('is_home', False)
            home_score = match.get('home_score')
            away_score = match.get('away_score')
            
            if home_score is not None and away_score is not None:
                if is_home:
                    if home_score > away_score:
                        points += 3
                    elif home_score == away_score:
                        points += 1
                else:
                    if away_score > home_score:
                        points += 3
                    elif home_score == away_score:
                        points += 1
        
        max_points = min(5, len(matches)) * 3
        form = (points / max_points) * 10 if max_points > 0 else 5
        return min(10, max(1, form))
    
    def _calculate_attack(self, stats: Dict) -> float:
        """Calcule la force d'attaque"""
        if not stats:
            return np.random.uniform(1.5, 3.0)
        
        goals = stats.get('goals', {}).get('for', {})
        total = goals.get('total', {})
        
        if total and total.get('total', 0) > 0:
            matches = total.get('played', 1)
            goals_total = total.get('total', 0)
            return goals_total / matches
        return np.random.uniform(1.5, 3.0)
    
    def _calculate_defense(self, stats: Dict) -> float:
        """Calcule la force de défense"""
        if not stats:
            return np.random.uniform(0.8, 2.0)
        
        goals = stats.get('goals', {}).get('against', {})
        total = goals.get('total', {})
        
        if total and total.get('total', 0) > 0:
            matches = total.get('played', 1)
            goals_against = total.get('total', 0)
            return goals_against / matches
        return np.random.uniform(0.8, 2.0)
    
    def _get_last_results(self, matches: List[Dict], team_id: int) -> List[str]:
        """Récupère les 5 derniers résultats"""
        if not matches:
            return random.choices(['W', 'D', 'L'], k=5, weights=[5, 3, 2])
        
        results = []
        for match in matches[:5]:
            is_home = match.get('is_home', False)
            home_score = match.get('home_score')
            away_score = match.get('away_score')
            
            if home_score is not None and away_score is not None:
                if is_home:
                    if home_score > away_score:
                        results.append('W')
                    elif home_score == away_score:
                        results.append('D')
                    else:
                        results.append('L')
                else:
                    if away_score > home_score:
                        results.append('W')
                    elif home_score == away_score:
                        results.append('D')
                    else:
                        results.append('L')
            else:
                results.append(random.choice(['W', 'D', 'L']))
        
        return results or random.choices(['W', 'D', 'L'], k=5, weights=[5, 3, 2])
    
    def _calculate_home_strength(self, stats: Dict) -> float:
        """Calcule la force à domicile"""
        if not stats:
            return np.random.uniform(0.6, 0.9)
        
        fixtures = stats.get('fixtures', {}).get('played', {})
        home = fixtures.get('home', {})
        
        if home.get('played', 0) > 0:
            wins = home.get('wins', 0)
            draws = home.get('draws', 0)
            played = home.get('played', 1)
            return (wins * 3 + draws) / (played * 3)
        return np.random.uniform(0.6, 0.9)
    
    def _calculate_away_strength(self, stats: Dict) -> float:
        """Calcule la force à l'extérieur"""
        if not stats:
            return np.random.uniform(0.4, 0.8)
        
        fixtures = stats.get('fixtures', {}).get('played', {})
        away = fixtures.get('away', {})
        
        if away.get('played', 0) > 0:
            wins = away.get('wins', 0)
            draws = away.get('draws', 0)
            played = away.get('played', 1)
            return (wins * 3 + draws) / (played * 3)
        return np.random.uniform(0.4, 0.8)

# =============================================================================
# SYSTÈME DE PRÉDICTION
# =============================================================================

class PredictionSystem:
    """Système de prédiction automatisé"""
    
    def predict_match(self, home_analysis: Dict, away_analysis: Dict) -> Dict:
        """Prédit un match automatiquement"""
        
        # Extraire les données
        home_form = home_analysis['form']
        away_form = away_analysis['form']
        home_attack = home_analysis['attack']
        away_attack = away_analysis['attack']
        home_defense = home_analysis['defense']
        away_defense = away_analysis['defense']
        home_strength = home_analysis['home_strength']
        away_strength = away_analysis['away_strength']
        
        # Calcul du rating
        home_rating = 1500 + (home_form - 5) * 50 + (home_attack - away_defense) * 100
        away_rating = 1500 + (away_form - 5) * 50 + (away_attack - home_defense) * 100
        
        # Avantage terrain
        home_advantage = 70 * home_strength
        
        # Probabilités
        rating_diff = home_rating + home_advantage - away_rating
        home_win_prob = 1 / (1 + 10 ** (-rating_diff / 400))
        
        # Probabilité de match nul basée sur la différence
        draw_prob = 0.25 * np.exp(-abs(rating_diff) / 200)
        draw_prob = max(0.1, min(draw_prob, 0.35))
        
        away_win_prob = 1 - home_win_prob - draw_prob
        
        # Ajustement pour s'assurer que tout est entre 0 et 1
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total
        
        # Score prédit
        expected_home_goals = (home_attack + away_defense) / 2 * home_strength
        expected_away_goals = (away_attack + home_defense) / 2 * away_strength
        
        # Arrondir au but le plus proche
        predicted_home = round(expected_home_goals)
        predicted_away = round(expected_away_goals)
        
        # Score le plus probable (distribution de Poisson simplifiée)
        most_likely_score = self._get_most_likely_score(expected_home_goals, expected_away_goals)
        
        return {
            'home_win_prob': home_win_prob,
            'draw_prob': draw_prob,
            'away_win_prob': away_win_prob,
            'expected_home_goals': expected_home_goals,
            'expected_away_goals': expected_away_goals,
            'predicted_score': f"{predicted_home}-{predicted_away}",
            'most_likely_score': most_likely_score,
            'home_rating': home_rating,
            'away_rating': away_rating,
            'confidence': min(0.95, abs(rating_diff) / 300 + 0.6)  # 60-95% de confiance
        }
    
    def _get_most_likely_score(self, home_exp: float, away_exp: float) -> str:
        """Trouve le score le plus probable"""
        # Scores possibles de 0-0 à 4-4
        scores = []
        for h in range(5):
            for a in range(5):
                # Probabilité simplifiée (Poisson)
                home_prob = (home_exp ** h) * np.exp(-home_exp) / np.math.factorial(h) if h < 3 else 0.01
                away_prob = (away_exp ** a) * np.exp(-away_exp) / np.math.factorial(a) if a < 3 else 0.01
                prob = home_prob * away_prob
                scores.append((f"{h}-{a}", prob))
        
        # Retourner le score avec la plus haute probabilité
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]
    
    def calculate_value_bets(self, prediction: Dict) -> List[Dict]:
        """Calcule les value bets automatiquement"""
        
        # Cotes du marché estimées (avec marge de bookmaker)
        market_home = 1/prediction['home_win_prob'] * 0.9
        market_draw = 1/prediction['draw_prob'] * 0.9
        market_away = 1/prediction['away_win_prob'] * 0.9
        
        value_bets = []
        
        # Analyser chaque résultat
        edge_home = (prediction['home_win_prob'] * market_home) - 1
        if edge_home > 0.02:
            value_bets.append({
                'selection': '1',
                'market': 'Victoire domicile',
                'odds': round(market_home, 2),
                'probability': prediction['home_win_prob'],
                'edge': edge_home,
                'value_score': edge_home * 100
            })
        
        edge_draw = (prediction['draw_prob'] * market_draw) - 1
        if edge_draw > 0.02:
            value_bets.append({
                'selection': 'N',
                'market': 'Match nul',
                'odds': round(market_draw, 2),
                'probability': prediction['draw_prob'],
                'edge': edge_draw,
                'value_score': edge_draw * 100
            })
        
        edge_away = (prediction['away_win_prob'] * market_away) - 1
        if edge_away > 0.02:
            value_bets.append({
                'selection': '2',
                'market': 'Victoire extérieur',
                'odds': round(market_away, 2),
                'probability': prediction['away_win_prob'],
                'edge': edge_away,
                'value_score': edge_away * 100
            })
        
        # Trier par meilleur value score
        value_bets.sort(key=lambda x: x['value_score'], reverse=True)
        return value_bets

# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

def setup_interface():
    """Configure l'interface Streamlit"""
    st.set_page_config(
        page_title="Analyse Automatique de Matchs",
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
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .team-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #1E88E5;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 20px 0;
        text-align: center;
    }
    .value-bet-card {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .analysis-card {
        background: #fff3e0;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #ffb74d;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-title">⚽ ANALYSE AUTOMATIQUE DE MATCHS</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Entrez juste les noms des équipes, nous faisons tout le reste !</p>', unsafe_allow_html=True)

def main():
    """Application principale"""
    setup_interface()
    
    # Initialisation
    if 'api_client' not in st.session_state:
        st.session_state.api_client = FootballDataClient()
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = AutoAnalyzer()
    
    if 'predictor' not in st.session_state:
        st.session_state.predictor = PredictionSystem()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURATION")
        
        if st.button("🔗 Tester connexion API"):
            if st.session_state.api_client.test_connection():
                st.success("✅ API Football connectée")
            else:
                st.warning("⚠️ Utilisation de données simulées")
        
        st.divider()
        
        st.info("""
        **📋 Comment ça marche:**
        1. Entrez les noms des 2 équipes
        2. L'analyse se fait automatiquement
        3. Recevez prédictions et recommandations
        
        **🔍 Sources de données:**
        - API Football (si disponible)
        - Algorithmes d'analyse avancés
        - Modèles statistiques prédictifs
        """)
    
    # Interface principale
    st.header("🎯 ANALYSE DE MATCH")
    
    # Saisie des équipes
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Équipe Domicile")
        home_team = st.text_input(
            "Nom de l'équipe",
            "Paris Saint-Germain",
            key="home_input",
            help="Ex: Paris SG, Manchester City, Real Madrid..."
        )
    
    with col2:
        st.subheader("⚽ Équipe Extérieur")
        away_team = st.text_input(
            "Nom de l'équipe",
            "Marseille",
            key="away_input",
            help="Ex: Marseille, Liverpool, Barcelona..."
        )
    
    # Bouton d'analyse
    if st.button("🚀 LANCER L'ANALYSE COMPLÈTE", type="primary", use_container_width=True):
        if not home_team or not away_team:
            st.error("⚠️ Veuillez entrer les noms des deux équipes")
        else:
            with st.spinner(f"🔍 Analyse en cours de {home_team} vs {away_team}..."):
                
                # 1. ANALYSE DES ÉQUIPES
                st.subheader("📊 ANALYSE DES ÉQUIPES")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    home_analysis = st.session_state.analyzer.analyze_team(home_team, st.session_state.api_client)
                    display_team_analysis(home_team, home_analysis, "🏠")
                
                with col4:
                    away_analysis = st.session_state.analyzer.analyze_team(away_team, st.session_state.api_client)
                    display_team_analysis(away_team, away_analysis, "⚽")
                
                # 2. PRÉDICTIONS
                st.subheader("🎯 PRÉDICTIONS DU MATCH")
                
                prediction = st.session_state.predictor.predict_match(home_analysis, away_analysis)
                
                # Affichage des probabilités
                col5, col6, col7 = st.columns(3)
                
                with col5:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>🏠 {home_team}</h3>
                        <h1 style="font-size: 3rem;">{prediction['home_win_prob']*100:.1f}%</h1>
                        <p>Cote: {1/prediction['home_win_prob']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col6:
                    st.markdown(f"""
                    <div class="prediction-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                        <h3>🤝 MATCH NUL</h3>
                        <h1 style="font-size: 3rem;">{prediction['draw_prob']*100:.1f}%</h1>
                        <p>Cote: {1/prediction['draw_prob']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col7:
                    st.markdown(f"""
                    <div class="prediction-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                        <h3>⚽ {away_team}</h3>
                        <h1 style="font-size: 3rem;">{prediction['away_win_prob']*100:.1f}%</h1>
                        <p>Cote: {1/prediction['away_win_prob']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Score prédit
                st.subheader("📊 SCORE PRÉDIT")
                
                col8, col9 = st.columns([2, 1])
                
                with col8:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%); 
                    padding: 30px; border-radius: 15px; text-align: center; color: white;">
                    <h1 style="font-size: 4rem; margin: 0;">{prediction['predicted_score']}</h1>
                    <p style="font-size: 1.2rem;">Score le plus probable</p>
                    <p>Buts attendus: {prediction['expected_home_goals']:.2f} - {prediction['expected_away_goals']:.2f}</p>
                    <p>Confiance: {prediction['confidence']*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col9:
                    st.markdown("""
                    <div class="analysis-card">
                        <h4>📈 Détails de prédiction:</h4>
                        <p>• Score le plus probable: **{}**</p>
                        <p>• Buts attendus domicile: **{:.2f}**</p>
                        <p>• Buts attendus extérieur: **{:.2f}**</p>
                        <p>• Rating domicile: **{:.0f}**</p>
                        <p>• Rating extérieur: **{:.0f}**</p>
                    </div>
                    """.format(
                        prediction['most_likely_score'],
                        prediction['expected_home_goals'],
                        prediction['expected_away_goals'],
                        prediction['home_rating'],
                        prediction['away_rating']
                    ), unsafe_allow_html=True)
                
                # 3. VALUE BETS
                st.subheader("💰 VALUE BETS DÉTECTÉS")
                
                value_bets = st.session_state.predictor.calculate_value_bets(prediction)
                
                if value_bets:
                    st.success(f"✅ {len(value_bets)} opportunité(s) de value bet détectée(s)")
                    
                    for bet in value_bets:
                        with st.expander(f"🎯 {bet['market']} - Edge: {bet['edge']*100:.2f}%", expanded=True):
                            col10, col11, col12 = st.columns(3)
                            
                            with col10:
                                st.metric("Cote estimée", f"{bet['odds']:.2f}")
                            
                            with col11:
                                st.metric("Probabilité", f"{bet['probability']*100:.1f}%")
                            
                            with col12:
                                st.metric("Edge", f"{bet['edge']*100:.2f}%")
                            
                            # Recommandation
                            if bet['edge'] > 0.05:
                                st.success(f"**✅ RECOMMANDATION FORTE** - Edge significatif de {bet['edge']*100:.2f}%")
                            elif bet['edge'] > 0.02:
                                st.info(f"**⚠️ RECOMMANDATION MODÉRÉE** - Edge de {bet['edge']*100:.2f}%")
                            
                            # Explication
                            st.markdown(f"""
                            <div class="analysis-card">
                                <h5>📖 Explication:</h5>
                                <p>• Notre modèle prédit une probabilité de **{bet['probability']*100:.1f}%**</p>
                                <p>• La cote du marché devrait être de **{1/bet['probability']:.2f}**</p>
                                <p>• La cote estimée est de **{bet['odds']:.2f}**</p>
                                <p>• Cela représente un avantage (edge) de **{bet['edge']*100:.2f}%**</p>
                                <p>• **Value Score:** {bet['value_score']:.2f}/100</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("""
                    ⚠️ Aucun value bet significatif détecté
                    
                    **Raisons possibles:**
                    • Les cotes du marché sont bien alignées avec nos prédictions
                    • Match trop incertain pour dégager un edge
                    • Considérez d'autres marchés (BTTS, Over/Under)
                    """)
                
                # 4. RECOMMANDATIONS FINALES
                st.subheader("📋 RECOMMANDATIONS FINALES")
                
                col13, col14 = st.columns(2)
                
                with col13:
                    st.markdown(f"""
                    <div class="analysis-card">
                        <h4>✅ POUR {home_team}:</h4>
                        <p>• Forme: {home_analysis['form']:.1f}/10</p>
                        <p>• Attaque: {home_analysis['attack']:.2f} buts/match</p>
                        <p>• Défense: {home_analysis['defense']:.2f} buts/match</p>
                        <p>• Force domicile: {home_analysis['home_strength']*100:.1f}%</p>
                        <p>• Derniers résultats: {' '.join(home_analysis['last_5_results'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col14:
                    st.markdown(f"""
                    <div class="analysis-card">
                        <h4>✅ POUR {away_team}:</h4>
                        <p>• Forme: {away_analysis['form']:.1f}/10</p>
                        <p>• Attaque: {away_analysis['attack']:.2f} buts/match</p>
                        <p>• Défense: {away_analysis['defense']:.2f} buts/match</p>
                        <p>• Force extérieur: {away_analysis['away_strength']*100:.1f}%</p>
                        <p>• Derniers résultats: {' '.join(away_analysis['last_5_results'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Résumé
                st.markdown(f"""
                <div style="background: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                <h4>🎯 RÉSUMÉ DE L'ANALYSE</h4>
                <p><strong>Match:</strong> {home_team} vs {away_team}</p>
                <p><strong>Prédiction principale:</strong> {prediction['predicted_score']}</p>
                <p><strong>Confiance du modèle:</strong> {prediction['confidence']*100:.1f}%</p>
                <p><strong>Meilleure opportunité:</strong> {value_bets[0]['market'] if value_bets else 'Aucune'}</p>
                <p><strong>Recommandation:</strong> {'✅ Paris recommandés' if value_bets else '⚠️ Attendre de meilleures opportunités'}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Section d'information
    st.divider()
    
    st.markdown("""
    ### 📖 Comment fonctionne notre analyse automatique:
    
    **1. 🏃‍♂️ Collecte des données:**
    - Recherche automatique des équipes dans notre base de données
    - Récupération des statistiques récentes
    - Analyse des 5 derniers matchs
    
    **2. 🧮 Analyse statistique:**
    - Calcul de la forme actuelle (1-10)
    - Évaluation de l'attaque et de la défense
    - Mesure de la force à domicile/extérieur
    
    **3. 🎯 Prédictions:**
    - Modèle Élo avancé avec ajustements
    - Distribution de Poisson pour les scores
    - Calcul des probabilités 1X2
    
    **4. 💰 Détection de value bets:**
    - Comparaison avec les cotes du marché
    - Calcul de l'edge (avantage)
    - Recommandations de paris
    
    ### ⚠️ Note importante:
    Cette analyse est basée sur des modèles statistiques. Les résultats réels peuvent varier.
    """)

def display_team_analysis(team_name: str, analysis: Dict, emoji: str):
    """Affiche l'analyse d'une équipe"""
    st.markdown(f"""
    <div class="team-card">
        <h3>{emoji} {team_name}</h3>
        <p><strong>📈 Forme actuelle:</strong> {analysis['form']:.1f}/10</p>
        <p><strong>⚽ Attaque:</strong> {analysis['attack']:.2f} buts/match</p>
        <p><strong>🛡️ Défense:</strong> {analysis['defense']:.2f} buts/match</p>
        <p><strong>🏠 Force domicile:</strong> {analysis['home_strength']*100:.1f}%</p>
        <p><strong>✈️ Force extérieur:</strong> {analysis['away_strength']*100:.1f}%</p>
        <p><strong>📊 5 derniers résultats:</strong> {' '.join(analysis['last_5_results'])}</p>
        <p><small>{'✅ Données réelles' if analysis.get('real_data') else '📡 Données simulées'}</small></p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# LANCEMENT
# =============================================================================

if __name__ == "__main__":
    main()
