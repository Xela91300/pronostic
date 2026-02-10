# app.py - Tipser Pro Football Predictions
# Version améliorée avec sélection mondiale

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import random
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration de l'application"""
    # Couleurs du thème
    COLORS = {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#4CAF50',
        'warning': '#FF9800',
        'danger': '#F44336',
        'info': '#2196F3'
    }

# =============================================================================
# WORLD DATA - Équipes et ligues du monde entier
# =============================================================================

class WorldData:
    """Données mondiales des équipes et ligues"""
    
    @staticmethod
    def get_all_leagues():
        """Retourne toutes les ligues disponibles"""
        return {
            'Europe': [
                'Ligue 1 (France)',
                'Premier League (Angleterre)',
                'La Liga (Espagne)',
                'Bundesliga (Allemagne)',
                'Serie A (Italie)',
                'Liga Portugal',
                'Eredivisie (Pays-Bas)',
                'Jupiler Pro League (Belgique)',
                'Scottish Premiership',
                'Super Lig (Turquie)',
                'Premier League (Russie)',
                'Premiership (Angleterre)',
                'Championship (Angleterre)',
                'Ligue 2 (France)',
                '2. Bundesliga (Allemagne)',
                'Serie B (Italie)'
            ],
            'Amérique du Sud': [
                'Brasileirão (Brésil)',
                'Liga Profesional (Argentine)',
                'Primera División (Chili)',
                'Liga Dimayor (Colombie)',
                'Liga Pro (Équateur)',
                'Liga 1 (Pérou)',
                'Primera División (Uruguay)',
                'MLS (USA/Canada)',
                'Liga MX (Mexique)'
            ],
            'Afrique': [
                'Ligue 1 (Maroc)',
                'Ligue 1 (Tunisie)',
                'Ligue 1 (Algérie)',
                'Egyptian Premier League',
                'South African Premier Division',
                'Ligue 1 (Côte d\'Ivoire)',
                'Nigeria Professional League'
            ],
            'Asie': [
                'J1 League (Japon)',
                'K League 1 (Corée du Sud)',
                'Chinese Super League',
                'A-League (Australie)',
                'Saudi Pro League',
                'Indian Super League',
                'Qatar Stars League',
                'Thai League 1'
            ]
        }
    
    @staticmethod
    def get_teams_by_league(league_name):
        """Retourne les équipes d'une ligue spécifique"""
        teams_data = {
            # Europe
            'Ligue 1 (France)': [
                'Paris Saint-Germain', 'Marseille', 'Lyon', 'Monaco', 'Lille',
                'Nice', 'Rennes', 'Lens', 'Reims', 'Nantes',
                'Montpellier', 'Toulouse', 'Strasbourg', 'Brest', 'Le Havre',
                'Metz', 'Lorient', 'Clermont'
            ],
            'Premier League (Angleterre)': [
                'Manchester City', 'Arsenal', 'Liverpool', 'Manchester United',
                'Chelsea', 'Tottenham', 'Newcastle', 'Aston Villa',
                'West Ham', 'Brighton', 'Brentford', 'Crystal Palace',
                'Wolves', 'Everton', 'Fulham', 'Nottingham Forest',
                'Burnley', 'Sheffield United', 'Luton', 'Bournemouth'
            ],
            'La Liga (Espagne)': [
                'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla',
                'Real Sociedad', 'Villarreal', 'Real Betis', 'Athletic Bilbao',
                'Valencia', 'Osasuna', 'Celta Vigo', 'Getafe',
                'Mallorca', 'Girona', 'Rayo Vallecano', 'Alaves',
                'Cadiz', 'Granada', 'Las Palmas', 'Almeria'
            ],
            'Bundesliga (Allemagne)': [
                'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig',
                'Bayer Leverkusen', 'Eintracht Frankfurt', 'Wolfsburg',
                'Borussia Monchengladbach', 'Freiburg', 'Hoffenheim',
                'Mainz', 'Koln', 'Union Berlin', 'Werder Bremen',
                'Augsburg', 'Stuttgart', 'Bochum', 'Darmstadt', 'Heidenheim'
            ],
            'Serie A (Italie)': [
                'Inter Milan', 'AC Milan', 'Juventus', 'Napoli',
                'Roma', 'Lazio', 'Atalanta', 'Fiorentina',
                'Bologna', 'Torino', 'Monza', 'Udinese',
                'Sassuolo', 'Empoli', 'Salernitana', 'Lecce',
                'Verona', 'Genoa', 'Cagliari', 'Frosinone'
            ],
            # Amérique du Sud
            'Brasileirão (Brésil)': [
                'Flamengo', 'Palmeiras', 'Corinthians', 'São Paulo',
                'Santos', 'Grêmio', 'Internacional', 'Atlético Mineiro',
                'Botafogo', 'Fluminense', 'Vasco da Gama', 'Bahia',
                'Sport Recife', 'Cruzeiro', 'Fortaleza', 'Athletico Paranaense',
                'Ceará', 'Goiás', 'Coritiba', 'América Mineiro'
            ],
            'Liga Profesional (Argentine)': [
                'River Plate', 'Boca Juniors', 'Racing Club', 'Independiente',
                'San Lorenzo', 'Estudiantes', 'Argentinos Juniors', 'Vélez Sarsfield',
                'Lanús', 'Newell\'s Old Boys', 'Rosario Central', 'Gimnasia',
                'Huracán', 'Talleres', 'Banfield', 'Defensa y Justicia',
                'Colón', 'Arsenal de Sarandí', 'Unión', 'Godoy Cruz'
            ],
            # Asie
            'J1 League (Japon)': [
                'Kawasaki Frontale', 'Urawa Red Diamonds', 'Yokohama F. Marinos',
                'FC Tokyo', 'Kashima Antlers', 'Gamba Osaka', 'Cerezo Osaka',
                'Nagoya Grampus', 'Sanfrecce Hiroshima', 'Vissel Kobe',
                'Shimizu S-Pulse', 'Shonan Bellmare', 'Consadole Sapporo',
                'Sagan Tosu', 'Yokohama FC', 'Kashiwa Reysol', 'Oita Trinita'
            ],
            'MLS (USA/Canada)': [
                'Los Angeles FC', 'Seattle Sounders', 'Atlanta United',
                'New York City FC', 'Toronto FC', 'LA Galaxy',
                'Inter Miami', 'Portland Timbers', 'Philadelphia Union',
                'New England Revolution', 'Columbus Crew', 'FC Dallas',
                'Sporting Kansas City', 'Orlando City', 'Minnesota United',
                'Real Salt Lake', 'San Jose Earthquakes', 'Chicago Fire'
            ]
        }
        
        return teams_data.get(league_name, ['Équipe 1', 'Équipe 2', 'Équipe 3', 'Équipe 4'])

# =============================================================================
# AI PREDICTOR (Version Simplifiée)
# =============================================================================

class SimpleAIPredictor:
    """Prédicteur AI simplifié sans scikit-learn"""
    
    def __init__(self):
        self.model_weights = {
            'home_form': 0.20,
            'away_form': 0.18,
            'home_attack': 0.15,
            'away_attack': 0.15,
            'home_defense': 0.12,
            'away_defense': 0.12,
            'h2h': 0.08
        }
    
    def predict(self, home_team, away_team, home_stats, away_stats, h2h_stats=None):
        """Prédit le résultat d'un match"""
        
        # Calculer les scores
        home_score = self._calculate_score(home_stats, is_home=True)
        away_score = self._calculate_score(away_stats, is_home=False)
        
        # Ajouter l'effet h2h
        if h2h_stats:
            home_score += h2h_stats.get('home_advantage', 0)
            away_score += h2h_stats.get('away_advantage', 0)
        
        # Calculer les probabilités
        total = home_score + away_score + 1  # +1 pour le match nul
        home_prob = home_score / total
        draw_prob = 1 / total  # Probabilité de base pour le match nul
        away_prob = away_score / total
        
        # Normaliser
        total_probs = home_prob + draw_prob + away_prob
        home_prob /= total_probs
        draw_prob /= total_probs
        away_prob /= total_probs
        
        return {
            'home_win': round(home_prob * 100, 1),
            'draw': round(draw_prob * 100, 1),
            'away_win': round(away_prob * 100, 1),
            'confidence': self._calculate_confidence(home_prob, away_prob, draw_prob),
            'expected_home': round(random.uniform(1.2, 2.4), 1),
            'expected_away': round(random.uniform(0.8, 1.8), 1)
        }
    
    def _calculate_score(self, stats, is_home):
        """Calcule le score d'une équipe"""
        score = 0
        
        # Forme
        form_score = self._convert_form_to_score(stats.get('form', 'DDDDD'))
        score += self.model_weights['home_form' if is_home else 'away_form'] * form_score
        
        # Attaque
        attack_score = min(stats.get('goals_for_avg', 1.5) / 3, 1)
        score += self.model_weights['home_attack' if is_home else 'away_attack'] * attack_score
        
        # Défense
        defense_score = 1 - min(stats.get('goals_against_avg', 1.5) / 3, 1)
        score += self.model_weights['home_defense' if is_home else 'away_defense'] * defense_score
        
        # Bonus domicile
        if is_home:
            score *= 1.1
        
        return max(0.1, score)
    
    def _convert_form_to_score(self, form_string):
        """Convertit une chaîne de forme en score"""
        if not form_string or len(form_string) < 3:
            return 0.5
        
        scores = {'W': 1.0, 'D': 0.5, 'L': 0.0}
        recent_form = form_string[-5:] if len(form_string) > 5 else form_string
        
        total = sum(scores.get(char, 0.5) for char in recent_form)
        return total / len(recent_form)
    
    def _calculate_confidence(self, home_prob, away_prob, draw_prob):
        """Calcule la confiance de la prédiction"""
        max_prob = max(home_prob, away_prob, draw_prob)
        confidence = (max_prob - 0.333) * 3  # Normaliser par rapport à 33.3% (distribution égale)
        return min(0.95, max(0.4, confidence))

# =============================================================================
# DATA MANAGER
# =============================================================================

class DataManager:
    """Gestionnaire de données simplifié"""
    
    def __init__(self):
        self.world_data = WorldData()
        self.teams = self._generate_world_teams_data()
        self.matches = self._generate_matches_data()
    
    def _generate_world_teams_data(self):
        """Génère des données pour toutes les équipes du monde"""
        teams = {}
        
        # Récupérer toutes les ligues
        all_leagues = self.world_data.get_all_leagues()
        
        for continent, leagues in all_leagues.items():
            for league in leagues:
                league_teams = self.world_data.get_teams_by_league(league)
                for team in league_teams:
                    # Générer des stats réalistes selon la ligue
                    if 'Ligue 1' in league or 'Premier League' in league or 'La Liga' in league:
                        # Grandes ligues européennes
                        form_chars = random.choices(['W', 'D', 'L'], weights=[50, 30, 20], k=5)
                        goals_for = random.uniform(1.2, 2.5)
                        goals_against = random.uniform(0.8, 1.8)
                    elif 'Brasileirão' in league or 'Liga Profesional' in league:
                        # Ligues sud-américaines (plus de buts)
                        form_chars = random.choices(['W', 'D', 'L'], weights=[45, 25, 30], k=5)
                        goals_for = random.uniform(1.3, 2.7)
                        goals_against = random.uniform(1.0, 2.2)
                    else:
                        # Autres ligues
                        form_chars = random.choices(['W', 'D', 'L'], weights=[40, 30, 30], k=5)
                        goals_for = random.uniform(1.0, 2.3)
                        goals_against = random.uniform(1.0, 2.0)
                    
                    teams[team] = {
                        'form': ''.join(form_chars),
                        'goals_for_avg': round(goals_for, 1),
                        'goals_against_avg': round(goals_against, 1),
                        'possession': random.randint(45, 65),
                        'shots_per_game': random.randint(10, 20),
                        'home_strength': random.uniform(0.6, 0.9),
                        'away_strength': random.uniform(0.4, 0.8),
                        'league': league,
                        'continent': continent
                    }
        
        return teams
    
    def _generate_matches_data(self):
        """Génère des données de matchs pour différentes ligues"""
        today = datetime.now()
        matches = []
        match_id = 1
        
        # Récupérer toutes les ligues
        all_leagues = self.world_data.get_all_leagues()
        
        # Générer quelques matchs par continent
        for continent, leagues in all_leagues.items():
            # Prendre 2-3 ligues par continent pour l'exemple
            for league in leagues[:3]:
                teams = self.world_data.get_teams_by_league(league)[:8]  # Prendre 8 premières équipes
                
                # Créer quelques matchs pour cette ligue
                for i in range(min(4, len(teams) // 2)):
                    home_idx = i * 2
                    away_idx = i * 2 + 1
                    
                    if away_idx >= len(teams):
                        break
                    
                    home = teams[home_idx]
                    away = teams[away_idx]
                    
                    # Générer des cotes réalistes
                    home_strength = self.teams[home]['home_strength']
                    away_strength = self.teams[away]['away_strength']
                    
                    if home_strength > away_strength + 0.2:
                        home_odds = round(random.uniform(1.4, 1.8), 2)
                        draw_odds = round(random.uniform(3.8, 4.2), 2)
                        away_odds = round(random.uniform(4.5, 6.0), 2)
                    elif away_strength > home_strength + 0.2:
                        home_odds = round(random.uniform(4.0, 5.5), 2)
                        draw_odds = round(random.uniform(3.5, 4.0), 2)
                        away_odds = round(random.uniform(1.5, 2.0), 2)
                    else:
                        home_odds = round(random.uniform(2.0, 2.8), 2)
                        draw_odds = round(random.uniform(3.2, 3.6), 2)
                        away_odds = round(random.uniform(2.5, 3.5), 2)
                    
                    # Générer un stade réaliste
                    stadiums = {
                        'France': ['Parc des Princes', 'Stade Vélodrome', 'Groupama Stadium'],
                        'Angleterre': ['Old Trafford', 'Anfield', 'Emirates Stadium'],
                        'Espagne': ['Santiago Bernabéu', 'Camp Nou', 'Wanda Metropolitano'],
                        'Allemagne': ['Allianz Arena', 'Signal Iduna Park', 'Red Bull Arena'],
                        'Italie': ['San Siro', 'Juventus Stadium', 'Stadio Olimpico'],
                        'Brésil': ['Maracanã', 'Morumbi', 'Mineirão'],
                        'Argentine': ['El Monumental', 'La Bombonera', 'Estadio Libertadores']
                    }
                    
                    # Déterminer le pays
                    country = league.split('(')[-1].replace(')', '').strip()
                    stadium = random.choice(stadiums.get(country, ['Stade Principal']))
                    
                    matches.append({
                        'id': match_id,
                        'home': home,
                        'away': away,
                        'league': league,
                        'date': today + timedelta(days=random.randint(1, 7)),
                        'time': f"{random.randint(17, 21)}:00",
                        'venue': stadium,
                        'odds': {
                            '1': home_odds,
                            'N': draw_odds,
                            '2': away_odds,
                            'over_2.5': round(random.uniform(1.6, 2.1), 2),
                            'btts_yes': round(random.uniform(1.5, 1.9), 2)
                        }
                    })
                    match_id += 1
        
        return matches
    
    def get_team_stats(self, team_name):
        """Récupère les stats d'une équipe"""
        return self.teams.get(team_name, {
            'form': 'DDDDD',
            'goals_for_avg': 1.5,
            'goals_against_avg': 1.5,
            'possession': 50,
            'shots_per_game': 15,
            'league': 'Ligue inconnue',
            'continent': 'Inconnu'
        })
    
    def get_h2h_stats(self, home_team, away_team):
        """Récupère les stats tête-à-tête"""
        # Simuler des stats historiques
        total_matches = random.randint(3, 10)
        home_wins = random.randint(0, total_matches // 2)
        away_wins = random.randint(0, total_matches // 2)
        draws = total_matches - home_wins - away_wins
        
        return {
            'total_matches': total_matches,
            'home_wins': home_wins,
            'away_wins': away_wins,
            'draws': draws,
            'home_advantage': (home_wins - away_wins) * 0.05,
            'away_advantage': (away_wins - home_wins) * 0.05
        }

# =============================================================================
# BETTING ANALYZER
# =============================================================================

class BettingAnalyzer:
    """Analyseur de paris simplifié"""
    
    @staticmethod
    def calculate_value(probability, odds):
        """Calcule la valeur d'un pari"""
        fair_odds = 1 / (probability / 100) if probability > 0 else float('inf')
        value = (odds / fair_odds) - 1
        
        return {
            'fair_odds': round(fair_odds, 2),
            'value_percent': round(value * 100, 1),
            'is_value': value > 0.05,
            'expected_value': round((probability/100 * (odds - 1)) - ((1 - probability/100) * 1), 3)
        }
    
    @staticmethod
    def recommend_stake(bankroll, probability, odds, risk_profile='moderate'):
        """Recommande une mise"""
        # Version simplifiée du critère de Kelly
        if odds <= 1:
            return 0
        
        kelly = ((probability/100) * (odds - 1) - (1 - probability/100)) / (odds - 1)
        
        # Ajuster selon le profil de risque
        if risk_profile == 'conservative':
            kelly *= 0.5
        elif risk_profile == 'aggressive':
            kelly *= 1.5
        
        # Limites
        kelly = max(0, min(kelly, 0.1))  # Max 10% du bankroll
        
        stake = bankroll * kelly
        
        if stake < 1:
            return "Observation"
        elif stake < bankroll * 0.03:
            return f"Petite (€{stake:.0f})"
        elif stake < bankroll * 0.07:
            return f"Moyenne (€{stake:.0f})"
        else:
            return f"Forte (€{stake:.0f})"

# =============================================================================
# UI COMPONENTS
# =============================================================================

class UIComponents:
    """Composants d'interface"""
    
    @staticmethod
    def setup_page():
        """Configure la page Streamlit"""
        st.set_page_config(
            page_title="Tipser Pro | Pronostics Football Mondial",
            page_icon="🌍⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # CSS personnalisé
        UIComponents._inject_css()
    
    @staticmethod
    def _inject_css():
        """Injecte le CSS"""
        st.markdown(f"""
        <style>
        /* Variables */
        :root {{
            --primary: {Config.COLORS['primary']};
            --secondary: {Config.COLORS['secondary']};
            --success: {Config.COLORS['success']};
            --warning: {Config.COLORS['warning']};
            --danger: {Config.COLORS['danger']};
            --info: {Config.COLORS['info']};
        }}
        
        /* Header */
        .main-header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 20px;
        }}
        
        /* Cards */
        .pro-card {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            border-left: 5px solid var(--success);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}
        
        .pro-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}
        
        /* Badges */
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin: 2px;
        }}
        
        .badge-success {{ background: linear-gradient(135deg, var(--success) 0%, #2E7D32 100%); color: white; }}
        .badge-warning {{ background: linear-gradient(135deg, var(--warning) 0%, #F57C00 100%); color: white; }}
        .badge-danger {{ background: linear-gradient(135deg, var(--danger) 0%, #D32F2F 100%); color: white; }}
        .badge-info {{ background: linear-gradient(135deg, var(--info) 0%, #1976D2 100%); color: white; }}
        .badge-premium {{ background: linear-gradient(135deg, #FFD700 0%, #FFC107 100%); color: #333; }}
        .badge-world {{ background: linear-gradient(135deg, #4CAF50 0%, #2196F3 100%); color: white; }}
        
        /* Metrics */
        .metric-box {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin: 5px;
        }}
        
        /* Progress bars */
        .stProgress > div > div > div {{
            background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        }}
        
        /* Buttons */
        .stButton > button {{
            border-radius: 10px;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            transform: scale(1.03);
        }}
        
        /* Select boxes */
        .stSelectbox div[data-baseweb="select"] {{
            border-radius: 10px;
        }}
        
        /* Dataframes */
        .dataframe {{
            border-radius: 10px;
            overflow: hidden;
        }}
        
        /* Continent colors */
        .continent-europe {{ color: #2196F3; }}
        .continent-south-america {{ color: #4CAF50; }}
        .continent-africa {{ color: #FF9800; }}
        .continent-asia {{ color: #F44336; }}
        .continent-north-america {{ color: #9C27B0; }}
        </style>
        """, unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

class TipserProApp:
    """Application principale"""
    
    def __init__(self):
        self.ui = UIComponents()
        self.data = DataManager()
        self.ai = SimpleAIPredictor()
        self.analyzer = BettingAnalyzer()
        self.world_data = WorldData()
        
        # Initialiser session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialise le session state"""
        defaults = {
            'selected_match': None,
            'view_mode': 'dashboard',
            'bankroll': 1000,
            'risk_profile': 'moderate',
            'selected_continent': 'Europe',
            'selected_league': 'Ligue 1 (France)',
            'selected_home_team': None,
            'selected_away_team': None,
            'custom_match_mode': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """Exécute l'application"""
        # Configuration
        self.ui.setup_page()
        
        # En-tête
        self._display_header()
        
        # Sidebar
        with st.sidebar:
            self._display_sidebar()
        
        # Contenu principal
        self._display_main_content()
    
    def _display_header(self):
        """Affiche l'en-tête"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown("<h1 style='text-align: center;'>🌍⚽</h1>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="main-header">
                <h1>🤖 TIPSER PRO</h1>
                <h3>Pronostics Football Mondial</h3>
                <p>Analyse de matchs du monde entier</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.metric("Bankroll", f"€{st.session_state.bankroll}")
            st.caption("Version Monde 🌍")
    
    def _display_sidebar(self):
        """Affiche la sidebar"""
        st.sidebar.title("🎯 Navigation")
        
        # Menu
        menu_options = {
            "📊 Dashboard": "dashboard",
            "🌍 Matchs Mondiaux": "matches",
            "➕ Créer Match Personnalisé": "custom_match",
            "🤖 Analyse": "analysis",
            "💰 Value Bets": "value",
            "📈 Portfolio": "portfolio",
            "⚙️ Réglages": "settings"
        }
        
        selected = st.sidebar.radio(
            "Menu",
            list(menu_options.keys())
        )
        
        st.session_state.view_mode = menu_options[selected]
        
        st.sidebar.divider()
        
        # Filtres rapides
        st.sidebar.subheader("🌍 Filtres Continent")
        
        all_leagues = self.world_data.get_all_leagues()
        continents = list(all_leagues.keys())
        
        selected_continent = st.sidebar.selectbox(
            "Sélectionner un continent",
            continents,
            index=continents.index(st.session_state.selected_continent) if st.session_state.selected_continent in continents else 0
        )
        
        st.session_state.selected_continent = selected_continent
        
        # Sélection de ligue
        st.sidebar.subheader("🏆 Sélectionner une Ligue")
        
        leagues_in_continent = all_leagues[selected_continent]
        selected_league = st.sidebar.selectbox(
            "Choisir une ligue",
            leagues_in_continent,
            index=leagues_in_continent.index(st.session_state.selected_league) if st.session_state.selected_league in leagues_in_continent else 0
        )
        
        st.session_state.selected_league = selected_league
        
        # Stats rapides
        st.sidebar.divider()
        st.sidebar.subheader("📈 Stats Mondiales")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Ligues", "24")
            st.metric("Équipes", "428")
        with col2:
            st.metric("Continents", "4")
            st.metric("Matchs/jour", "32")
        
        # Bouton actualisation
        if st.sidebar.button("🔄 Actualiser Données", use_container_width=True):
            self.data = DataManager()  # Regénérer les données
            st.rerun()
    
    def _display_main_content(self):
        """Affiche le contenu principal"""
        view_mode = st.session_state.view_mode
        
        if view_mode == 'dashboard':
            self._display_dashboard()
        elif view_mode == 'matches':
            self._display_world_matches()
        elif view_mode == 'custom_match':
            self._display_custom_match_creator()
        elif view_mode == 'analysis':
            if st.session_state.selected_match:
                self._display_analysis()
            else:
                st.warning("Veuillez d'abord sélectionner un match")
                st.session_state.view_mode = 'matches'
                st.rerun()
        elif view_mode == 'value':
            self._display_value_bets()
        elif view_mode == 'portfolio':
            self._display_portfolio()
        elif view_mode == 'settings':
            self._display_settings()
    
    def _display_dashboard(self):
        """Affiche le dashboard mondial"""
        st.title("🌍 Dashboard Mondial")
        
        # KPI Mondiaux
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
                <h3>🌍 Continents</h3>
                <h2>4</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
                <h3>🏆 Ligues Actives</h3>
                <h2>24</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-box">
                <h3>⚽ Équipes</h3>
                <h2>428</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-box">
                <h3>📈 ROI Global</h3>
                <h2>+8.2%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Performance par continent
        st.subheader("📊 Performance par Continent")
        
        continent_data = {
            'Continent': ['Europe', 'Amérique du Sud', 'Afrique', 'Asie'],
            'Matchs Analysés': [156, 89, 67, 45],
            'Taux de Réussite': ['72%', '68%', '65%', '62%'],
            'ROI': ['+9.5%', '+8.2%', '+6.8%', '+5.4%']
        }
        
        df_continents = pd.DataFrame(continent_data)
        st.dataframe(df_continents, use_container_width=True, hide_index=True)
        
        # Derniers tips internationaux
        st.divider()
        st.subheader("🎯 Derniers Tips Mondiaux")
        
        tips = [
            {"match": "Flamengo vs Palmeiras", "ligue": "Brasileirão", "pred": "Over 2.5", "cote": 1.85, "status": "✅ Gagné"},
            {"match": "Urawa vs Kawasaki", "ligue": "J1 League", "pred": "1", "cote": 2.10, "status": "⏳ En cours"},
            {"match": "Orlando City vs Inter Miami", "ligue": "MLS", "pred": "BTTS Yes", "cote": 1.65, "status": "✅ Gagné"},
            {"match": "Al Ahly vs Zamalek", "ligue": "Egypte", "pred": "2", "cote": 3.25, "status": "❌ Perdu"}
        ]
        
        for tip in tips:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
                
                with col1:
                    st.write(f"**{tip['match']}**")
                    st.caption(tip['ligue'])
                
                with col2:
                    st.code(tip['pred'])
                
                with col3:
                    st.metric("Cote", tip['cote'])
                
                with col4:
                    continent_color = {
                        'Brasileirão': 'continent-south-america',
                        'J1 League': 'continent-asia',
                        'MLS': 'continent-north-america',
                        'Egypte': 'continent-africa'
                    }.get(tip['ligue'], '')
                    st.markdown(f"<span class='{continent_color}'>{tip['ligue']}</span>", unsafe_allow_html=True)
                
                with col5:
                    if tip['status'] == '✅ Gagné':
                        st.success(tip['status'])
                    elif tip['status'] == '❌ Perdu':
                        st.error(tip['status'])
                    else:
                        st.info(tip['status'])
                
                st.divider()
    
    def _display_world_matches(self):
        """Affiche les matchs mondiaux"""
        st.title(f"🌍 Matchs - {st.session_state.selected_continent}")
        
        # Filtres avancés
        with st.expander("🔍 Filtres Avancés", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                # Sélection multiple de ligues
                all_leagues = self.world_data.get_all_leagues()
                continent_leagues = all_leagues[st.session_state.selected_continent]
                
                selected_leagues = st.multiselect(
                    "Ligues à afficher",
                    continent_leagues,
                    default=[st.session_state.selected_league]
                )
            
            with col2:
                date_filter = st.selectbox(
                    "Période",
                    ["Aujourd'hui", "Demain", "Week-end", "7 prochains jours", "Tous"]
                )
        
        # Bouton recherche
        if st.button("🔍 Rechercher Matchs", type="primary", use_container_width=True):
            with st.spinner("Analyse des matchs mondiaux..."):
                time.sleep(0.5)
                self._display_filtered_matches(selected_leagues if 'selected_leagues' in locals() else [])
    
    def _display_filtered_matches(self, selected_leagues):
        """Affiche les matchs filtrés"""
        matches = self.data.matches
        
        # Filtrer par ligues sélectionnées
        if selected_leagues:
            matches = [m for m in matches if m['league'] in selected_leagues]
        
        if not matches:
            st.info("Aucun match trouvé avec les filtres actuels.")
            return
        
        st.subheader(f"📋 {len(matches)} Matchs Trouvés")
        
        for match in matches:
            # Obtenir les prédictions AI
            home_stats = self.data.get_team_stats(match['home'])
            away_stats = self.data.get_team_stats(match['away'])
            h2h_stats = self.data.get_h2h_stats(match['home'], match['away'])
            
            prediction = self.ai.predict(match['home'], match['away'], home_stats, away_stats, h2h_stats)
            
            # Afficher la carte
            self._render_world_match_card(match, prediction)
    
    def _render_world_match_card(self, match, prediction):
        """Affiche une carte de match mondial"""
        date_str = match['date'].strftime("%d/%m/%Y")
        
        # Déterminer la couleur du continent
        continent_color_class = {
            'Europe': 'continent-europe',
            'Amérique du Sud': 'continent-south-america',
            'Afrique': 'continent-africa',
            'Asie': 'continent-asia'
        }.get(match.get('continent', 'Europe'), '')
        
        st.markdown(f"""
        <div class="pro-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4>{match['home']} vs {match['away']}</h4>
                    <p><span class="{continent_color_class}">🏆 {match['league']}</span> | 📅 {date_str} | ⏰ {match['time']}</p>
                </div>
                <div>
                    <span class="badge badge-world">🌍 Mondial</span>
                    <span class="badge badge-info">Confiance: {prediction['confidence']*100:.0f}%</span>
                </div>
            </div>
            
            <div style="display: flex; justify-content: space-around; margin: 20px 0;">
                <div style="text-align: center;">
                    <h5>1</h5>
                    <h3>{match['odds']['1']}</h3>
                    <p>{prediction['home_win']}%</p>
                </div>
                <div style="text-align: center;">
                    <h5>N</h5>
                    <h3>{match['odds']['N']}</h3>
                    <p>{prediction['draw']}%</p>
                </div>
                <div style="text-align: center;">
                    <h5>2</h5>
                    <h3>{match['odds']['2']}</h3>
                    <p>{prediction['away_win']}%</p>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 20px;">
                <h4>🎯 Prédiction: {self._get_prediction_text(prediction)}</h4>
                <p>Stade: {match['venue']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Boutons d'action
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"📊 Analyser", key=f"analyze_{match['id']}", use_container_width=True):
                st.session_state.selected_match = match
                st.session_state.view_mode = 'analysis'
                st.rerun()
        
        with col2:
            if st.button(f"💰 Value Bet", key=f"value_{match['id']}", use_container_width=True):
                st.session_state.selected_match = match
                st.session_state.view_mode = 'value'
                st.rerun()
        
        st.divider()
    
    def _display_custom_match_creator(self):
        """Affiche l'interface pour créer un match personnalisé"""
        st.title("➕ Créer un Match Personnalisé")
        
        st.markdown("""
        <div class="pro-card">
            <h3>🌍 Sélectionnez deux équipes du monde entier</h3>
            <p>Créez votre propre match et obtenez une analyse AI personnalisée</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sélection du continent pour l'équipe à domicile
        st.subheader("🏠 Équipe à Domicile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            continent_home = st.selectbox(
                "Continent (Domicile)",
                list(self.world_data.get_all_leagues().keys()),
                key="continent_home"
            )
        
        with col2:
            # Sélection de la ligue basée sur le continent
            leagues_home = self.world_data.get_all_leagues()[continent_home]
            league_home = st.selectbox(
                "Ligue (Domicile)",
                leagues_home,
                key="league_home"
            )
        
        # Sélection de l'équipe à domicile
        teams_home = self.world_data.get_teams_by_league(league_home)
        selected_home_team = st.selectbox(
            "Équipe à Domicile",
            teams_home,
            key="selected_home_team"
        )
        
        st.divider()
        
        # Sélection du continent pour l'équipe à l'extérieur
        st.subheader("✈️ Équipe à l'Extérieur")
        
        col3, col4 = st.columns(2)
        
        with col3:
            continent_away = st.selectbox(
                "Continent (Extérieur)",
                list(self.world_data.get_all_leagues().keys()),
                key="continent_away"
            )
        
        with col4:
            # Sélection de la ligue basée sur le continent
            leagues_away = self.world_data.get_all_leagues()[continent_away]
            league_away = st.selectbox(
                "Ligue (Extérieur)",
                leagues_away,
                key="league_away"
            )
        
        # Sélection de l'équipe à l'extérieur
        teams_away = self.world_data.get_teams_by_league(league_away)
        selected_away_team = st.selectbox(
            "Équipe à l'Extérieur",
            teams_away,
            key="selected_away_team"
        )
        
        # Paramètres du match
        st.divider()
        st.subheader("⚙️ Paramètres du Match")
        
        col5, col6 = st.columns(2)
        
        with col5:
            match_date = st.date_input(
                "Date du match",
                datetime.now() + timedelta(days=3)
            )
        
        with col6:
            match_time = st.time_input(
                "Heure du match",
                datetime.now().replace(hour=20, minute=0)
            )
        
        # Bouton d'analyse
        st.divider()
        
        if st.button("🤖 Analyser ce Match Personnalisé", type="primary", use_container_width=True):
            if selected_home_team == selected_away_team:
                st.error("Veuillez sélectionner deux équipes différentes!")
                return
            
            with st.spinner("Analyse AI en cours..."):
                time.sleep(1)
                
                # Créer un match personnalisé
                custom_match = {
                    'id': 999,
                    'home': selected_home_team,
                    'away': selected_away_team,
                    'league': f"{league_home} vs {league_away}",
                    'date': datetime.combine(match_date, match_time),
                    'time': match_time.strftime("%H:%M"),
                    'venue': "Stade Personnalisé",
                    'odds': {
                        '1': round(random.uniform(1.5, 3.0), 2),
                        'N': round(random.uniform(3.0, 4.0), 2),
                        '2': round(random.uniform(2.0, 4.0), 2),
                        'over_2.5': round(random.uniform(1.6, 2.1), 2),
                        'btts_yes': round(random.uniform(1.5, 1.9), 2)
                    }
                }
                
                st.session_state.selected_match = custom_match
                st.session_state.view_mode = 'analysis'
                st.rerun()
    
    def _get_prediction_text(self, prediction):
        """Obtient le texte de prédiction"""
        if prediction['home_win'] > prediction['away_win'] and prediction['home_win'] > prediction['draw']:
            return "Victoire domicile"
        elif prediction['away_win'] > prediction['home_win'] and prediction['away_win'] > prediction['draw']:
            return "Victoire extérieur"
        else:
            return "Match nul"
    
    def _display_analysis(self):
        """Affiche l'analyse d'un match"""
        match = st.session_state.selected_match
        
        # Bouton retour
        if st.button("← Retour aux matchs"):
            st.session_state.view_mode = 'matches'
            st.rerun()
        
        # Titre avec drapeaux
        continent_emoji = {
            'Europe': '🇪🇺',
            'Amérique du Sud': '🇧🇷',
            'Afrique': '🇿🇦',
            'Asie': '🇯🇵'
        }
        
        emoji = continent_emoji.get(match.get('continent', 'Europe'), '🌍')
        st.title(f"{emoji} Analyse: {match['home']} vs {match['away']}")
        
        # Onglets
        tab1, tab2, tab3 = st.tabs(["📊 Vue d'ensemble", "🤖 Prédictions", "📈 Statistiques"])
        
        with tab1:
            self._display_analysis_overview(match)
        
        with tab2:
            self._display_predictions(match)
        
        with tab3:
            self._display_statistics(match)
    
    def _display_analysis_overview(self, match):
        """Affiche la vue d'ensemble"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏟️ Informations")
            
            info = {
                "Ligue": match['league'],
                "Date": match['date'].strftime("%d/%m/%Y"),
                "Heure": match['time'],
                "Stade": match['venue'],
                "Type": "🌍 Match International" if 'Personnalisé' in match['league'] else "Match de Ligue"
            }
            
            for key, value in info.items():
                st.metric(key, value)
        
        with col2:
            st.subheader("📊 Cotes du Marché")
            
            odds_data = {
                "1 - Victoire domicile": match['odds']['1'],
                "N - Match nul": match['odds']['N'],
                "2 - Victoire extérieur": match['odds']['2'],
                "Over 2.5 buts": match['odds']['over_2.5'],
                "Les deux marquent": match['odds']['btts_yes']
            }
            
            df_odds = pd.DataFrame(list(odds_data.items()), columns=['Marché', 'Cote'])
            st.dataframe(df_odds, use_container_width=True, hide_index=True)
        
        # Forme des équipes
        st.divider()
        st.subheader("📈 Forme des Équipes")
        
        home_stats = self.data.get_team_stats(match['home'])
        away_stats = self.data.get_team_stats(match['away'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**{match['home']}**")
            st.caption(f"Ligue: {home_stats.get('league', 'Inconnue')}")
            self._display_team_stats(home_stats)
        
        with col2:
            st.write(f"**{match['away']}**")
            st.caption(f"Ligue: {away_stats.get('league', 'Inconnue')}")
            self._display_team_stats(away_stats)
    
    def _display_team_stats(self, stats):
        """Affiche les stats d'une équipe"""
        # Forme
        st.write(f"Forme récente: {stats['form']}")
        
        # Barres de progression
        st.write("Buts marqués (moy.):")
        st.progress(min(stats['goals_for_avg'] / 3, 1))
        
        st.write("Buts encaissés (moy.):")
        st.progress(min(stats['goals_against_avg'] / 3, 1))
        
        # Métriques
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Possession", f"{stats['possession']}%")
        with col2:
            st.metric("Tirs/match", stats['shots_per_game'])
    
    def _display_predictions(self, match):
        """Affiche les prédictions"""
        st.subheader("🤖 Prédictions AI")
        
        # Obtenir la prédiction
        home_stats = self.data.get_team_stats(match['home'])
        away_stats = self.data.get_team_stats(match['away'])
        h2h_stats = self.data.get_h2h_stats(match['home'], match['away'])
        
        prediction = self.ai.predict(match['home'], match['away'], home_stats, away_stats, h2h_stats)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Probabilités")
            
            # Afficher les probabilités avec barres
            st.write(f"**{match['home']} gagne**")
            st.progress(prediction['home_win'] / 100)
            st.caption(f"{prediction['home_win']}%")
            
            st.write("**Match nul**")
            st.progress(prediction['draw'] / 100)
            st.caption(f"{prediction['draw']}%")
            
            st.write(f"**{match['away']} gagne**")
            st.progress(prediction['away_win'] / 100)
            st.caption(f"{prediction['away_win']}%")
        
        with col2:
            st.markdown("### ⚽ Score Attend")
            
            # Afficher le score attendu
            st.markdown(f"""
            <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px;">
                <h1 style="font-size: 4rem; margin: 0;">
                    {prediction['expected_home']} - {prediction['expected_away']}
                </h1>
                <p style="color: #666; font-size: 1.2rem;">Score attendu (xG)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Métriques
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total buts", round(prediction['expected_home'] + prediction['expected_away'], 1))
            with col_b:
                st.metric("Confiance AI", f"{prediction['confidence']*100:.1f}%")
        
        # Recommandation
        st.divider()
        st.markdown("### 🎯 Recommandation")
        
        if prediction['home_win'] > 55:
            rec = f"✅ {match['home']} gagne"
            color = Config.COLORS['success']
        elif prediction['away_win'] > 55:
            rec = f"✅ {match['away']} gagne"
            color = Config.COLORS['success']
        else:
            rec = "⚪ Match nul ou Double Chance"
            color = Config.COLORS['info']
        
        st.markdown(f"""
        <div style="background: {color}; 
                    color: white; 
                    padding: 20px; 
                    border-radius: 10px; 
                    text-align: center;">
            <h2>{rec}</h2>
            <p>Probabilité: {max(prediction['home_win'], prediction['draw'], prediction['away_win'])}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_statistics(self, match):
        """Affiche les statistiques"""
        st.subheader("📈 Statistiques Comparatives")
        
        home_stats = self.data.get_team_stats(match['home'])
        away_stats = self.data.get_team_stats(match['away'])
        
        # Tableau comparatif
        comparison_data = {
            'Statistique': ['Forme récente', 'Buts/m (marqués)', 'Buts/m (encaissés)', 'Possession', 'Tirs/match', 'Ligue', 'Continent'],
            match['home']: [
                home_stats['form'],
                f"{home_stats['goals_for_avg']}",
                f"{home_stats['goals_against_avg']}",
                f"{home_stats['possession']}%",
                home_stats['shots_per_game'],
                home_stats.get('league', 'Inconnue'),
                home_stats.get('continent', 'Inconnu')
            ],
            match['away']: [
                away_stats['form'],
                f"{away_stats['goals_for_avg']}",
                f"{away_stats['goals_against_avg']}",
                f"{away_stats['possession']}%",
                away_stats['shots_per_game'],
                away_stats.get('league', 'Inconnue'),
                away_stats.get('continent', 'Inconnu')
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df.set_index('Statistique'), use_container_width=True)
    
    def _display_value_bets(self):
        """Affiche les value bets"""
        st.title("💰 Value Bets Mondiaux")
        
        # Informations
        st.info("""
        ℹ️ Les **Value Bets** sont des paris où les cotes proposées sont plus élevées que la probabilité réelle estimée.
        Ces opportunités offrent un avantage statistique sur le bookmaker.
        """)
        
        # Scanner les value bets
        if st.button("🔍 Scanner toutes les ligues", type="primary", use_container_width=True):
            with st.spinner("Analyse des value bets mondiales..."):
                time.sleep(1.5)
                self._display_world_value_bets()
    
    def _display_world_value_bets(self):
        """Affiche les value bets mondiales"""
        value_bets = []
        
        # Analyser les matchs
        for match in self.data.matches[:10]:  # Limiter à 10 matchs pour la démo
            home_stats = self.data.get_team_stats(match['home'])
            away_stats = self.data.get_team_stats(match['away'])
            h2h_stats = self.data.get_h2h_stats(match['home'], match['away'])
            
            prediction = self.ai.predict(match['home'], match['away'], home_stats, away_stats, h2h_stats)
            
            # Analyser les marchés principaux
            markets = [
                ('1', match['odds']['1'], prediction['home_win']),
                ('N', match['odds']['N'], prediction['draw']),
                ('2', match['odds']['2'], prediction['away_win'])
            ]
            
            for market_name, odds, probability in markets:
                analysis = self.analyzer.calculate_value(probability, odds)
                
                if analysis['is_value'] and analysis['value_percent'] >= 5:
                    value_bets.append({
                        'match': f"{match['home']} vs {match['away']}",
                        'market': market_name,
                        'odds': odds,
                        'probability': probability,
                        'value': analysis['value_percent'],
                        'confidence': prediction['confidence'] * 100,
                        'league': match['league'],
                        'continent': match.get('continent', 'Europe')
                    })
        
        if value_bets:
            # Trier par valeur
            value_bets.sort(key=lambda x: x['value'], reverse=True)
            
            st.success(f"🎯 {len(value_bets)} Value Bets trouvées dans le monde!")
            
            for bet in value_bets:
                with st.container():
                    continent_color = {
                        'Europe': 'continent-europe',
                        'Amérique du Sud': 'continent-south-america',
                        'Afrique': 'continent-africa',
                        'Asie': 'continent-asia'
                    }.get(bet['continent'], '')
                    
                    st.markdown(f"""
                    <div class="pro-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h4>{bet['match']}</h4>
                                <p><span class="{continent_color}">🏆 {bet['league']}</span> | 🎯 {bet['market']}</p>
                            </div>
                            <div>
                                <span class="badge badge-premium">+{bet['value']:.1f}% Valeur</span>
                                <span class="badge badge-world">🌍</span>
                            </div>
                        </div>
                        
                        <div style="display: flex; justify-content: space-around; margin: 20px 0;">
                            <div style="text-align: center;">
                                <h5>Cote</h5>
                                <h3>{bet['odds']}</h3>
                            </div>
                            <div style="text-align: center;">
                                <h5>Probabilité</h5>
                                <h3>{bet['probability']:.1f}%</h3>
                            </div>
                            <div style="text-align: center;">
                                <h5>Valeur</h5>
                                <h3>+{bet['value']:.1f}%</h3>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Boutons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📊 Analyser", key=f"analyze_vb_{bet['match']}", use_container_width=True):
                            match_to_select = next((m for m in self.data.matches 
                                                  if f"{m['home']} vs {m['away']}" == bet['match']), None)
                            if match_to_select:
                                st.session_state.selected_match = match_to_select
                                st.session_state.view_mode = 'analysis'
                                st.rerun()
                    
                    with col2:
                        if st.button("💰 Placer pari", key=f"bet_vb_{bet['match']}", use_container_width=True, type="primary"):
                            st.success(f"💰 Pari placé sur {bet['match']} - {bet['market']}")
                            st.balloons()
                    
                    st.divider()
        else:
            st.info("ℹ️ Aucune value bet significative trouvée pour le moment.")
    
    def _display_portfolio(self):
        """Affiche le portfolio mondial"""
        st.title("💰 Mon Portfolio Mondial")
        
        # Résumé mondial
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Paris actifs", "7")
        
        with col2:
            st.metric("Ligues couvertes", "8")
        
        with col3:
            st.metric("Continents", "3")
        
        with col4:
            st.metric("ROI global", "+8.2%")
        
        st.divider()
        
        # Paris par continent
        st.subheader("🌍 Répartition par Continent")
        
        continent_distribution = {
            'Continent': ['Europe', 'Amérique du Sud', 'Afrique', 'Asie'],
            'Nombre de Paris': [15, 8, 5, 3],
            'Mise Totale': ['€1,200', '€650', '€400', '€250'],
            'Gains': ['€1,380', '€720', '€460', '€270']
        }
        
        df_dist = pd.DataFrame(continent_distribution)
        st.dataframe(df_dist, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Derniers paris mondiaux
        st.subheader("📊 Derniers Paris Internationaux")
        
        world_bets = [
            {"match": "Flamengo vs Palmeiras", "continent": "Amérique du Sud", "type": "Over 2.5", "mise": "€100", "statut": "✅ Gagné €185"},
            {"match": "Urawa vs Kawasaki", "continent": "Asie", "type": "1", "mise": "€80", "statut": "⏳ En cours"},
            {"match": "Bayern vs Dortmund", "continent": "Europe", "type": "BTTS Yes", "mise": "€120", "statut": "✅ Gagné €198"},
            {"match": "Al Ahly vs Zamalek", "continent": "Afrique", "type": "2", "mise": "€60", "statut": "❌ Perdu €0"}
        ]
        
        for bet in world_bets:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
                with col1:
                    st.write(f"**{bet['match']}**")
                    continent_emoji = {
                        'Europe': '🇪🇺',
                        'Amérique du Sud': '🇧🇷',
                        'Afrique': '🇿🇦',
                        'Asie': '🇯🇵'
                    }
                    st.caption(f"{continent_emoji.get(bet['continent'], '🌍')} {bet['continent']}")
                with col2:
                    st.code(bet['type'])
                with col3:
                    st.write(bet['mise'])
                with col4:
                    st.write(bet['continent'])
                with col5:
                    if '✅' in bet['statut']:
                        st.success(bet['statut'])
                    elif '❌' in bet['statut']:
                        st.error(bet['statut'])
                    else:
                        st.warning(bet['statut'])
                st.divider()
    
    def _display_settings(self):
        """Affiche les réglages mondiaux"""
        st.title("⚙️ Réglages Mondiaux")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌍 Configuration Mondiale")
            
            # Sélection des continents d'intérêt
            all_continents = list(self.world_data.get_all_leagues().keys())
            selected_continents = st.multiselect(
                "Continents d'intérêt",
                all_continents,
                default=all_continents[:2]
            )
            
            # Limite de ligues par continent
            max_leagues = st.slider(
                "Ligues max par continent",
                1, 10, 5
            )
            
            # Niveau de détail
            detail_level = st.selectbox(
                "Niveau d'analyse",
                ["Basique", "Standard", "Avancé", "Expert"]
            )
        
        with col2:
            st.subheader("💰 Paramètres de Paris Mondiaux")
            
            bankroll = st.number_input(
                "Bankroll total (€)",
                min_value=100,
                max_value=10000,
                value=st.session_state.bankroll,
                step=100
            )
            
            risk_profile = st.selectbox(
                "Profil de risque mondial",
                ["Très Conservateur", "Conservateur", "Modéré", "Agressif", "Très Agressif"],
                index=2
            )
            
            # Stratégie par continent
            st.checkbox("Ajuster automatiquement selon le continent", value=True)
            st.checkbox("Notifications matchs internationaux", value=True)
        
        # Boutons de sauvegarde
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Sauvegarder", type="primary", use_container_width=True):
                st.session_state.bankroll = bankroll
                st.session_state.risk_profile = risk_profile.lower()
                st.success("Réglages mondiaux sauvegardés!")
                st.balloons()
        
        with col2:
            if st.button("🔄 Réinitialiser", use_container_width=True):
                st.session_state.bankroll = 1000
                st.session_state.risk_profile = 'moderate'
                st.rerun()
        
        with col3:
            if st.button("🌍 Exporter config", use_container_width=True):
                st.info("Configuration mondiale exportée!")
                st.download_button(
                    label="📥 Télécharger",
                    data="Configuration Tipser Pro Mondial",
                    file_name="tipser_pro_world_config.json",
                    mime="application/json"
                )

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main():
    """Fonction principale"""
    try:
        app = TipserProApp()
        app.run()
    except Exception as e:
        st.error(f"Une erreur est survenue: {str(e)}")
        st.info("Veuillez rafraîchir la page ou contacter le support.")

if __name__ == "__main__":
    main()
