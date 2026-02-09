# app.py - Tipser Pro Football Predictions
# Version ultra légère et fonctionnelle

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
        self.teams = self._generate_teams_data()
        self.matches = self._generate_matches_data()
    
    def _generate_teams_data(self):
        """Génère des données d'équipes"""
        teams = {}
        team_list = [
            'Paris SG', 'Marseille', 'Lyon', 'Monaco', 'Lille', 'Nice',
            'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla',
            'Manchester City', 'Liverpool', 'Arsenal', 'Manchester United',
            'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen',
            'Juventus', 'Inter Milan', 'AC Milan', 'Napoli'
        ]
        
        for team in team_list:
            # Générer une forme réaliste (plus de victoires pour les meilleures équipes)
            if team in ['Paris SG', 'Real Madrid', 'Manchester City', 'Bayern Munich']:
                form_chars = random.choices(['W', 'D', 'L'], weights=[60, 25, 15], k=5)
            elif team in ['Marseille', 'Barcelona', 'Liverpool', 'Borussia Dortmund']:
                form_chars = random.choices(['W', 'D', 'L'], weights=[50, 30, 20], k=5)
            else:
                form_chars = random.choices(['W', 'D', 'L'], weights=[40, 30, 30], k=5)
            
            teams[team] = {
                'form': ''.join(form_chars),
                'goals_for_avg': round(random.uniform(1.0, 2.5), 1),
                'goals_against_avg': round(random.uniform(0.8, 2.0), 1),
                'possession': random.randint(45, 65),
                'shots_per_game': random.randint(10, 20),
                'home_strength': random.uniform(0.6, 0.9),
                'away_strength': random.uniform(0.4, 0.8)
            }
        
        return teams
    
    def _generate_matches_data(self):
        """Génère des données de matchs"""
        today = datetime.now()
        matches = []
        
        # Matchs de Ligue 1
        ligue1_matches = [
            ('Paris SG', 'Marseille', 'Ligue 1', 'Parc des Princes'),
            ('Lyon', 'Monaco', 'Ligue 1', 'Groupama Stadium'),
            ('Lille', 'Nice', 'Ligue 1', 'Stade Pierre-Mauroy'),
            ('Marseille', 'Lyon', 'Ligue 1', 'Orange Vélodrome'),
            ('Monaco', 'Paris SG', 'Ligue 1', 'Stade Louis-II')
        ]
        
        # Matchs de La Liga
        laliga_matches = [
            ('Real Madrid', 'Barcelona', 'La Liga', 'Santiago Bernabéu'),
            ('Atletico Madrid', 'Sevilla', 'La Liga', 'Wanda Metropolitano'),
            ('Barcelona', 'Atletico Madrid', 'La Liga', 'Camp Nou')
        ]
        
        # Matchs de Premier League
        pl_matches = [
            ('Manchester City', 'Liverpool', 'Premier League', 'Etihad Stadium'),
            ('Arsenal', 'Manchester United', 'Premier League', 'Emirates Stadium'),
            ('Liverpool', 'Arsenal', 'Premier League', 'Anfield')
        ]
        
        all_matches = ligue1_matches + laliga_matches + pl_matches
        
        for idx, (home, away, league, venue) in enumerate(all_matches[:8], 1):
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
            
            matches.append({
                'id': idx,
                'home': home,
                'away': away,
                'league': league,
                'date': today + timedelta(days=random.randint(1, 7)),
                'time': f"{random.randint(17, 21)}:00",
                'venue': venue,
                'odds': {
                    '1': home_odds,
                    'N': draw_odds,
                    '2': away_odds,
                    'over_2.5': round(random.uniform(1.6, 2.1), 2),
                    'btts_yes': round(random.uniform(1.5, 1.9), 2)
                }
            })
        
        return matches
    
    def get_team_stats(self, team_name):
        """Récupère les stats d'une équipe"""
        return self.teams.get(team_name, {
            'form': 'DDDDD',
            'goals_for_avg': 1.5,
            'goals_against_avg': 1.5,
            'possession': 50,
            'shots_per_game': 15
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
            page_title="Tipser Pro | Pronostics Football",
            page_icon="⚽",
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
        
        /* Dataframes */
        .dataframe {{
            border-radius: 10px;
            overflow: hidden;
        }}
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
        
        # Initialiser session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialise le session state"""
        defaults = {
            'selected_match': None,
            'view_mode': 'dashboard',
            'bankroll': 1000,
            'risk_profile': 'moderate'
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
            st.markdown("<h1 style='text-align: center;'>⚽</h1>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="main-header">
                <h1>🤖 TIPSER PRO</h1>
                <h3>Pronostics Football Intelligents</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.metric("Bankroll", f"€{st.session_state.bankroll}")
            st.caption("Version Pro")
    
    def _display_sidebar(self):
        """Affiche la sidebar"""
        st.sidebar.title("🎯 Navigation")
        
        # Menu
        menu_options = {
            "📊 Dashboard": "dashboard",
            "🔍 Matchs": "matches",
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
        st.sidebar.subheader("🔎 Filtres")
        
        st.sidebar.multiselect(
            "Ligues",
            ["Ligue 1", "Premier League", "La Liga", "Bundesliga", "Serie A"],
            default=["Ligue 1", "Premier League"],
            key="league_filter"
        )
        
        min_confidence = st.sidebar.slider(
            "Confiance min",
            50, 95, 65
        )
        
        # Stats rapides
        st.sidebar.divider()
        st.sidebar.subheader("📈 Stats")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Tips", "18")
            st.metric("ROI", "+12.4%")
        with col2:
            st.metric("Hit Rate", "67%")
            st.metric("Value", "€124")
        
        # Bouton actualisation
        if st.sidebar.button("🔄 Actualiser", use_container_width=True):
            st.rerun()
    
    def _display_main_content(self):
        """Affiche le contenu principal"""
        view_mode = st.session_state.view_mode
        
        if view_mode == 'dashboard':
            self._display_dashboard()
        elif view_mode == 'matches':
            self._display_matches()
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
        """Affiche le dashboard"""
        st.title("📊 Dashboard")
        
        # KPI Principaux
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
                <h3>🎯 Tips Actifs</h3>
                <h2>5</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
                <h3>📈 ROI 7j</h3>
                <h2>+5.8%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-box">
                <h3>✅ Hit Rate</h3>
                <h2>68.2%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-box">
                <h3>💰 Bankroll</h3>
                <h2>€1,124</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Derniers tips
        st.subheader("🎯 Derniers Tips")
        
        tips = [
            {"match": "PSG vs Marseille", "pred": "1", "cote": 1.65, "stake": "3%", "status": "✅ Gagné"},
            {"match": "Real vs Barca", "pred": "Over 2.5", "cote": 1.85, "stake": "4%", "status": "⏳ En cours"},
            {"match": "City vs Liverpool", "pred": "BTTS Yes", "cote": 1.65, "stake": "2%", "status": "✅ Gagné"},
            {"match": "Bayern vs Dortmund", "pred": "1", "cote": 1.75, "stake": "3%", "status": "❌ Perdu"}
        ]
        
        for tip in tips:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
                
                with col1:
                    st.write(f"**{tip['match']}**")
                
                with col2:
                    st.code(tip['pred'])
                
                with col3:
                    st.metric("Cote", tip['cote'])
                
                with col4:
                    st.write(tip['stake'])
                
                with col5:
                    if tip['status'] == '✅ Gagné':
                        st.success(tip['status'])
                    elif tip['status'] == '❌ Perdu':
                        st.error(tip['status'])
                    else:
                        st.info(tip['status'])
                
                st.divider()
        
        # Performance graph (simple)
        st.subheader("📈 Performance")
        
        # Créer un DataFrame pour le graphique
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        values = [1000 + i * 8 + random.randint(-15, 20) for i in range(30)]
        
        df = pd.DataFrame({
            'Date': dates,
            'Bankroll': values
        })
        
        # Utiliser le line_chart de Streamlit
        st.line_chart(df.set_index('Date'))
    
    def _display_matches(self):
        """Affiche la sélection des matchs"""
        st.title("🔍 Matchs Disponibles")
        
        # Filtres
        with st.expander("🎯 Filtres Avancés", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                date_filter = st.selectbox(
                    "Période",
                    ["Aujourd'hui", "Demain", "Week-end", "7 jours"]
                )
            
            with col2:
                league_filter = st.multiselect(
                    "Ligues",
                    ["Ligue 1", "Premier League", "La Liga", "Bundesliga", "Serie A"],
                    default=["Ligue 1", "Premier League"]
                )
            
            with col3:
                min_odds = st.number_input("Cote min", 1.2, 5.0, 1.5, 0.1)
        
        # Bouton recherche
        if st.button("🔍 Rechercher Matchs", type="primary", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                time.sleep(0.5)
                self._display_match_cards()
    
    def _display_match_cards(self):
        """Affiche les cartes de match"""
        matches = self.data.matches
        
        st.subheader(f"📋 Matchs Trouvés ({len(matches)})")
        
        for match in matches:
            # Obtenir les prédictions AI
            home_stats = self.data.get_team_stats(match['home'])
            away_stats = self.data.get_team_stats(match['away'])
            h2h_stats = self.data.get_h2h_stats(match['home'], match['away'])
            
            prediction = self.ai.predict(match['home'], match['away'], home_stats, away_stats, h2h_stats)
            
            # Afficher la carte
            self._render_match_card(match, prediction)
    
    def _render_match_card(self, match, prediction):
        """Affiche une carte de match"""
        date_str = match['date'].strftime("%d/%m/%Y")
        
        st.markdown(f"""
        <div class="pro-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4>{match['home']} vs {match['away']}</h4>
                    <p>🏆 {match['league']} | 📅 {date_str} | ⏰ {match['time']}</p>
                </div>
                <div>
                    <span class="badge badge-info">Confiance: {prediction['confidence']*100:.0f}%</span>
                    <span class="badge badge-premium">{match['venue']}</span>
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
        
        st.title(f"📈 Analyse: {match['home']} vs {match['away']}")
        
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
                "Météo": "☀️ 18°C"
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
            self._display_team_stats(home_stats)
        
        with col2:
            st.write(f"**{match['away']}**")
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
            'Statistique': ['Forme récente', 'Buts/m (marqués)', 'Buts/m (encaissés)', 'Possession', 'Tirs/match'],
            match['home']: [
                home_stats['form'],
                f"{home_stats['goals_for_avg']}",
                f"{home_stats['goals_against_avg']}",
                f"{home_stats['possession']}%",
                home_stats['shots_per_game']
            ],
            match['away']: [
                away_stats['form'],
                f"{away_stats['goals_for_avg']}",
                f"{away_stats['goals_against_avg']}",
                f"{away_stats['possession']}%",
                away_stats['shots_per_game']
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df.set_index('Statistique'), use_container_width=True)
        
        # Métriques avancées
        st.divider()
        st.subheader("📊 Métriques Avancées")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            xg_diff = home_stats['goals_for_avg'] - away_stats['goals_for_avg']
            st.metric("Différence xG", f"{xg_diff:+.1f}")
        
        with col2:
            defense_ratio = home_stats['goals_against_avg'] / away_stats['goals_against_avg']
            st.metric("Ratio défense", f"{defense_ratio:.2f}")
        
        with col3:
            attack_power = (home_stats['goals_for_avg'] + away_stats['goals_for_avg']) / 2
            st.metric("Puissance offensive", f"{attack_power:.1f}")
    
    def _display_value_bets(self):
        """Affiche les value bets"""
        st.title("💰 Value Bets")
        
        # Filtrer les value bets
        st.subheader("🔍 Scanner les Value Bets")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_value = st.slider("Valeur minimum (%)", 0, 20, 5)
        
        with col2:
            min_confidence = st.slider("Confiance AI minimum", 50, 90, 65)
        
        if st.button("🔍 Scanner tous les matchs", type="primary", use_container_width=True):
            with st.spinner("Recherche de value bets..."):
                time.sleep(1)
                self._display_value_bets_results(min_value, min_confidence)
    
    def _display_value_bets_results(self, min_value, min_confidence):
        """Affiche les résultats des value bets"""
        value_bets = []
        
        # Analyser quelques matchs
        for match in self.data.matches[:4]:
            home_stats = self.data.get_team_stats(match['home'])
            away_stats = self.data.get_team_stats(match['away'])
            h2h_stats = self.data.get_h2h_stats(match['home'], match['away'])
            
            prediction = self.ai.predict(match['home'], match['away'], home_stats, away_stats, h2h_stats)
            
            # Vérifier la confiance
            if prediction['confidence'] * 100 >= min_confidence:
                # Analyser les marchés
                markets = [
                    ('1', match['odds']['1'], prediction['home_win']),
                    ('N', match['odds']['N'], prediction['draw']),
                    ('2', match['odds']['2'], prediction['away_win'])
                ]
                
                for market_name, odds, probability in markets:
                    analysis = self.analyzer.calculate_value(probability, odds)
                    
                    if analysis['is_value'] and analysis['value_percent'] >= min_value:
                        value_bets.append({
                            'match': f"{match['home']} vs {match['away']}",
                            'market': market_name,
                            'odds': odds,
                            'probability': probability,
                            'value': analysis['value_percent'],
                            'confidence': prediction['confidence'] * 100,
                            'league': match['league']
                        })
        
        if value_bets:
            st.success(f"🎯 {len(value_bets)} Value Bets trouvées!")
            
            for bet in value_bets:
                with st.container():
                    st.markdown(f"""
                    <div class="pro-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h4>{bet['match']}</h4>
                                <p>🏆 {bet['league']} | 🎯 {bet['market']}</p>
                            </div>
                            <div>
                                <span class="badge badge-premium">+{bet['value']}% Valeur</span>
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
                                <h3>+{bet['value']}%</h3>
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
                            st.success(f"Pari placé sur {bet['match']} - {bet['market']}")
                    
                    st.divider()
        else:
            st.info("ℹ️ Aucune value bet trouvée avec les filtres actuels.")
    
    def _display_portfolio(self):
        """Affiche le portfolio"""
        st.title("💰 Mon Portfolio")
        
        # Résumé
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Paris actifs", "3")
        
        with col2:
            st.metric("Investissement", "€150")
        
        with col3:
            st.metric("Gains potentiels", "€285")
        
        with col4:
            st.metric("ROI projeté", "+90%")
        
        st.divider()
        
        # Paris en cours
        st.subheader("📊 Paris Actifs")
        
        active_bets = [
            {"match": "PSG vs Marseille", "type": "Over 2.5", "odds": 1.85, "stake": "€50", "status": "⏳ En attente"},
            {"match": "Real Madrid vs Barca", "type": "1", "odds": 2.10, "stake": "€60", "status": "⏳ En attente"},
            {"match": "City vs Liverpool", "type": "BTTS Yes", "odds": 1.65, "stake": "€40", "status": "⏳ En cours"}
        ]
        
        for bet in active_bets:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
                with col1:
                    st.write(f"**{bet['match']}**")
                with col2:
                    st.code(bet['type'])
                with col3:
                    st.metric("Cote", bet['odds'])
                with col4:
                    st.write(bet['stake'])
                with col5:
                    if bet['status'] == '⏳ En cours':
                        st.warning(bet['status'])
                    else:
                        st.info(bet['status'])
                st.divider()
        
        # Graphique de performance (simple)
        st.subheader("📈 Performance")
        
        # Créer un DataFrame simple
        dates = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
        values = [1000, 1020, 1045, 1060, 1080, 1100, 1124]
        
        df = pd.DataFrame({
            'Jour': dates,
            'Bankroll': values
        })
        
        st.line_chart(df.set_index('Jour'))
    
    def _display_settings(self):
        """Affiche les réglages"""
        st.title("⚙️ Réglages")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Profil")
            
            st.selectbox(
                "Niveau d'abonnement",
                ["Gratuit", "Basique", "Pro", "Entreprise"],
                index=2
            )
            
            bankroll = st.number_input(
                "Bankroll (€)",
                min_value=100,
                max_value=10000,
                value=st.session_state.bankroll,
                step=100
            )
            
            risk_profile = st.selectbox(
                "Profil de risque",
                ["Conservateur", "Modéré", "Agressif"],
                index=1
            )
        
        with col2:
            st.subheader("🔧 Paramètres de pari")
            
            max_stake = st.slider("Mise maximum (%)", 1, 20, 10)
            
            min_value = st.slider("Valeur minimum (%)", 0, 20, 5)
            
            st.checkbox("Notifications", value=True)
            st.checkbox("Alertes value bets", value=True)
        
        # Boutons
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Sauvegarder", type="primary", use_container_width=True):
                st.session_state.bankroll = bankroll
                st.session_state.risk_profile = risk_profile.lower()
                st.success("Réglages sauvegardés!")
        
        with col2:
            if st.button("🔄 Réinitialiser", use_container_width=True):
                st.session_state.bankroll = 1000
                st.session_state.risk_profile = 'moderate'
                st.rerun()
        
        with col3:
            if st.button("📤 Exporter", use_container_width=True):
                st.info("Fonctionnalité à venir...")

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main():
    """Fonction principale"""
    app = TipserProApp()
    app.run()

if __name__ == "__main__":
    main()
