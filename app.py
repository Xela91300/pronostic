# app.py - Système de Paris Sportifs IA
# Version compatible Streamlit Cloud & GitHub

import pandas as pd
import numpy as np
import streamlit as st
import json
import time
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Configuration par défaut - fonctionne sans secrets
DEMO_MODE = True

# Ligues de football populaires
LEAGUES = {
    "Premier League (Angleterre)": {"id": 39, "name": "Premier League"},
    "La Liga (Espagne)": {"id": 140, "name": "La Liga"},
    "Serie A (Italie)": {"id": 135, "name": "Serie A"},
    "Bundesliga (Allemagne)": {"id": 78, "name": "Bundesliga"},
    "Ligue 1 (France)": {"id": 61, "name": "Ligue 1"},
    "Champions League": {"id": 2, "name": "UEFA Champions League"},
}

# Équipes populaires avec Elo initial
TEAMS_DATA = {
    # Premier League
    33: {"name": "Manchester United", "elo": 1850, "league": "Premier League"},
    34: {"name": "Newcastle", "elo": 1750, "league": "Premier League"},
    35: {"name": "Bournemouth", "elo": 1650, "league": "Premier League"},
    36: {"name": "Fulham", "elo": 1700, "league": "Premier League"},
    39: {"name": "Wolves", "elo": 1720, "league": "Premier League"},
    40: {"name": "Liverpool", "elo": 1950, "league": "Premier League"},
    41: {"name": "Southampton", "elo": 1680, "league": "Premier League"},
    42: {"name": "Arsenal", "elo": 1900, "league": "Premier League"},
    45: {"name": "Everton", "elo": 1750, "league": "Premier League"},
    46: {"name": "Leicester", "elo": 1780, "league": "Premier League"},
    47: {"name": "Tottenham", "elo": 1880, "league": "Premier League"},
    48: {"name": "West Ham", "elo": 1770, "league": "Premier League"},
    49: {"name": "Chelsea", "elo": 1870, "league": "Premier League"},
    50: {"name": "Manchester City", "elo": 2000, "league": "Premier League"},
    51: {"name": "Brighton", "elo": 1800, "league": "Premier League"},
    52: {"name": "Crystal Palace", "elo": 1730, "league": "Premier League"},
    
    # La Liga
    529: {"name": "Barcelona", "elo": 1950, "league": "La Liga"},
    530: {"name": "Atletico Madrid", "elo": 1900, "league": "La Liga"},
    531: {"name": "Athletic Club", "elo": 1820, "league": "La Liga"},
    532: {"name": "Valencia", "elo": 1800, "league": "La Liga"},
    533: {"name": "Villarreal", "elo": 1850, "league": "La Liga"},
    534: {"name": "Real Madrid", "elo": 1980, "league": "La Liga"},
    536: {"name": "Sevilla", "elo": 1830, "league": "La Liga"},
    537: {"name": "Real Betis", "elo": 1810, "league": "La Liga"},
    
    # Serie A
    487: {"name": "Juventus", "elo": 1920, "league": "Serie A"},
    488: {"name": "Napoli", "elo": 1900, "league": "Serie A"},
    489: {"name": "AC Milan", "elo": 1880, "league": "Serie A"},
    490: {"name": "Inter Milan", "elo": 1910, "league": "Serie A"},
    492: {"name": "Roma", "elo": 1860, "league": "Serie A"},
    496: {"name": "Lazio", "elo": 1840, "league": "Serie A"},
    
    # Bundesliga
    157: {"name": "Bayern Munich", "elo": 1990, "league": "Bundesliga"},
    158: {"name": "Dortmund", "elo": 1880, "league": "Bundesliga"},
    159: {"name": "Leverkusen", "elo": 1850, "league": "Bundesliga"},
    160: {"name": "RB Leipzig", "elo": 1870, "league": "Bundesliga"},
    161: {"name": "Wolfsburg", "elo": 1780, "league": "Bundesliga"},
    162: {"name": "Frankfurt", "elo": 1820, "league": "Bundesliga"},
    
    # Ligue 1
    77: {"name": "Marseille", "elo": 1830, "league": "Ligue 1"},
    79: {"name": "PSG", "elo": 1970, "league": "Ligue 1"},
    80: {"name": "Lens", "elo": 1800, "league": "Ligue 1"},
    81: {"name": "Monaco", "elo": 1850, "league": "Ligue 1"},
    82: {"name": "Rennes", "elo": 1820, "league": "Ligue 1"},
    83: {"name": "Lyon", "elo": 1810, "league": "Ligue 1"},
    84: {"name": "Lille", "elo": 1840, "league": "Ligue 1"},
}

# =============================================================================
# SYSTÈME ELO AVANCÉ
# =============================================================================

class AdvancedEloSystem:
    """Système Elo avancé avec form et momentum"""
    
    def __init__(self, k_factor=32, home_advantage=70):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings = {team_id: self._init_team_data(data) for team_id, data in TEAMS_DATA.items()}
        self.match_history = []
    
    def _init_team_data(self, team_data):
        """Initialise les données d'une équipe"""
        return {
            'name': team_data['name'],
            'elo': team_data['elo'],
            'games_played': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_for': 0,
            'goals_against': 0,
            'form': [],  # Derniers résultats (1=win, 0.5=draw, 0=loss)
            'streak': 0,  # Série actuelle (positive=gagnante, negative=perdante)
            'last_5_avg': 0.5,  # Moyenne sur 5 derniers matchs
            'home_record': {'played': 0, 'wins': 0, 'draws': 0, 'losses': 0},
            'away_record': {'played': 0, 'wins': 0, 'draws': 0, 'losses': 0},
        }
    
    def calculate_expected_score(self, rating_a, rating_b, home_advantage=0):
        """Calcule le score attendu"""
        return 1 / (1 + 10 ** ((rating_b - rating_a - home_advantage) / 400))
    
    def margin_of_victory_multiplier(self, goal_diff, elo_diff):
        """Multiplicateur basé sur la marge de victoire"""
        return np.log(abs(goal_diff) + 1) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))
    
    def update_ratings(self, home_id, away_id, home_score, away_score, is_neutral=False):
        """Met à jour les ratings Elo après un match"""
        
        home_data = self.ratings[home_id]
        away_data = self.ratings[away_id]
        
        # Déterminer le résultat
        if home_score > away_score:
            actual_home = 1.0
            actual_away = 0.0
            home_result = 'win'
            away_result = 'loss'
        elif home_score < away_score:
            actual_home = 0.0
            actual_away = 1.0
            home_result = 'loss'
            away_result = 'win'
        else:
            actual_home = 0.5
            actual_away = 0.5
            home_result = 'draw'
            away_result = 'draw'
        
        # Avantage terrain
        home_adv = 0 if is_neutral else self.home_advantage
        
        # Score attendu
        expected_home = self.calculate_expected_score(
            home_data['elo'], away_data['elo'], home_adv
        )
        expected_away = 1 - expected_home
        
        # Multiplicateur marge de victoire
        goal_diff = home_score - away_score
        elo_diff = home_data['elo'] - away_data['elo']
        mov_multiplier = self.margin_of_victory_multiplier(goal_diff, elo_diff)
        
        # Mise à jour Elo
        k = self.k_factor * mov_multiplier
        home_elo_change = k * (actual_home - expected_home)
        away_elo_change = k * (actual_away - expected_away)
        
        home_data['elo'] += home_elo_change
        away_data['elo'] -= home_elo_change  # Conservation du total Elo
        
        # Mise à jour statistiques
        self._update_stats(home_data, home_score, away_score, home_result, is_home=True)
        self._update_stats(away_data, away_score, home_score, away_result, is_home=False)
        
        # Mise à jour forme
        self._update_form(home_data, actual_home)
        self._update_form(away_data, actual_away)
        
        # Sauvegarde historique
        match_record = {
            'date': datetime.now(),
            'home_id': home_id,
            'away_id': away_id,
            'home_name': home_data['name'],
            'away_name': away_data['name'],
            'home_score': home_score,
            'away_score': away_score,
            'home_elo_before': home_data['elo'] - home_elo_change,
            'away_elo_before': away_data['elo'] + home_elo_change,
            'home_elo_change': home_elo_change,
            'result': home_result
        }
        self.match_history.append(match_record)
        
        return home_elo_change
    
    def _update_stats(self, team_data, goals_for, goals_against, result, is_home):
        """Met à jour les statistiques d'une équipe"""
        team_data['games_played'] += 1
        team_data['goals_for'] += goals_for
        team_data['goals_against'] += goals_against
        
        if result == 'win':
            team_data['wins'] += 1
            team_data['streak'] = max(1, team_data['streak'] + 1)
        elif result == 'draw':
            team_data['draws'] += 1
            team_data['streak'] = 0
        else:
            team_data['losses'] += 1
            team_data['streak'] = min(-1, team_data['streak'] - 1)
        
        # Record domicile/extérieur
        if is_home:
            record = team_data['home_record']
        else:
            record = team_data['away_record']
        
        record['played'] += 1
        if result == 'win':
            record['wins'] += 1
        elif result == 'draw':
            record['draws'] += 1
        else:
            record['losses'] += 1
    
    def _update_form(self, team_data, result_value):
        """Met à jour la forme de l'équipe"""
        team_data['form'].append(result_value)
        if len(team_data['form']) > 10:
            team_data['form'] = team_data['form'][-10:]
        
        # Moyenne sur 5 derniers
        last_5 = team_data['form'][-5:] if len(team_data['form']) >= 5 else team_data['form']
        team_data['last_5_avg'] = np.mean(last_5) if last_5 else 0.5
    
    def get_team_form(self, team_id, last_n=5):
        """Retourne la forme sur les N derniers matchs"""
        team_data = self.ratings.get(team_id)
        if not team_data or len(team_data['form']) < last_n:
            return 0.5
        return np.mean(team_data['form'][-last_n:])
    
    def get_team_strength(self, team_id, include_form=True):
        """Calcule la force d'une équipe (Elo + form)"""
        team_data = self.ratings.get(team_id)
        if not team_data:
            return 1500
        
        base_elo = team_data['elo']
        
        if include_form:
            # Ajustement basé sur la forme récente
            form_factor = 1.0 + (team_data['last_5_avg'] - 0.5) * 0.2
            return base_elo * form_factor
        
        return base_elo
    
    def predict_match(self, home_id, away_id, is_neutral=False):
        """Prédit les probabilités d'un match"""
        home_data = self.ratings.get(home_id)
        away_data = self.ratings.get(away_id)
        
        if not home_data or not away_data:
            return {'home_win': 0.33, 'draw': 0.34, 'away_win': 0.33}
        
        # Force des équipes avec form
        home_strength = self.get_team_strength(home_id)
        away_strength = self.get_team_strength(away_id)
        
        # Avantage terrain
        home_adv = 0 if is_neutral else self.home_advantage
        
        # Probabilité de victoire domicile
        home_win_prob = self.calculate_expected_score(
            home_strength, away_strength, home_adv
        )
        
        # Modèle de match nul basé sur la différence Elo
        elo_diff = abs(home_strength - away_strength)
        draw_prob = 0.25 * np.exp(-elo_diff / 200)  # Plus de nuls quand équipes proches
        
        # Normalisation
        total = home_win_prob + draw_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob = 1 - home_win_prob - draw_prob
        
        return {
            'home_win': home_win_prob,
            'draw': draw_prob,
            'away_win': away_win_prob
        }

# =============================================================================
# GÉNÉRATION DE DONNÉES DÉMO
# =============================================================================

class DemoDataGenerator:
    """Générateur de données démo réalistes"""
    
    def __init__(self, elo_system):
        self.elo_system = elo_system
        self.generated_matches = []
        
    def generate_season_matches(self, league_teams, num_rounds=5):
        """Génère des matchs pour une saison"""
        matches = []
        today = datetime.now()
        
        for round_num in range(num_rounds):
            match_date = today - timedelta(days=(num_rounds - round_num - 1) * 3)
            
            # Mélanger les équipes pour créer des paires
            teams = list(league_teams)
            np.random.shuffle(teams)
            
            for i in range(0, len(teams) - 1, 2):
                home_id = teams[i]
                away_id = teams[i + 1]
                
                # Prédiction
                probs = self.elo_system.predict_match(home_id, away_id)
                
                # Générer score basé sur les probabilités
                home_score, away_score = self.generate_score(probs)
                
                match = {
                    'date': match_date,
                    'home_id': home_id,
                    'away_id': away_id,
                    'home_name': self.elo_system.ratings[home_id]['name'],
                    'away_name': self.elo_system.ratings[away_id]['name'],
                    'league': self.elo_system.ratings[home_id]['league'],
                    'home_score': home_score,
                    'away_score': away_score,
                    'pred_home_win': probs['home_win'],
                    'pred_draw': probs['draw'],
                    'pred_away_win': probs['away_win'],
                    'status': 'FT',
                    'round': round_num + 1
                }
                
                matches.append(match)
                
                # Mettre à jour Elo
                self.elo_system.update_ratings(home_id, away_id, home_score, away_score)
        
        self.generated_matches.extend(matches)
        return matches
    
    def generate_score(self, probs):
        """Génère un score réaliste basé sur les probabilités"""
        # Déterminer le résultat
        result = np.random.choice(['home', 'draw', 'away'], p=[probs['home_win'], probs['draw'], probs['away_win']])
        
        # Générer les buts selon le résultat
        if result == 'home':
            home_goals = np.random.poisson(1.8) + 1  # Plus de buts pour le gagnant
            away_goals = np.random.poisson(1.0)
        elif result == 'away':
            home_goals = np.random.poisson(1.0)
            away_goals = np.random.poisson(1.8) + 1
        else:  # draw
            goals = np.random.poisson(1.2)
            home_goals = goals
            away_goals = goals
        
        return int(home_goals), int(away_goals)
    
    def generate_upcoming_matches(self, league_teams, days_ahead=7):
        """Génère des matchs à venir"""
        upcoming = []
        today = datetime.now()
        
        for day in range(1, days_ahead + 1):
            match_date = today + timedelta(days=day)
            
            # 2-3 matchs par jour
            num_matches = np.random.randint(2, 4)
            teams = list(league_teams)
            np.random.shuffle(teams)
            
            for i in range(0, min(num_matches * 2, len(teams) - 1), 2):
                home_id = teams[i]
                away_id = teams[i + 1]
                
                probs = self.elo_system.predict_match(home_id, away_id)
                
                # Cotes simulées (avec marges bookmakers)
                margin = 1.05  # Marge de 5%
                home_odds = round(1 / (probs['home_win'] * margin), 2)
                draw_odds = round(1 / (probs['draw'] * margin), 2)
                away_odds = round(1 / (probs['away_win'] * margin), 2)
                
                match = {
                    'date': match_date,
                    'home_id': home_id,
                    'away_id': away_id,
                    'home_name': self.elo_system.ratings[home_id]['name'],
                    'away_name': self.elo_system.ratings[away_id]['name'],
                    'league': self.elo_system.ratings[home_id]['league'],
                    'pred_home_win': probs['home_win'],
                    'pred_draw': probs['draw'],
                    'pred_away_win': probs['away_win'],
                    'home_odds': home_odds,
                    'draw_odds': draw_odds,
                    'away_odds': away_odds,
                    'status': 'NS',
                    'value_bet': None,
                    'value_score': 0
                }
                
                # Détection value bet
                value_info = self.detect_value_bet(match)
                if value_info:
                    match['value_bet'] = value_info['type']
                    match['value_score'] = value_info['score']
                    match['edge'] = value_info['edge']
                
                upcoming.append(match)
        
        return upcoming
    
    def detect_value_bet(self, match):
        """Détecte les value bets"""
        probs = {
            'home': match['pred_home_win'],
            'draw': match['pred_draw'],
            'away': match['pred_away_win']
        }
        
        odds = {
            'home': match['home_odds'],
            'draw': match['draw_odds'],
            'away': match['away_odds']
        }
        
        best_value = None
        best_edge = 0
        
        for bet_type in ['home', 'draw', 'away']:
            if odds[bet_type] > 1:  # Éviter division par zéro
                implied_prob = 1 / odds[bet_type]
                edge = probs[bet_type] - implied_prob
                
                if edge > best_edge:
                    best_edge = edge
                    best_value = {
                        'type': bet_type,
                        'edge': edge,
                        'prob': probs[bet_type],
                        'odds': odds[bet_type],
                        'implied_prob': implied_prob
                    }
        
        if best_edge > 0.02:  # Edge minimum de 2%
            # Score de value (edge * confidence)
            confidence = best_value['prob']
            value_score = best_edge * confidence * 100
            
            return {
                'type': best_value['type'],
                'edge': best_edge,
                'score': value_score,
                'details': best_value
            }
        
        return None

# =============================================================================
# GESTION DE BANKROLL
# =============================================================================

class BankrollManager:
    """Gestion avancée de bankroll"""
    
    def __init__(self, initial_bankroll=10000):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.bet_history = []
        self.performance = {
            'total_bets': 0,
            'won_bets': 0,
            'lost_bets': 0,
            'void_bets': 0,
            'total_staked': 0,
            'total_return': 0,
            'roi': 0.0,
            'biggest_win': 0,
            'biggest_loss': 0,
            'current_streak': 0,
            'longest_win_streak': 0,
            'longest_loss_streak': 0
        }
    
    def calculate_kelly_stake(self, bankroll, probability, odds, fraction=0.25):
        """Calcule la mise selon le critère de Kelly fractionnaire"""
        if odds <= 1:
            return 0
        
        b = odds - 1
        p = probability
        q = 1 - p
        
        # Formule Kelly: f* = (bp - q) / b
        kelly_fraction = (b * p - q) / b
        
        # Limiter et fractionner
        kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Limite à 50%
        kelly_fraction *= fraction  # Fraction de Kelly
        
        return kelly_fraction * bankroll
    
    def place_bet(self, amount, odds, probability, result=None):
        """Place un pari"""
        if amount > self.current_bankroll:
            return {"error": "Fonds insuffisants"}
        
        bet_id = len(self.bet_history) + 1
        bet = {
            'id': bet_id,
            'date': datetime.now(),
            'amount': amount,
            'odds': odds,
            'probability': probability,
            'potential_win': amount * (odds - 1),
            'result': result,
            'status': 'pending'
        }
        
        self.current_bankroll -= amount
        self.bet_history.append(bet)
        self.performance['total_bets'] += 1
        self.performance['total_staked'] += amount
        
        return bet
    
    def settle_bet(self, bet_id, result):
        """Règle un pari (win/loss)"""
        for bet in self.bet_history:
            if bet['id'] == bet_id and bet['status'] == 'pending':
                bet['result'] = result
                bet['status'] = 'settled'
                
                if result == 'win':
                    winnings = bet['amount'] * bet['odds']
                    self.current_bankroll += winnings
                    self.performance['won_bets'] += 1
                    self.performance['total_return'] += winnings
                    self.performance['current_streak'] = max(0, self.performance['current_streak'] + 1)
                    
                    if winnings > self.performance['biggest_win']:
                        self.performance['biggest_win'] = winnings
                
                elif result == 'loss':
                    self.performance['lost_bets'] += 1
                    self.performance['current_streak'] = min(0, self.performance['current_streak'] - 1)
                    
                    if bet['amount'] > self.performance['biggest_loss']:
                        self.performance['biggest_loss'] = bet['amount']
                
                # Mettre à jour ROI
                if self.performance['total_staked'] > 0:
                    self.performance['roi'] = (
                        (self.performance['total_return'] - self.performance['total_staked']) / 
                        self.performance['total_staked'] * 100
                    )
                
                # Mettre à jour les séries
                if self.performance['current_streak'] > self.performance['longest_win_streak']:
                    self.performance['longest_win_streak'] = self.performance['current_streak']
                elif abs(self.performance['current_streak']) > self.performance['longest_loss_streak']:
                    self.performance['longest_loss_streak'] = abs(self.performance['current_streak'])
                
                return bet
        
        return {"error": "Pari non trouvé"}

# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

def setup_page():
    """Configure la page Streamlit"""
    st.set_page_config(
        page_title="Système de Paris Sportifs IA",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1E88E5, #4A00E0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-title {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .value-bet-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border-left: 5px solid #4CAF50;
    }
    .match-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-title">⚽ Système de Paris Sportifs IA</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Elo Rating • Value Bets • Gestion de Bankroll • Compatible GitHub</div>', unsafe_allow_html=True)

def main():
    """Fonction principale"""
    setup_page()
    
    # Initialisation session state
    if 'elo_system' not in st.session_state:
        st.session_state.elo_system = AdvancedEloSystem()
    
    if 'data_gen' not in st.session_state:
        st.session_state.data_gen = DemoDataGenerator(st.session_state.elo_system)
    
    if 'bankroll' not in st.session_state:
        st.session_state.bankroll = BankrollManager()
    
    if 'generated_matches' not in st.session_state:
        st.session_state.generated_matches = []
        st.session_state.upcoming_matches = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Bankroll
        bankroll_amount = st.number_input(
            "💰 Bankroll Initial (€)",
            min_value=100,
            max_value=100000,
            value=10000,
            step=500
        )
        
        if bankroll_amount != st.session_state.bankroll.initial_bankroll:
            st.session_state.bankroll = BankrollManager(bankroll_amount)
        
        # Paramètres risque
        st.subheader("📊 Paramètres Risque")
        
        kelly_fraction = st.slider(
            "Fraction de Kelly",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Pourcentage du Kelly full à utiliser"
        )
        
        min_edge = st.slider(
            "Edge Minimum (%)",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=0.5
        )
        
        # Sélection ligue
        st.subheader("🏆 Sélection Ligue")
        
        selected_league = st.selectbox(
            "Choisir une ligue",
            list(LEAGUES.keys())
        )
        
        league_info = LEAGUES[selected_league]
        
        # Boutons d'action
        st.subheader("🔄 Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎯 Générer Matchs", type="primary"):
                with st.spinner("Génération des matchs..."):
                    league_teams = [id for id, data in TEAMS_DATA.items() 
                                   if data['league'] == league_info['name']]
                    
                    # Générer matchs passés
                    historical = st.session_state.data_gen.generate_season_matches(
                        league_teams, num_rounds=5
                    )
                    st.session_state.generated_matches = historical
                    
                    # Générer matchs à venir
                    upcoming = st.session_state.data_gen.generate_upcoming_matches(
                        league_teams, days_ahead=7
                    )
                    st.session_state.upcoming_matches = upcoming
                    
                    st.success(f"{len(historical)} matchs historiques générés!")
                    st.success(f"{len(upcoming)} matchs à venir générés!")
        
        with col2:
            if st.button("🧹 Réinitialiser"):
                for key in list(st.session_state.keys()):
                    if key not in ['elo_system', 'data_gen']:
                        del st.session_state[key]
                st.session_state.elo_system = AdvancedEloSystem()
                st.session_state.data_gen = DemoDataGenerator(st.session_state.elo_system)
                st.rerun()
        
        # Info
        st.divider()
        st.info("""
        **📌 Mode Démo Activé**
        
        Données simulées pour démonstration.
        
        Pour utiliser des données réelles:
        1. Créez un compte sur api-sports.io
        2. Ajoutez votre clé API dans les secrets
        3. Désactivez le mode démo
        """)
    
    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", 
        "🎯 Value Bets", 
        "🤖 Modèles", 
        "💰 Bankroll"
    ])
    
    with tab1:
        display_dashboard()
    
    with tab2:
        display_value_bets()
    
    with tab3:
        display_models()
    
    with tab4:
        display_bankroll()

def display_dashboard():
    """Affiche le dashboard principal"""
    st.header("📊 Dashboard Principal")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Bankroll",
            f"€{st.session_state.bankroll.current_bankroll:,.0f}",
            f"€{st.session_state.bankroll.current_bankroll - st.session_state.bankroll.initial_bankroll:+,.0f}"
        )
    
    with col2:
        total_bets = st.session_state.bankroll.performance['total_bets']
        won_bets = st.session_state.bankroll.performance['won_bets']
        win_rate = (won_bets / total_bets * 100) if total_bets > 0 else 0
        st.metric("📈 Hit Rate", f"{win_rate:.1f}%", f"{total_bets} paris")
    
    with col3:
        roi = st.session_state.bankroll.performance['roi']
        st.metric("🎯 ROI", f"{roi:.1f}%", "Cumulé")
    
    with col4:
        value_bets = len([m for m in st.session_state.upcoming_matches 
                         if m.get('value_bet')])
        st.metric("⚡ Value Bets", value_bets, "Aujourd'hui")
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Évolution Bankroll")
        
        if st.session_state.bankroll.bet_history:
            dates = [bet['date'] for bet in st.session_state.bankroll.bet_history]
            amounts = []
            current = st.session_state.bankroll.initial_bankroll
            
            for bet in st.session_state.bankroll.bet_history:
                if bet.get('result') == 'win':
                    current += bet['potential_win']
                amounts.append(current)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=amounts,
                mode='lines+markers',
                name='Bankroll',
                line=dict(color='#1E88E5', width=3)
            ))
            
            fig.update_layout(
                title="Évolution du Bankroll",
                xaxis_title="Date",
                yaxis_title="Bankroll (€)",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
