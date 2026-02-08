# app.py - Système de Paris Sportifs IA
# Version 100% compatible Streamlit Cloud & GitHub

import pandas as pd
import numpy as np
import streamlit as st
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
import random
import math

# =============================================================================
# CONFIGURATION
# =============================================================================

# Configuration par défaut
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
            'streak': 0,  # Série actuelle
            'last_5_avg': 0.5,
            'home_record': {'played': 0, 'wins': 0, 'draws': 0, 'losses': 0},
            'away_record': {'played': 0, 'wins': 0, 'draws': 0, 'losses': 0},
        }
    
    def calculate_expected_score(self, rating_a, rating_b, home_advantage=0):
        """Calcule le score attendu"""
        return 1 / (1 + 10 ** ((rating_b - rating_a - home_advantage) / 400))
    
    def margin_of_victory_multiplier(self, goal_diff, elo_diff):
        """Multiplicateur basé sur la marge de victoire"""
        return math.log(abs(goal_diff) + 1) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))
    
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
        away_data['elo'] -= home_elo_change
        
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
        """Met à jour les statistiques"""
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
        """Calcule la force d'une équipe"""
        team_data = self.ratings.get(team_id)
        if not team_data:
            return 1500
        
        base_elo = team_data['elo']
        
        if include_form:
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
        
        # Modèle de match nul
        elo_diff = abs(home_strength - away_strength)
        draw_prob = 0.25 * np.exp(-elo_diff / 200)
        
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
            
            # Mélanger les équipes
            teams = list(league_teams)
            random.shuffle(teams)
            
            for i in range(0, len(teams) - 1, 2):
                home_id = teams[i]
                away_id = teams[i + 1]
                
                # Prédiction
                probs = self.elo_system.predict_match(home_id, away_id)
                
                # Générer score
                home_score, away_score = self.generate_score(probs)
                
                match = {
                    'date': match_date.strftime('%Y-%m-%d'),
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
        """Génère un score réaliste"""
        # Déterminer le résultat
        outcomes = ['home', 'draw', 'away']
        probabilities = [probs['home_win'], probs['draw'], probs['away_win']]
        result = random.choices(outcomes, weights=probabilities, k=1)[0]
        
        # Générer les buts
        if result == 'home':
            home_goals = np.random.poisson(1.8) + 1
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
            num_matches = random.randint(2, 4)
            teams = list(league_teams)
            random.shuffle(teams)
            
            for i in range(0, min(num_matches * 2, len(teams) - 1), 2):
                home_id = teams[i]
                away_id = teams[i + 1]
                
                probs = self.elo_system.predict_match(home_id, away_id)
                
                # Cotes simulées
                margin = 1.05
                home_odds = round(1 / (probs['home_win'] * margin), 2)
                draw_odds = round(1 / (probs['draw'] * margin), 2)
                away_odds = round(1 / (probs['away_win'] * margin), 2)
                
                match = {
                    'date': match_date.strftime('%Y-%m-%d'),
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
            if odds[bet_type] > 1:
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
        
        if best_edge > 0.02:
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
        """Calcule la mise selon Kelly"""
        if odds <= 1:
            return 0
        
        b = odds - 1
        p = probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0, min(kelly_fraction, 0.5))
        kelly_fraction *= fraction
        
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
        """Règle un pari"""
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
# FONCTIONS D'AFFICHAGE
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
        color: #1E88E5;
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
        background: #1E88E5;
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
    .tab-content {
        padding: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-title">⚽ Système de Paris Sportifs IA</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Elo Rating • Value Bets • Gestion de Bankroll • Compatible GitHub</div>', unsafe_allow_html=True)

def display_dashboard():
    """Affiche le dashboard principal"""
    st.header("📊 Dashboard Principal")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        bankroll_change = st.session_state.bankroll.current_bankroll - st.session_state.bankroll.initial_bankroll
        st.metric(
            "💰 Bankroll",
            f"€{st.session_state.bankroll.current_bankroll:,.0f}",
            f"€{bankroll_change:+,.0f}"
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
        value_bets = len([m for m in st.session_state.get('upcoming_matches', []) 
                         if m.get('value_bet')])
        st.metric("⚡ Value Bets", value_bets, "Aujourd'hui")
    
    # Évolution Bankroll
    st.subheader("📈 Évolution du Bankroll")
    
    if st.session_state.bankroll.bet_history:
        dates = [bet['date'].strftime('%Y-%m-%d') for bet in st.session_state.bankroll.bet_history]
        amounts = []
        current = st.session_state.bankroll.initial_bankroll
        
        for bet in st.session_state.bankroll.bet_history:
            if bet.get('result') == 'win':
                current += bet['potential_win']
            amounts.append(current)
        
        # Créer un tableau pour l'évolution
        evolution_df = pd.DataFrame({
            'Date': dates,
            'Bankroll': amounts
        })
        
        # Afficher sous forme de tableau avec graphique
        st.line_chart(evolution_df.set_index('Date'))
    else:
        st.info("Aucun pari enregistré")
    
    # Top Value Bets
    st.subheader("🏆 Top Value Bets")
    
    upcoming_matches = st.session_state.get('upcoming_matches', [])
    value_bets_list = [m for m in upcoming_matches if m.get('value_bet')]
    
    if value_bets_list:
        # Trier par score de value
        value_bets_list.sort(key=lambda x: x.get('value_score', 0), reverse=True)
        
        for i, match in enumerate(value_bets_list[:5], 1):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**#{i} - {match['home_name']} vs {match['away_name']}**")
                    st.write(f"*{match['league']} - {match['date']}*")
                with col2:
                    st.metric("Edge", f"{match.get('edge', 0)*100:.1f}%")
                with col3:
                    st.metric("Score", f"{match.get('value_score', 0):.1f}")
                st.divider()
    else:
        st.info("Aucun value bet détecté pour le moment")

def display_value_bets():
    """Affiche les value bets"""
    st.header("🎯 Détection de Value Bets")
    
    if 'upcoming_matches' not in st.session_state:
        st.warning("Générez d'abord des matchs depuis le Dashboard")
        return
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_edge = st.slider("Edge minimum (%)", 1.0, 10.0, 2.0, 0.5)
    
    with col2:
        min_confidence = st.slider("Confiance minimum (%)", 50.0, 95.0, 60.0, 5.0)
    
    with col3:
        leagues = list(set(m['league'] for m in st.session_state.upcoming_matches))
        league_filter = st.selectbox("Filtrer par ligue", ["Toutes"] + leagues)
    
    # Filtrer les matchs
    filtered_matches = []
    for match in st.session_state.upcoming_matches:
        if match.get('value_bet'):
            edge = match.get('edge', 0) * 100
            
            if match['value_bet'] == 'home':
                prob = match.get('pred_home_win', 0)
            elif match['value_bet'] == 'draw':
                prob = match.get('pred_draw', 0)
            else:
                prob = match.get('pred_away_win', 0)
            
            if (edge >= min_edge and 
                prob * 100 >= min_confidence and
                (league_filter == "Toutes" or match['league'] == league_filter)):
                filtered_matches.append(match)
    
    # Afficher les value bets
    if filtered_matches:
        st.success(f"🎯 {len(filtered_matches)} value bets détectés!")
        
        for match in filtered_matches:
            with st.expander(f"🏆 **{match['home_name']} vs {match['away_name']}** - Edge: {match.get('edge', 0)*100:.1f}%"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Ligue:** {match['league']}")
                    st.write(f"**Date:** {match['date']}")
                    st.write(f"**Type de pari:** {match['value_bet'].upper()}")
                    
                    # Probabilités
                    st.write("**Probabilités:**")
                    st.write(f"- Domicile: {match['pred_home_win']*100:.1f}% (cote {match['home_odds']})")
                    st.write(f"- Nul: {match['pred_draw']*100:.1f}% (cote {match['draw_odds']})")
                    st.write(f"- Extérieur: {match['pred_away_win']*100:.1f}% (cote {match['away_odds']})")
                
                with col2:
                    # Calculs de value
                    edge_pct = match.get('edge', 0) * 100
                    st.metric("📈 Edge", f"{edge_pct:.1f}%")
                    
                    # Mise recommandée
                    if match['value_bet'] == 'home':
                        prob = match['pred_home_win']
                        odds = match['home_odds']
                    elif match['value_bet'] == 'draw':
                        prob = match['pred_draw']
                        odds = match['draw_odds']
                    else:
                        prob = match['pred_away_win']
                        odds = match['away_odds']
                    
                    kelly_stake = st.session_state.bankroll.calculate_kelly_stake(
                        st.session_state.bankroll.current_bankroll,
                        prob,
                        odds,
                        fraction=0.25
                    )
                    
                    st.metric("💰 Mise Kelly", f"€{kelly_stake:,.0f}")
                    
                    # Expected Value
                    ev = kelly_stake * match.get('edge',
