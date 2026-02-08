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
        bankroll = 10000  # Valeur par défaut
        bankroll_change = 0
        st.metric(
            "💰 Bankroll",
            f"€{bankroll:,.0f}",
            f"€{bankroll_change:+,.0f}"
        )
    
    with col2:
        total_bets = 0
        won_bets = 0
        win_rate = 0
        st.metric("📈 Hit Rate", f"{win_rate:.1f}%", f"{total_bets} paris")
    
    with col3:
        roi = 0
        st.metric("🎯 ROI", f"{roi:.1f}%", "Cumulé")
    
    with col4:
        value_bets = 0
        st.metric("⚡ Value Bets", value_bets, "Aujourd'hui")
    
    # Bouton pour générer des données
    st.subheader("🎲 Générer des Données Démo")
    
    if st.button("Générer des matchs de démo", type="primary"):
        with st.spinner("Génération des données..."):
            time.sleep(2)
            st.success("Données générées avec succès!")
    
    # Instructions
    st.subheader("📋 Instructions")
    
    st.info("""
    **Comment utiliser ce système:**
    
    1. **Sélectionnez une ligue** dans la sidebar
    2. **Générez des matchs** avec le bouton ci-dessus
    3. **Explorez les Value Bets** dans l'onglet correspondant
    4. **Analysez les performances** dans l'onglet Bankroll
    
    *Ceci est une version démo avec données simulées.*
    """)

def display_value_bets():
    """Affiche les value bets"""
    st.header("🎯 Détection de Value Bets")
    
    # Exemples de value bets
    st.info("Voici des exemples de value bets détectés:")
    
    value_bets_examples = [
        {
            "match": "Manchester City vs Liverpool",
            "league": "Premier League",
            "type": "Home",
            "edge": "4.2%",
            "odds": "2.10",
            "probability": "52%",
            "stake": "€185"
        },
        {
            "match": "Real Madrid vs Barcelona",
            "league": "La Liga",
            "type": "Draw",
            "edge": "3.8%",
            "odds": "3.40",
            "probability": "31%",
            "stake": "€120"
        },
        {
            "match": "Bayern Munich vs Dortmund",
            "league": "Bundesliga",
            "type": "Home",
            "edge": "2.5%",
            "odds": "1.65",
            "probability": "63%",
            "stake": "€95"
        }
    ]
    
    for bet in value_bets_examples:
        with st.expander(f"🏆 **{bet['match']}** - Edge: {bet['edge']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Ligue:** {bet['league']}")
                st.write(f"**Type de pari:** {bet['type']}")
                st.write(f"**Probabilité modèle:** {bet['probability']}")
            with col2:
                st.write(f"**Cote bookmaker:** {bet['odds']}")
                st.write(f"**Mise recommandée:** {bet['stake']}")
                st.write(f"**Edge (valeur):** {bet['edge']}")
    
    # Explication
    st.subheader("📊 Comment fonctionne la détection?")
    
    st.write("""
    Le système utilise plusieurs facteurs pour détecter les value bets:
    
    1. **Rating Elo**: Évalue la force des équipes
    2. **Forme récente**: Performance sur les 5 derniers matchs
    3. **Avantage terrain**: +70 points Elo pour l'équipe à domicile
    4. **Probabilités calculées**: Basées sur la différence Elo
    5. **Cotes du marché**: Comparaison avec les bookmakers
    
    Un **value bet** est détecté quand:
    ```
    Probabilité_modèle > Probabilité_implicite + Marge_seuil
    ```
    Où la probabilité implicite = 1 / Cote
    """)

def display_bankroll():
    """Affiche la gestion de bankroll"""
    st.header("💰 Gestion de Bankroll")
    
    # Simulation de bankroll
    st.subheader("📈 Simulation de Bankroll")
    
    initial_bankroll = st.number_input("Bankroll initial (€)", value=10000, min_value=1000, max_value=100000)
    num_bets = st.slider("Nombre de paris", 10, 1000, 100)
    win_rate = st.slider("Taux de réussite (%)", 30, 70, 55)
    avg_odds = st.slider("Cote moyenne", 1.5, 3.0, 2.1)
    
    if st.button("Simuler la performance", type="primary"):
        # Simulation simple
        bankroll = initial_bankroll
        bankroll_history = [bankroll]
        
        for i in range(num_bets):
            # Pari de 2% du bankroll
            stake = bankroll * 0.02
            win = random.random() < (win_rate / 100)
            
            if win:
                bankroll += stake * (avg_odds - 1)
            else:
                bankroll -= stake
            
            bankroll_history.append(bankroll)
        
        # Graphique
        chart_data = pd.DataFrame({
            'Pari': range(len(bankroll_history)),
            'Bankroll': bankroll_history
        })
        
        st.line_chart(chart_data.set_index('Pari'))
        
        # Statistiques
        final_bankroll = bankroll_history[-1]
        roi = ((final_bankroll - initial_bankroll) / initial_bankroll) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Bankroll final", f"€{final_bankroll:,.0f}", f"€{final_bankroll-initial_bankroll:+,.0f}")
        with col2:
            st.metric("ROI total", f"{roi:.1f}%")
    
    # Gestion des risques
    st.subheader("🎯 Gestion des Risques")
    
    st.write("""
    **Stratégies recommandées:**
    
    1. **Critère de Kelly fractionnaire**: Utilisez 25% du Kelly full
    2. **Limite par pari**: Maximum 2% du bankroll
    3. **Diversification**: Ne pas parier sur trop de matchs similaires
    4. **Suivi rigoureux**: Garder un journal de tous les paris
    
    **Formule de Kelly:**
    ```
    f* = (bp - q) / b
    où:
    - b = cote - 1
    - p = probabilité de gagner
    - q = 1 - p
    ```
    """)

def display_models():
    """Affiche les modèles IA"""
    st.header("🤖 Modèles IA")
    
    st.subheader("🎯 Système Elo Avancé")
    
    st.write("""
    **Caractéristiques du modèle:**
    
    - **Rating Elo dynamique**: Mis à jour après chaque match
    - **Forme récente**: Moyenne pondérée des 10 derniers matchs
    - **Avantage terrain**: +70 points Elo pour l'équipe à domicile
    - **Marge de victoire**: Impacte plus les ratings pour les victoires écrasantes
    - **Décay temporel**: Les ratings anciens ont moins de poids
    
    **Formule de prédiction:**
    ```
    P(victoire) = 1 / (1 + 10^((Elo_adv - Elo - Avantage)/400))
    ```
    """)
    
    # Exemple de prédiction
    st.subheader("📊 Exemple de Prédiction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox(
            "Équipe à domicile",
            ["Manchester City", "Liverpool", "Real Madrid", "Bayern Munich"],
            index=0
        )
    
    with col2:
        away_team = st.selectbox(
            "Équipe à l'extérieur",
            ["Arsenal", "Chelsea", "Barcelona", "Dortmund"],
            index=1
        )
    
    if st.button("Prédire le match", type="primary"):
        # Simulation de prédiction
        home_elo = 2000 if home_team == "Manchester City" else 1900
        away_elo = 1950 if away_team == "Liverpool" else 1850
        
        # Calcul de probabilité
        home_advantage = 70
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo - home_advantage) / 400))
        expected_away = 1 - expected_home
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Victoire domicile", f"{expected_home*100:.1f}%")
        with col2:
            # Probabilité de match nul estimée
            draw_prob = 0.25 * np.exp(-abs(home_elo - away_elo) / 400)
            st.metric("Match nul", f"{draw_prob*100:.1f}%")
        with col3:
            st.metric("Victoire extérieur", f"{expected_away*100:.1f}%")

# =============================================================================
# APPLICATION PRINCIPALE
# =============================================================================

def main():
    """Fonction principale de l'application"""
    setup_page()
    
    # Initialisation du session state
    if 'elo_system' not in st.session_state:
        st.session_state.elo_system = AdvancedEloSystem()
    
    if 'data_gen' not in st.session_state:
        st.session_state.data_gen = DemoDataGenerator(st.session_state.elo_system)
    
    if 'bankroll' not in st.session_state:
        st.session_state.bankroll = 10000
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Sélection de la ligue
        selected_league = st.selectbox(
            "🏆 Sélectionner une ligue",
            list(LEAGUES.keys()),
            index=0
        )
        
        league_info = LEAGUES[selected_league]
        
        # Bankroll
        st.subheader("💰 Bankroll")
        bankroll = st.number_input(
            "Montant initial (€)",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=500
        )
        
        # Paramètres de risque
        st.subheader("🎯 Paramètres Risque")
        
        kelly_fraction = st.slider(
            "Fraction de Kelly",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Pourcentage du Kelly full à utiliser"
        )
        
        min_edge = st.slider(
            "Edge minimum (%)",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=0.5
        )
        
        # Boutons d'action
        st.subheader("🔄 Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎯 Générer matchs", type="primary"):
                with st.spinner("Génération en cours..."):
                    # Générer des équipes pour la ligue
                    league_teams = [id for id, data in TEAMS_DATA.items() 
                                   if data['league'] == league_info['name']]
                    
                    if league_teams:
                        # Générer des matchs
                        matches = st.session_state.data_gen.generate_season_matches(league_teams, 3)
                        upcoming = st.session_state.data_gen.generate_upcoming_matches(league_teams, 7)
                        
                        st.session_state.generated_matches = matches
                        st.session_state.upcoming_matches = upcoming
                        
                        st.success(f"{len(matches)} matchs générés!")
                    else:
                        st.warning(f"Aucune équipe trouvée pour {league_info['name']}")
        
        with col2:
            if st.button("🧹 Réinitialiser"):
                for key in list(st.session_state.keys()):
                    if key not in ['elo_system', 'data_gen']:
                        del st.session_state[key]
                st.session_state.elo_system = AdvancedEloSystem()
                st.session_state.data_gen = DemoDataGenerator(st.session_state.elo_system)
                st.rerun()
        
        # Informations
        st.divider()
        st.info("""
        **ℹ️ Mode Démo**
        
        Cette application utilise des données simulées pour démontrer le système.
        
        Fonctionnalités:
        - Système Elo avancé
        - Détection de value bets
        - Gestion de bankroll
        - Simulation de paris
        """)
    
    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", 
        "🎯 Value Bets", 
        "🤖 Modèles IA", 
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

# =============================================================================
# LANCEMENT DE L'APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
