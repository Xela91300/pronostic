# tipser_pro.py - Système Professionnel de Pronostics Football
# Version Pro Simplifiée (Sans Plotly)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import random
import time

# =============================================================================
# CONFIGURATION AVANCÉE
# =============================================================================

class ProConfig:
    """Configuration professionnelle"""
    
    @staticmethod
    def get_default_matches():
        """Retourne des matchs par défaut"""
        today = datetime.now()
        return [
            {
                'id': 1,
                'home': 'Paris SG',
                'away': 'Marseille',
                'league': 'Ligue 1',
                'time': '20:00',
                'date': today.strftime('%d/%m/%Y'),
                'home_odds': 1.65,
                'draw_odds': 3.80,
                'away_odds': 4.50,
                'prediction': '1',
                'confidence': 72,
                'value': 8.2,
                'venue': 'Parc des Princes'
            },
            {
                'id': 2,
                'home': 'Lyon',
                'away': 'Monaco',
                'league': 'Ligue 1',
                'time': '18:00',
                'date': (today + timedelta(days=1)).strftime('%d/%m/%Y'),
                'home_odds': 2.10,
                'draw_odds': 3.40,
                'away_odds': 3.20,
                'prediction': '1X',
                'confidence': 65,
                'value': 5.8,
                'venue': 'Groupama Stadium'
            },
            {
                'id': 3,
                'home': 'Real Madrid',
                'away': 'Barcelona',
                'league': 'La Liga',
                'time': '21:00',
                'date': (today + timedelta(days=2)).strftime('%d/%m/%Y'),
                'home_odds': 2.30,
                'draw_odds': 3.50,
                'away_odds': 2.90,
                'prediction': 'Over 2.5',
                'confidence': 68,
                'value': 7.1,
                'venue': 'Santiago Bernabéu'
            }
        ]

# =============================================================================
# INTELLIGENCE ARTIFICIELLE
# =============================================================================

class AIPredictor:
    """Système de prédiction simplifié"""
    
    def predict_match(self, home_team, away_team):
        """Prédit le résultat d'un match"""
        # Simulation simple basée sur le nom des équipes
        home_factor = self._get_team_factor(home_team)
        away_factor = self._get_team_factor(away_team)
        
        total = home_factor + away_factor
        home_prob = (home_factor / total) * 100 if total > 0 else 50
        away_prob = (away_factor / total) * 100 if total > 0 else 50
        
        # Ajouter une probabilité de match nul
        draw_prob = 100 - home_prob - away_prob
        draw_prob = max(20, min(40, draw_prob))
        
        # Réajuster
        adjustment = (100 - (home_prob + away_prob + draw_prob)) / 3
        home_prob += adjustment
        away_prob += adjustment
        draw_prob += adjustment
        
        return {
            'home_win': round(home_prob, 1),
            'draw': round(draw_prob, 1),
            'away_win': round(away_prob, 1),
            'expected_home': round(random.uniform(1.5, 2.5), 1),
            'expected_away': round(random.uniform(0.8, 1.8), 1),
            'confidence': round(random.uniform(60, 85), 1)
        }
    
    def _get_team_factor(self, team_name):
        """Donne un facteur basé sur le nom de l'équipe"""
        team_strengths = {
            'Paris SG': 9, 'Marseille': 7, 'Lyon': 6, 'Monaco': 6,
            'Real Madrid': 9, 'Barcelona': 8, 'Liverpool': 8, 'Manchester City': 9
        }
        
        # Recherche partielle
        for team, strength in team_strengths.items():
            if team.lower() in team_name.lower():
                return strength
        
        # Valeur par défaut basée sur la longueur du nom
        return len(team_name) / 10 + 5

# =============================================================================
# INTERFACE UTILISATEUR
# =============================================================================

def setup_page():
    """Configure la page Streamlit"""
    st.set_page_config(
        page_title="Tipser Pro | Pronostics Intelligents",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
    /* Thème principal */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Cartes */
    .match-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .match-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin: 2px;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        color: white;
    }
    
    .badge-danger {
        background: linear-gradient(135deg, #F44336 0%, #D32F2F 100%);
        color: white;
    }
    
    .badge-info {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
    }
    
    .badge-premium {
        background: linear-gradient(135deg, #FFD700 0%, #FFC107 100%);
        color: #333;
        font-weight: bold;
    }
    
    /* Métriques */
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    
    /* Boutons améliorés */
    .stButton > button {
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
    }
    
    /* Amélioration des onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    /* Scrollbar personnalisée */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    /* Amélioration des dataframes */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Tooltips */
    [title] {
        position: relative;
    }
    
    [title]:hover:after {
        content: attr(title);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: #333;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
        white-space: nowrap;
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# APPLICATION PRINCIPALE
# =============================================================================

def main():
    """Application principale"""
    
    # Configuration
    setup_page()
    
    # Initialisation session state
    if 'selected_match' not in st.session_state:
        st.session_state.selected_match = None
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'dashboard'
    if 'bankroll' not in st.session_state:
        st.session_state.bankroll = 1000
    if 'ai_predictor' not in st.session_state:
        st.session_state.ai_predictor = AIPredictor()
    
    # En-tête
    display_header()
    
    # Sidebar
    with st.sidebar:
        display_sidebar()
    
    # Contenu principal
    if st.session_state.view_mode == 'dashboard':
        display_dashboard()
    elif st.session_state.view_mode == 'selection':
        display_selection()
    elif st.session_state.view_mode == 'analysis':
        display_analysis()
    elif st.session_state.view_mode == 'portfolio':
        display_portfolio()

def display_header():
    """Affiche l'en-tête"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.image("⚽", width=80)
    
    with col2:
        st.markdown("""
        <div class="main-header">
            <h1>⚽ TIPSER PRO</h1>
            <h3>Système Intelligent de Pronostics Football</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.metric("Bankroll", f"€{st.session_state.bankroll}")
        st.caption("Version 2.0 PRO")

def display_sidebar():
    """Affiche la sidebar"""
    st.sidebar.title("🎯 Navigation")
    
    # Menu
    menu_options = {
        "📊 Dashboard": "dashboard",
        "🔍 Sélection": "selection",
        "📈 Analyse": "analysis",
        "💰 Portfolio": "portfolio"
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
        key="filter_leagues"
    )
    
    min_confidence = st.sidebar.slider(
        "Confiance minimum",
        0, 100, 60
    )
    
    st.sidebar.divider()
    
    # Stats rapides
    st.sidebar.subheader("📈 Statistiques")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Tips", "24")
        st.metric("ROI", "+14.5%")
    with col2:
        st.metric("Hit Rate", "68%")
        st.metric("Value", "€145")
    
    # Bouton actualisation
    if st.sidebar.button("🔄 Actualiser", use_container_width=True):
        st.rerun()

def display_dashboard():
    """Affiche le dashboard"""
    st.title("📊 Dashboard Tipser Pro")
    
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
            <h3>📈 ROI 30j</h3>
            <h2>+12.4%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h3>✅ Hit Rate</h3>
            <h2>67.8%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-box">
            <h3>💰 Bankroll</h3>
            <h2>€1,145</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Graphiques simplifiés
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Performance")
        
        # Graphique simple avec bar chart
        months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun']
        roi = [2.1, 5.3, 8.7, 10.2, 12.4, 14.5]
        
        chart_data = pd.DataFrame({
            'Mois': months,
            'ROI (%)': roi
        })
        
        st.bar_chart(chart_data.set_index('Mois'))
    
    with col2:
        st.subheader("🎯 Distribution")
        
        # Camembert simple
        labels = ['1N2', 'Over/Under', 'BTTS', 'Handicap']
        values = [45, 25, 15, 15]
        
        pie_data = pd.DataFrame({
            'Type': labels,
            'Pourcentage': values
        })
        
        st.dataframe(pie_data, use_container_width=True)
    
    # Derniers tips
    st.divider()
    st.subheader("🎯 Derniers Tips")
    
    tips = [
        {"match": "PSG vs Marseille", "tip": "Over 2.5", "odds": 1.85, "stake": "3%", "status": "✅ Gagné"},
        {"match": "Real Madrid vs Barca", "tip": "1", "odds": 2.10, "stake": "2%", "status": "✅ Gagné"},
        {"match": "Liverpool vs City", "tip": "BTTS Yes", "odds": 1.65, "stake": "4%", "status": "⚪ En cours"},
    ]
    
    for tip in tips:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
            
            with col1:
                st.write(f"**{tip['match']}**")
            
            with col2:
                st.code(tip['tip'])
            
            with col3:
                st.metric("Cote", tip['odds'])
            
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

def display_selection():
    """Affiche la sélection des matchs"""
    st.title("🔍 Sélection des Matchs")
    
    # Filtres
    with st.expander("🎯 Filtres Avancés", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_filter = st.selectbox(
                "Période",
                ["Aujourd'hui", "Demain", "Week-end", "7 jours"],
                key="date_filter"
            )
        
        with col2:
            league_filter = st.multiselect(
                "Ligues",
                ["Ligue 1", "Premier League", "La Liga", "Bundesliga", "Serie A"],
                default=["Ligue 1", "Premier League"],
                key="league_filter"
            )
        
        with col3:
            min_odds = st.number_input("Cote min", 1.2, 10.0, 1.5, 0.1)
            max_odds = st.number_input("Cote max", 1.5, 20.0, 3.0, 0.1)
    
    # Bouton recherche
    if st.button("🔍 Rechercher Matchs", type="primary", use_container_width=True):
        with st.spinner("Recherche en cours..."):
            time.sleep(1)
            display_match_cards()

def display_match_cards():
    """Affiche les cartes de match"""
    matches = ProConfig.get_default_matches()
    
    st.subheader(f"📋 Matchs Trouvés ({len(matches)})")
    
    for match in matches:
        # Vérifier les filtres
        if match['league'] not in st.session_state.get('league_filter', []):
            if st.session_state.get('league_filter'):
                continue
        
        # Carte de match
        st.markdown(f"""
        <div class="match-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4>{match['home']} vs {match['away']}</h4>
                    <p>🏆 {match['league']} | 📅 {match['date']} | ⏰ {match['time']}</p>
                </div>
                <div>
                    <span class="badge badge-success">Confiance: {match['confidence']}%</span>
                    <span class="badge badge-premium">Valeur: +{match['value']}%</span>
                </div>
            </div>
            
            <div style="display: flex; justify-content: space-around; margin: 20px 0;">
                <div style="text-align: center;">
                    <h5>1</h5>
                    <h3>{match['home_odds']}</h3>
                </div>
                <div style="text-align: center;">
                    <h5>N</h5>
                    <h3>{match['draw_odds']}</h3>
                </div>
                <div style="text-align: center;">
                    <h5>2</h5>
                    <h3>{match['away_odds']}</h3>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 20px;">
                <h4>🎯 Prédiction: {match['prediction']}</h4>
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
            if st.button(f"💰 Ajouter", key=f"add_{match['id']}", use_container_width=True):
                st.success(f"✅ Match ajouté au portfolio!")
        
        st.divider()

def display_analysis():
    """Affiche l'analyse d'un match"""
    if not st.session_state.selected_match:
        st.warning("Veuillez sélectionner un match d'abord")
        st.session_state.view_mode = 'selection'
        st.rerun()
        return
    
    match = st.session_state.selected_match
    
    # Bouton retour
    if st.button("← Retour à la sélection"):
        st.session_state.view_mode = 'selection'
        st.rerun()
    
    # Titre
    st.title(f"📈 Analyse: {match['home']} vs {match['away']}")
    
    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Vue d'ensemble",
        "🎯 Prédictions",
        "💰 Opportunités",
        "📈 Statistiques"
    ])
    
    with tab1:
        display_overview(match)
    
    with tab2:
        display_predictions(match)
    
    with tab3:
        display_value_bets(match)
    
    with tab4:
        display_stats(match)

def display_overview(match):
    """Affiche la vue d'ensemble"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏟️ Informations")
        
        info = {
            "Ligue": match['league'],
            "Date": match['date'],
            "Heure": match['time'],
            "Stade": match['venue'],
            "Météo": "☀️ 18°C",
            "Arbitre": "M. Turpin"
        }
        
        for key, value in info.items():
            st.metric(key, value)
    
    with col2:
        st.subheader("📊 Cotes Marché")
        
        odds_data = {
            "1 - Victoire domicile": match['home_odds'],
            "N - Match nul": match['draw_odds'],
            "2 - Victoire extérieur": match['away_odds'],
            "Over 2.5 goals": 1.85,
            "BTTS Oui": 1.65,
            "1X Double Chance": 1.35
        }
        
        for market, odd in odds_data.items():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(market)
            with col_b:
                st.write(f"**{odd}**")
    
    # Forme des équipes
    st.divider()
    st.subheader("📈 Forme Récente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{match['home']}**")
        st.write("Derniers 5 matchs: W W D L W")
        
        # Barres de progression pour la forme
        st.write("Forme:")
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        with col_a:
            st.success("W")
        with col_b:
            st.success("W")
        with col_c:
            st.warning("D")
        with col_d:
            st.error("L")
        with col_e:
            st.success("W")
        
        st.metric("Buts marqués (moy.)", "2.1")
        st.metric("Buts encaissés (moy.)", "0.8")
    
    with col2:
        st.write(f"**{match['away']}**")
        st.write("Derniers 5 matchs: D W W D L")
        
        # Barres de progression pour la forme
        st.write("Forme:")
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        with col_a:
            st.warning("D")
        with col_b:
            st.success("W")
        with col_c:
            st.success("W")
        with col_d:
            st.warning("D")
        with col_e:
            st.error("L")
        
        st.metric("Buts marqués (moy.)", "1.6")
        st.metric("Buts encaissés (moy.)", "1.2")

def display_predictions(match):
    """Affiche les prédictions"""
    st.subheader("🤖 Prédictions IA")
    
    # Obtenir les prédictions
    predictions = st.session_state.ai_predictor.predict_match(
        match['home'], match['away']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Probabilités")
        
        # Barres de progression pour les probabilités
        st.write(f"**{match['home']} gagne**")
        st.progress(predictions['home_win'] / 100)
        st.caption(f"{predictions['home_win']}%")
        
        st.write("**Match nul**")
        st.progress(predictions['draw'] / 100)
        st.caption(f"{predictions['draw']}%")
        
        st.write(f"**{match['away']} gagne**")
        st.progress(predictions['away_win'] / 100)
        st.caption(f"{predictions['away_win']}%")
    
    with col2:
        st.markdown("### ⚽ Score Attend")
        
        # Afficher le score attendu
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px;">
            <h1 style="font-size: 4rem; margin: 0;">
                {predictions['expected_home']} - {predictions['expected_away']}
            </h1>
            <p style="color: #666; font-size: 1.2rem;">Score attendu (xG)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        # Métriques
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total buts", round(predictions['expected_home'] + predictions['expected_away'], 1))
        with col_b:
            if predictions['expected_home'] > predictions['expected_away']:
                st.metric("Favori", match['home'])
            else:
                st.metric("Favori", match['away'])
        with col_c:
            st.metric("Confiance", f"{predictions['confidence']}%")
    
    # Recommandation
    st.divider()
    st.subheader("🎯 Recommandation")
    
    if predictions['home_win'] > 55:
        recommendation = f"✅ {match['home']} gagne"
        color = "success"
    elif predictions['away_win'] > 55:
        recommendation = f"✅ {match['away']} gagne"
        color = "success"
    else:
        recommendation = "⚪ Match nul ou double chance"
        color = "info"
    
    st.markdown(f"""
    <div style="background: {'#4CAF50' if color == 'success' else '#2196F3'}; 
                color: white; 
                padding: 20px; 
                border-radius: 10px; 
                text-align: center;">
        <h2>{recommendation}</h2>
        <p>Probabilité: {max(predictions['home_win'], predictions['draw'], predictions['away_win'])}%</p>
    </div>
    """, unsafe_allow_html=True)

def display_value_bets(match):
    """Affiche les opportunités de value bet"""
    st.subheader("💰 Détection de Value Bets")
    
    # Simuler des value bets
    value_bets = [
        {
            'bookmaker': 'Bet365',
            'market': '1',
            'odds': match['home_odds'],
            'fair_odds': round(1 / (match['confidence'] / 100), 2),
            'value': match['value']
        },
        {
            'bookmaker': 'Unibet',
            'market': 'Over 2.5',
            'odds': 1.85,
            'fair_odds': 1.72,
            'value': 7.6
        }
    ]
    
    if value_bets:
        st.success(f"🎯 {len(value_bets)} opportunités détectées!")
        
        for bet in value_bets:
            with st.container():
                st.markdown(f"""
                <div style="background: white; padding: 15px; border-radius: 10px; border-left: 5px solid #FFD700; margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4>{bet['bookmaker']} - {bet['market']}</h4>
                            <p>Cote juste: {bet['fair_odds']}</p>
                        </div>
                        <div>
                            <span class="badge badge-premium">+{bet['value']}% valeur</span>
                        </div>
                    </div>
                    
                    <div style="display: flex; justify-content: space-between; margin-top: 15px;">
                        <div>
                            <h5>Cote offerte</h5>
                            <h2>{bet['odds']}</h2>
                        </div>
                        <div>
                            <h5>Valeur</h5>
                            <h3>+{bet['value']}%</h3>
                        </div>
                        <div>
                            <h5>Recommandation</h5>
                            <h4>2-3% bankroll</h4>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ Aucune opportunité de value bet significative.")
    
    # Calculateur de mise
    st.divider()
    st.subheader("🏦 Calculateur de Mise")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bankroll = st.number_input("Bankroll (€)", 
                                  min_value=10, 
                                  max_value=10000, 
                                  value=st.session_state.bankroll)
    
    with col2:
        probability = st.slider("Probabilité réelle (%)", 0, 100, match['confidence'])
    
    with col3:
        odds = st.number_input("Cote", value=match['home_odds'], min_value=1.1, max_value=100.0, step=0.1)
    
    # Calcul Kelly
    if odds > 1 and probability > 0:
        kelly = ((probability/100) * (odds - 1) - (1 - probability/100)) / (odds - 1)
        kelly_percent = max(0, kelly * 100)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Fraction Kelly", f"{kelly_percent:.1f}%")
        with col_b:
            suggested_stake = (kelly_percent/100) * bankroll
            st.metric("Mise suggérée", f"€{suggested_stake:.2f}")

def display_stats(match):
    """Affiche les statistiques"""
    st.subheader("📈 Statistiques Comparatives")
    
    # Générer des statistiques aléatoires
    home_stats = generate_random_stats()
    away_stats = generate_random_stats()
    
    # Tableau comparatif
    comparison = {
        'Statistique': ['Forme récente', 'Victoires', 'Buts/match', 'Buts encaissés', 'Possession', 'Précision passes'],
        match['home']: [
            'W W D L W',
            f"{random.randint(5, 8)}/{random.randint(10, 15)}",
            f"{home_stats['goals_for']:.1f}",
            f"{home_stats['goals_against']:.1f}",
            f"{home_stats['possession']}%",
            f"{home_stats['pass_accuracy']}%"
        ],
        match['away']: [
            'D W W D L',
            f"{random.randint(3, 6)}/{random.randint(10, 15)}",
            f"{away_stats['goals_for']:.1f}",
            f"{away_stats['goals_against']:.1f}",
            f"{away_stats['possession']}%",
            f"{away_stats['pass_accuracy']}%"
        ]
    }
    
    df = pd.DataFrame(comparison)
    st.dataframe(df.set_index('Statistique'), use_container_width=True)
    
    # Métriques additionnelles
    st.divider()
    st.subheader("📊 Métriques Avancées")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("xG Différence", 
                 f"{home_stats['xg'] - away_stats['xg']:+.1f}",
                 f"en faveur de {match['home'] if home_stats['xg'] > away_stats['xg'] else match['away']}")
    
    with col2:
        st.metric("Danger offensif", 
                 f"{home_stats['shot_creation']} vs {away_stats['shot_creation']}",
                 "actions créatrices/match")
    
    with col3:
        st.metric("Solidité défensive", 
                 f"{home_stats['defense_rating']}/10 vs {away_stats['defense_rating']}/10",
                 "note défensive")

def generate_random_stats():
    """Génère des statistiques aléatoires"""
    return {
        'goals_for': round(random.uniform(1.2, 2.5), 1),
        'goals_against': round(random.uniform(0.8, 1.8), 1),
        'possession': random.randint(45, 65),
        'pass_accuracy': random.randint(75, 90),
        'xg': round(random.uniform(1.5, 2.3), 1),
        'shot_creation': random.randint(12, 25),
        'defense_rating': random.randint(6, 9)
    }

def display_portfolio():
    """Affiche le portfolio"""
    st.title("💰 Mon Portfolio")
    
    # Résumé
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Paris actifs", "3", "+1")
    with col2:
        st.metric("Investissement", "€150")
    with col3:
        st.metric("Gains potentiels", "€285")
    with col4:
        st.metric("ROI", "+90%")
    
    st.divider()
    
    # Paris en cours
    st.subheader("📈 Paris Actifs")
    
    active_bets = [
        {"match": "PSG vs Marseille", "type": "Over 2.5", "odds": 1.85, "stake": "€50", "potentiel": "€92.5"},
        {"match": "Real Madrid vs Barca", "type": "1", "odds": 2.10, "stake": "€60", "potentiel": "€126"},
        {"match": "Liverpool vs City", "type": "BTTS Yes", "odds": 1.65, "stake": "€40", "potentiel": "€66"}
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
                st.metric("Potentiel", bet['potentiel'])
            st.divider()
    
    # Historique
    with st.expander("📊 Historique des Paris"):
        history = {
            "Date": ["15/03", "14/03", "13/03", "12/03", "11/03"],
            "Match": ["PSG vs Lille", "Marseille vs Lyon", "Real vs Atletico", "City vs Arsenal", "Bayern vs Dortmund"],
            "Type": ["1", "Over 2.5", "BTTS Yes", "2", "1 & Over 2.5"],
            "Cote": [1.65, 1.85, 1.70, 2.40, 2.10],
            "Mise": ["€30", "€40", "€25", "€20", "€35"],
            "Résultat": ["✅ +€19.5", "✅ +€34", "✅ +€17.5", "❌ -€20", "✅ +€38.5"]
        }
        
        df = pd.DataFrame(history)
        st.dataframe(df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
