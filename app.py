# app.py - Système de Paris Football Simplifié

import pandas as pd
import numpy as np
import streamlit as st
import requests
from datetime import datetime, date, timedelta
import math

# =============================================================================
# CONFIGURATION
# =============================================================================

class APIConfig:
    """Configuration"""
    API_FOOTBALL_KEY: str = "249b3051eCA063F0e381609128c00d7d"
    API_FOOTBALL_URL: str = "https://v3.football.api-sports.io"

# =============================================================================
# CLIENT API SIMPLIFIÉ
# =============================================================================

class FootballDataClient:
    """Client API simplifié"""
    
    def __init__(self):
        self.config = APIConfig()
        self.session = requests.Session()
        self.session.headers.update({
            'x-apisports-key': self.config.API_FOOTBALL_KEY
        })
    
    def test_connection(self) -> bool:
        """Teste la connexion"""
        try:
            url = f"{self.config.API_FOOTBALL_URL}/status"
            response = self.session.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False

# =============================================================================
# SYSTÈME DE PRÉDICTION SIMPLIFIÉ
# =============================================================================

class PredictionSystem:
    """Système de prédiction simplifié"""
    
    def __init__(self):
        self.team_ratings = {}
    
    def predict_match(self, home_form: float, away_form: float, 
                     home_advantage: bool = True) -> Dict:
        """Prédit un match de manière simplifiée"""
        
        # Calcul basique
        home_rating = 1500 + (home_form - 5) * 50
        away_rating = 1500 + (away_form - 5) * 50
        
        if home_advantage:
            home_rating += 70
        
        # Probabilités
        rating_diff = home_rating - away_rating
        home_win_prob = 1 / (1 + 10 ** (-rating_diff / 400))
        draw_prob = 0.25 * (1 - abs(rating_diff) / 800)
        away_win_prob = 1 - home_win_prob - draw_prob
        
        return {
            'home_win': max(0, min(1, home_win_prob)),
            'draw': max(0, min(1, draw_prob)),
            'away_win': max(0, min(1, away_win_prob))
        }

# =============================================================================
# BET MANAGER SIMPLIFIÉ
# =============================================================================

class BetManager:
    """Gestionnaire de paris simplifié"""
    
    def __init__(self, initial_bankroll: float = 10000.0):
        self.bankroll = initial_bankroll
        self.bets = []
    
    def place_bet(self, match: str, selection: str, 
                  stake: float, odds: float) -> bool:
        """Place un pari simple"""
        if stake > self.bankroll:
            return False
        
        bet = {
            'id': len(self.bets) + 1,
            'timestamp': datetime.now(),
            'match': match,
            'selection': selection,
            'stake': stake,
            'odds': odds,
            'status': 'pending'
        }
        
        self.bankroll -= stake
        self.bets.append(bet)
        return True

# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

def setup_interface():
    """Configure l'interface"""
    st.set_page_config(
        page_title="Système Paris Football",
        page_icon="⚽",
        layout="wide"
    )
    
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-title">⚽ SYSTÈME DE PARIS FOOTBALL</div>', unsafe_allow_html=True)

def main():
    """Application principale"""
    setup_interface()
    
    # Initialisation
    if 'api_client' not in st.session_state:
        st.session_state.api_client = FootballDataClient()
    
    if 'prediction_system' not in st.session_state:
        st.session_state.prediction_system = PredictionSystem()
    
    if 'bet_manager' not in st.session_state:
        st.session_state.bet_manager = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURATION")
        
        if st.button("🔗 Tester connexion API"):
            if st.session_state.api_client.test_connection():
                st.success("✅ Connecté")
            else:
                st.error("❌ Déconnecté")
        
        if st.session_state.bet_manager is None:
            initial = st.number_input("Bankroll initial (€)", 1000.0, 100000.0, 10000.0, 1000.0)
            if st.button("💰 Initialiser"):
                st.session_state.bet_manager = BetManager(initial)
                st.success(f"Bankroll: €{initial:,.2f}")
                st.rerun()
        else:
            st.metric("💶 Bankroll", f"€{st.session_state.bet_manager.bankroll:,.2f}")
    
    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Accueil", 
        "🎯 Analyse Match", 
        "💰 Paris", 
        "📊 Historique"
    ])
    
    with tab1:
        display_home()
    
    with tab2:
        display_analysis()
    
    with tab3:
        display_betting()
    
    with tab4:
        display_history()

def display_home():
    """Page d'accueil"""
    st.header("🏠 ACCUEIL")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.bet_manager:
            st.metric("💰 Bankroll", f"€{st.session_state.bet_manager.bankroll:,.2f}")
        else:
            st.metric("💰 Bankroll", "Non initialisé")
    
    with col2:
        if st.session_state.api_client.test_connection():
            st.metric("🌐 API", "✅ Connectée")
        else:
            st.metric("🌐 API", "❌ Déconnectée")
    
    with col3:
        if st.session_state.bet_manager:
            total_bets = len(st.session_state.bet_manager.bets)
            st.metric("📊 Paris", total_bets)
        else:
            st.metric("📊 Paris", "0")
    
    st.divider()
    
    # Analyse rapide
    st.subheader("⚡ Analyse Rapide")
    
    with st.form("quick_analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            home_team = st.text_input("Équipe domicile", "Paris SG")
            home_form = st.slider("Forme domicile", 1, 10, 7)
        
        with col2:
            away_team = st.text_input("Équipe extérieur", "Marseille")
            away_form = st.slider("Forme extérieur", 1, 10, 6)
        
        if st.form_submit_button("🔍 Analyser"):
            prediction = st.session_state.prediction_system.predict_match(
                home_form, away_form
            )
            
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric(f"🏠 {home_team}", f"{prediction['home_win']*100:.1f}%")
            
            with col4:
                st.metric("🤝 Nul", f"{prediction['draw']*100:.1f}%")
            
            with col5:
                st.metric(f"⚽ {away_team}", f"{prediction['away_win']*100:.1f}%")

def display_analysis():
    """Analyse détaillée d'un match"""
    st.header("🎯 ANALYSE DE MATCH")
    
    st.info("Entrez les détails pour une analyse complète")
    
    with st.form("detailed_analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Équipe Domicile")
            home_team = st.text_input("Nom", "Manchester City")
            home_form = st.slider("Forme (1-10)", 1, 10, 7)
            home_attack = st.number_input("Attaque", 0.0, 5.0, 2.3, 0.1)
            home_defense = st.number_input("Défense", 0.0, 5.0, 0.8, 0.1)
        
        with col2:
            st.subheader("⚽ Équipe Extérieur")
            away_team = st.text_input("Nom", "Liverpool")
            away_form = st.slider("Forme (1-10)", 1, 10, 6)
            away_attack = st.number_input("Attaque", 0.0, 5.0, 1.9, 0.1)
            away_defense = st.number_input("Défense", 0.0, 5.0, 1.2, 0.1)
        
        col3, col4 = st.columns(2)
        with col3:
            is_neutral = st.checkbox("Terrain neutre")
        
        with col4:
            importance = st.selectbox("Importance", 
                                    ["Normal", "Coupe", "Dernière journée", "Finale"])
        
        if st.form_submit_button("🚀 ANALYSER", type="primary"):
            # Prédiction
            prediction = st.session_state.prediction_system.predict_match(
                home_form, away_form, not is_neutral
            )
            
            # Afficher résultats
            st.subheader("📊 RÉSULTATS")
            
            col5, col6, col7 = st.columns(3)
            
            with col5:
                st.markdown(f"""
                <div style="background: #E3F2FD; padding: 20px; border-radius: 10px; text-align: center;">
                <h3>🏠 VICTOIRE</h3>
                <h2 style="color: #1E88E5;">{prediction['home_win']*100:.1f}%</h2>
                <p>Cote équivalente: {1/prediction['home_win']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                st.markdown(f"""
                <div style="background: #F3E5F5; padding: 20px; border-radius: 10px; text-align: center;">
                <h3>🤝 NUL</h3>
                <h2 style="color: #9C27B0;">{prediction['draw']*100:.1f}%</h2>
                <p>Cote équivalente: {1/prediction['draw']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col7:
                st.markdown(f"""
                <div style="background: #E8F5E9; padding: 20px; border-radius: 10px; text-align: center;">
                <h3>⚽ VICTOIRE</h3>
                <h2 style="color: #4CAF50;">{prediction['away_win']*100:.1f}%</h2>
                <p>Cote équivalente: {1/prediction['away_win']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Score prédit
            st.subheader("🎯 SCORE PRÉDIT")
            
            expected_home = (home_attack + away_defense) / 2
            expected_away = (away_attack + home_defense) / 2
            
            if importance in ["Finale", "Dernière journée"]:
                expected_home *= 0.95
                expected_away *= 0.95
            
            predicted_home = round(expected_home)
            predicted_away = round(expected_away)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 30px; border-radius: 15px; text-align: center; color: white;">
            <h1 style="font-size: 4rem;">{predicted_home} - {predicted_away}</h1>
            <p>Score le plus probable</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommandations
            st.subheader("💰 RECOMMANDATIONS")
            
            # Simuler des cotes de bookmaker
            market_home = 1/prediction['home_win'] * 0.9
            market_draw = 1/prediction['draw'] * 0.9
            market_away = 1/prediction['away_win'] * 0.9
            
            edge_home = (prediction['home_win'] * market_home) - 1
            edge_draw = (prediction['draw'] * market_draw) - 1
            edge_away = (prediction['away_win'] * market_away) - 1
            
            best_edge = max(edge_home, edge_draw, edge_away)
            
            if best_edge > 0.02:
                if best_edge == edge_home:
                    st.success(f"✅ MEILLEURE OPPORTUNITÉ: {home_team} @ {market_home:.2f} (Edge: {edge_home*100:.1f}%)")
                elif best_edge == edge_draw:
                    st.success(f"✅ MEILLEURE OPPORTUNITÉ: Match Nul @ {market_draw:.2f} (Edge: {edge_draw*100:.1f}%)")
                else:
                    st.success(f"✅ MEILLEURE OPPORTUNITÉ: {away_team} @ {market_away:.2f} (Edge: {edge_away*100:.1f}%)")
            else:
                st.warning("⚠️ Aucune opportunité significative détectée")

def display_betting():
    """Interface de paris"""
    st.header("💰 PLACER UN PARI")
    
    if st.session_state.bet_manager is None:
        st.warning("Veuillez d'abord initialiser le bankroll dans la sidebar.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        match_name = st.text_input("Match", "Paris SG vs Marseille")
        selection = st.selectbox("Sélection", ["1", "N", "2"])
        odds = st.number_input("Cote", 1.01, 100.0, 2.0, 0.01)
    
    with col2:
        bankroll = st.session_state.bet_manager.bankroll
        st.metric("💶 Bankroll disponible", f"€{bankroll:,.2f}")
        
        stake_method = st.radio("Méthode de mise", 
                               ["Montant fixe", "Pourcentage"])
        
        if stake_method == "Montant fixe":
            stake = st.number_input("Mise (€)", 1.0, bankroll, 100.0, 10.0)
        else:
            percent = st.slider("% du bankroll", 0.5, 10.0, 2.0, 0.5)
            stake = bankroll * (percent / 100)
            st.write(f"**Mise:** €{stake:,.2f} ({percent}%)")
    
    potential_return = stake * odds
    potential_profit = potential_return - stake
    
    st.metric("📈 Retour potentiel", f"€{potential_return:,.2f}")
    st.metric("💰 Profit potentiel", f"€{potential_profit:,.2f}")
    
    if st.button("✅ PLACER LE PARI", type="primary"):
        success = st.session_state.bet_manager.place_bet(
            match_name, selection, stake, odds
        )
        
        if success:
            st.success(f"""
            ✅ Pari placé avec succès !
            - Mise: €{stake:,.2f}
            - Bankroll restant: €{st.session_state.bet_manager.bankroll:,.2f}
            """)
            st.rerun()
        else:
            st.error("❌ Erreur: Bankroll insuffisant")

def display_history():
    """Historique des paris"""
    st.header("📊 HISTORIQUE DES PARIS")
    
    if st.session_state.bet_manager is None:
        st.warning("Aucun bankroll initialisé")
        return
    
    bets = st.session_state.bet_manager.bets
    
    if not bets:
        st.info("Aucun pari enregistré")
        return
    
    # Afficher les paris
    for bet in bets:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{bet['match']}**")
                st.write(f"Sélection: {bet['selection']} @ {bet['odds']:.2f}")
                st.write(f"Date: {bet['timestamp'].strftime('%d/%m/%Y %H:%M')}")
            
            with col2:
                st.write(f"**Mise:** €{bet['stake']:,.2f}")
            
            with col3:
                st.write(f"**Statut:** {bet['status']}")
            
            st.divider()
    
    # Statistiques
    total_staked = sum(b['stake'] for b in bets)
    st.metric("💵 Total misé", f"€{total_staked:,.2f}")

# =============================================================================
# LANCEMENT
# =============================================================================

if __name__ == "__main__":
    main()
