# app.py - Système de Paris Football Simplifié

import pandas as pd
import numpy as np
import streamlit as st
import requests
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple  # Ajout de Dict
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
        
        # S'assurer que les probabilités sont entre 0 et 1
        home_win_prob = max(0.0, min(1.0, home_win_prob))
        draw_prob = max(0.0, min(1.0, draw_prob))
        away_win_prob = max(0.0, min(1.0, away_win_prob))
        
        return {
            'home_win': home_win_prob,
            'draw': draw_prob,
            'away_win': away_win_prob,
            'home_rating': home_rating,
            'away_rating': away_rating
        }

# =============================================================================
# BET MANAGER SIMPLIFIÉ
# =============================================================================

class BetManager:
    """Gestionnaire de paris simplifié"""
    
    def __init__(self, initial_bankroll: float = 10000.0):
        self.bankroll = initial_bankroll
        self.bets = []
        self.initial_bankroll = initial_bankroll
    
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
    
    def settle_bet(self, bet_id: int, result: str) -> bool:
        """Règle un pari"""
        for bet in self.bets:
            if bet['id'] == bet_id:
                bet['status'] = 'settled'
                bet['result'] = result
                bet['settled_at'] = datetime.now()
                
                if result == 'win':
                    winnings = bet['stake'] * bet['odds']
                    self.bankroll += winnings
                elif result == 'void':
                    self.bankroll += bet['stake']
                
                return True
        return False

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
    .prediction-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    .bet-card {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
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
                st.success("✅ API Connectée")
            else:
                st.error("❌ API Déconnectée")
        
        st.divider()
        
        if st.session_state.bet_manager is None:
            initial = st.number_input("Bankroll initial (€)", 1000.0, 100000.0, 10000.0, 1000.0)
            if st.button("💰 Initialiser Bankroll", type="primary"):
                st.session_state.bet_manager = BetManager(initial)
                st.success(f"Bankroll initialisé: €{initial:,.2f}")
                st.rerun()
        else:
            current = st.session_state.bet_manager.bankroll
            initial = st.session_state.bet_manager.initial_bankroll
            profit = current - initial
            
            st.metric("💶 Bankroll Actuel", f"€{current:,.2f}")
            
            if profit >= 0:
                st.metric("📈 Profit", f"€{profit:,.2f}", delta="€{:+,.0f}".format(profit))
            else:
                st.metric("📉 Perte", f"€{abs(profit):,.2f}", delta="€{:+,.0f}".format(profit))
    
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
    st.header("🏠 BIENVENUE DANS VOTRE SYSTÈME DE PARIS")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.bet_manager:
            st.metric("💰 Bankroll", f"€{st.session_state.bet_manager.bankroll:,.2f}")
        else:
            st.metric("💰 Bankroll", "Non initialisé")
    
    with col2:
        api_status = st.session_state.api_client.test_connection()
        if api_status:
            st.metric("🌐 API Football", "✅ Connectée")
        else:
            st.metric("🌐 API Football", "❌ Déconnectée")
    
    with col3:
        if st.session_state.bet_manager:
            total_bets = len(st.session_state.bet_manager.bets)
            st.metric("📊 Paris Totaux", total_bets)
        else:
            st.metric("📊 Paris Totaux", "0")
    
    st.divider()
    
    # Analyse rapide
    st.subheader("⚡ Analyse Rapide de Match")
    
    with st.form("quick_analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            home_team = st.text_input("Équipe domicile", "Paris SG")
            home_form = st.slider("Forme domicile (1-10)", 1, 10, 7)
        
        with col2:
            away_team = st.text_input("Équipe extérieur", "Marseille")
            away_form = st.slider("Forme extérieur (1-10)", 1, 10, 6)
        
        home_advantage = st.checkbox("Avantage terrain domicile", value=True)
        
        if st.form_submit_button("🔍 Analyser ce match"):
            prediction = st.session_state.prediction_system.predict_match(
                home_form, away_form, home_advantage
            )
            
            st.markdown(f"""
            <div class="prediction-card">
                <h3>📊 Prédictions pour {home_team} vs {away_team}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric(f"🏠 {home_team}", f"{prediction['home_win']*100:.1f}%")
            
            with col4:
                st.metric("🤝 Match Nul", f"{prediction['draw']*100:.1f}%")
            
            with col5:
                st.metric(f"⚽ {away_team}", f"{prediction['away_win']*100:.1f}%")
            
            # Cotes équivalentes
            st.write("**Cotes équivalentes:**")
            col6, col7, col8 = st.columns(3)
            with col6:
                st.write(f"Victoire {home_team}: {1/prediction['home_win']:.2f}")
            with col7:
                st.write(f"Match nul: {1/prediction['draw']:.2f}")
            with col8:
                st.write(f"Victoire {away_team}: {1/prediction['away_win']:.2f}")

def display_analysis():
    """Analyse détaillée d'un match"""
    st.header("🎯 ANALYSE DÉTAILLÉE DE MATCH")
    
    st.info("Entrez les détails du match pour obtenir une analyse complète avec recommandations de paris.")
    
    with st.form("detailed_analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Équipe Domicile")
            home_team = st.text_input("Nom de l'équipe", "Manchester City", key="home_team_detailed")
            home_form = st.slider("Forme récente (1-10)", 1, 10, 7, key="home_form_detailed")
            home_attack = st.number_input("Attaque (buts/moy)", 0.0, 5.0, 2.3, 0.1, key="home_attack")
            home_defense = st.number_input("Défense (buts/moy)", 0.0, 5.0, 0.8, 0.1, key="home_defense")
        
        with col2:
            st.subheader("⚽ Équipe Extérieur")
            away_team = st.text_input("Nom de l'équipe", "Liverpool", key="away_team_detailed")
            away_form = st.slider("Forme récente (1-10)", 1, 10, 6, key="away_form_detailed")
            away_attack = st.number_input("Attaque (buts/moy)", 0.0, 5.0, 1.9, 0.1, key="away_attack")
            away_defense = st.number_input("Défense (buts/moy)", 0.0, 5.0, 1.2, 0.1, key="away_defense")
        
        col3, col4 = st.columns(2)
        with col3:
            is_neutral = st.checkbox("Terrain neutre", key="is_neutral")
            weather = st.selectbox("Conditions météo", 
                                 ["Bonnes", "Pluie", "Vent", "Froid", "Chaud"], 
                                 key="weather")
        
        with col4:
            importance = st.selectbox("Importance du match", 
                                    ["Normal", "Coupe", "Dernière journée", "Finale"], 
                                    key="importance")
            home_missing = st.number_input("Joueurs absents (dom)", 0, 10, 1, key="home_missing")
            away_missing = st.number_input("Joueurs absents (ext)", 0, 10, 2, key="away_missing")
        
        if st.form_submit_button("🚀 LANCER L'ANALYSE", type="primary"):
            # Prédiction
            prediction = st.session_state.prediction_system.predict_match(
                home_form, away_form, not is_neutral
            )
            
            # Ajustements selon les paramètres
            weather_factor = 0.95 if weather != "Bonnes" else 1.0
            missing_factor = max(0.7, 1.0 - (away_missing - home_missing) * 0.05)
            importance_factor = 0.95 if importance in ["Finale", "Dernière journée"] else 1.0
            
            adjusted_home = prediction['home_win'] * weather_factor * missing_factor * importance_factor
            adjusted_draw = prediction['draw'] * weather_factor * importance_factor
            adjusted_away = prediction['away_win'] * weather_factor * (1/missing_factor) * importance_factor
            
            # Normaliser
            total = adjusted_home + adjusted_draw + adjusted_away
            if total > 0:
                adjusted_home /= total
                adjusted_draw /= total
                adjusted_away /= total
            
            # Afficher résultats
            st.subheader("📊 RÉSULTATS DE L'ANALYSE")
            
            col5, col6, col7 = st.columns(3)
            
            with col5:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h3>🏠 {home_team}</h3>
                <h2 style="font-size: 2.5rem;">{adjusted_home*100:.1f}%</h2>
                <p>Cote équivalente: {1/adjusted_home:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h3>🤝 MATCH NUL</h3>
                <h2 style="font-size: 2.5rem;">{adjusted_draw*100:.1f}%</h2>
                <p>Cote équivalente: {1/adjusted_draw:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col7:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h3>⚽ {away_team}</h3>
                <h2 style="font-size: 2.5rem;">{adjusted_away*100:.1f}%</h2>
                <p>Cote équivalente: {1/adjusted_away:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Score prédit
            st.subheader("🎯 SCORE LE PLUS PROBABLE")
            
            expected_home = (home_attack + away_defense) / 2
            expected_away = (away_attack + home_defense) / 2
            
            # Ajustements
            expected_home *= weather_factor * missing_factor
            expected_away *= weather_factor * (1/missing_factor)
            
            predicted_home = round(expected_home)
            predicted_away = round(expected_away)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%); 
            padding: 30px; border-radius: 15px; text-align: center; color: white;">
            <h1 style="font-size: 4rem; margin: 0;">{predicted_home} - {predicted_away}</h1>
            <p style="font-size: 1.2rem;">Score le plus probable</p>
            <p>Buts attendus: {expected_home:.2f} - {expected_away:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommandations de paris
            st.subheader("💰 RECOMMANDATIONS DE PARIS")
            
            # Simuler des cotes de bookmaker (avec marge)
            market_home = 1/adjusted_home * 0.9
            market_draw = 1/adjusted_draw * 0.9
            market_away = 1/adjusted_away * 0.9
            
            edge_home = (adjusted_home * market_home) - 1
            edge_draw = (adjusted_draw * market_draw) - 1
            edge_away = (adjusted_away * market_away) - 1
            
            edges = [
                {'selection': f"{home_team} (1)", 'edge': edge_home, 'odds': market_home},
                {'selection': "Match Nul (X)", 'edge': edge_draw, 'odds': market_draw},
                {'selection': f"{away_team} (2)", 'edge': edge_away, 'odds': market_away}
            ]
            
            # Trier par edge décroissant
            edges.sort(key=lambda x: x['edge'], reverse=True)
            
            best_bet = edges[0]
            
            if best_bet['edge'] > 0.02:
                st.success(f"""
                🎯 **MEILLEURE OPPORTUNITÉ:** {best_bet['selection']}
                • **Cote estimée:** {best_bet['odds']:.2f}
                • **Edge (avantage):** {best_bet['edge']*100:.1f}%
                • **Recommandation:** {"✅ FORTE" if best_bet['edge'] > 0.05 else "⚠️ MODÉRÉE"}
                """)
                
                # Afficher les autres opportunités
                for bet in edges[1:]:
                    if bet['edge'] > 0.02:
                        with st.expander(f"Autre opportunité: {bet['selection']}"):
                            st.write(f"**Cote:** {bet['odds']:.2f}")
                            st.write(f"**Edge:** {bet['edge']*100:.1f}%")
            else:
                st.warning("⚠️ Aucune opportunité de value bet significative détectée avec les paramètres actuels.")

def display_betting():
    """Interface de paris"""
    st.header("💰 PLACER UN PARI")
    
    if st.session_state.bet_manager is None:
        st.warning("⚠️ Veuillez d'abord initialiser le bankroll dans la sidebar.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        match_name = st.text_input("Match", "Paris SG vs Marseille")
        selection = st.selectbox("Sélection", ["1 (Victoire domicile)", "N (Match nul)", "2 (Victoire extérieur)"])
        odds = st.number_input("Cote", 1.01, 100.0, 2.0, 0.01)
    
    with col2:
        bankroll = st.session_state.bet_manager.bankroll
        st.metric("💶 Bankroll disponible", f"€{bankroll:,.2f}")
        
        stake_method = st.radio("Méthode de mise", 
                               ["Montant fixe", "Pourcentage du bankroll"])
        
        if stake_method == "Montant fixe":
            stake = st.number_input("Mise (€)", 1.0, bankroll, 100.0, 10.0)
        else:
            percent = st.slider("% du bankroll", 0.5, 10.0, 2.0, 0.5)
            stake = bankroll * (percent / 100)
            st.write(f"**Mise calculée:** €{stake:,.2f} ({percent}%)")
    
    potential_return = stake * odds
    potential_profit = potential_return - stake
    
    st.metric("📈 Retour potentiel", f"€{potential_return:,.2f}")
    st.metric("💰 Profit potentiel", f"€{potential_profit:,.2f}")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("✅ PLACER LE PARI", type="primary", use_container_width=True):
            # Simplifier la sélection pour le pari
            if selection == "1 (Victoire domicile)":
                simple_selection = "1"
            elif selection == "N (Match nul)":
                simple_selection = "N"
            else:
                simple_selection = "2"
            
            success = st.session_state.bet_manager.place_bet(
                match_name, simple_selection, stake, odds
            )
            
            if success:
                st.success(f"""
                ✅ Pari placé avec succès !
                
                **Détails:**
                - Match: {match_name}
                - Sélection: {selection}
                - Mise: €{stake:,.2f}
                - Cote: {odds:.2f}
                - Bankroll restant: €{st.session_state.bet_manager.bankroll:,.2f}
                """)
                st.balloons()
                st.rerun()
            else:
                st.error("❌ Erreur: Bankroll insuffisant pour cette mise.")
    
    with col4:
        if st.button("🔄 Réinitialiser", use_container_width=True):
            st.rerun()

def display_history():
    """Historique des paris"""
    st.header("📊 HISTORIQUE DES PARIS")
    
    if st.session_state.bet_manager is None:
        st.warning("⚠️ Aucun bankroll initialisé. Veuillez initialiser le bankroll dans la sidebar.")
        return
    
    bets = st.session_state.bet_manager.bets
    
    if not bets:
        st.info("📝 Aucun pari enregistré pour le moment. Placez votre premier pari dans l'onglet '💰 Paris'.")
        return
    
    # Afficher les paris avec style
    st.subheader(f"📋 {len(bets)} Pari(s) enregistré(s)")
    
    for bet in bets:
        with st.container():
            col1, col2, col3 = st.columns([4, 2, 2])
            
            with col1:
                st.write(f"**{bet['match']}**")
                st.write(f"📍 Sélection: **{bet['selection']}** @ {bet['odds']:.2f}")
                st.write(f"📅 {bet['timestamp'].strftime('%d/%m/%Y à %H:%M')}")
            
            with col2:
                st.write(f"**💶 Mise:**")
                st.write(f"€{bet['stake']:,.2f}")
            
            with col3:
                st.write(f"**📊 Statut:**")
                if bet['status'] == 'pending':
                    st.info("⏳ En attente")
                else:
                    if bet.get('result') == 'win':
                        st.success("✅ Gagné")
                    elif bet.get('result') == 'loss':
                        st.error("❌ Perdu")
                    else:
                        st.warning("🔄 Annulé")
            
            st.divider()
    
    # Statistiques
    st.subheader("📈 STATISTIQUES")
    
    total_staked = sum(b['stake'] for b in bets)
    pending_bets = [b for b in bets if b['status'] == 'pending']
    settled_bets = [b for b in bets if b.get('result') in ['win', 'loss']]
    won_bets = [b for b in settled_bets if b.get('result') == 'win']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💵 Total misé", f"€{total_staked:,.2f}")
    
    with col2:
        st.metric("⏳ En attente", len(pending_bets))
    
    with col3:
        if settled_bets:
            win_rate = (len(won_bets) / len(settled_bets)) * 100
            st.metric("🎯 Taux réussite", f"{win_rate:.1f}%")
        else:
            st.metric("🎯 Taux réussite", "0.0%")

# =============================================================================
# LANCEMENT
# =============================================================================

if __name__ == "__main__":
    main()
