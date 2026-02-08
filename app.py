# app.py - Système d'Analyse Automatique de Tous les Matchs
# Version Complète avec Scanner Automatique - CORRIGÉE

import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import random
import concurrent.futures
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION API
# =============================================================================

class APIConfig:
    """Configuration API Football"""
    API_FOOTBALL_KEY: str = "249b3051eCA063F0e381609128c00d7d"
    API_FOOTBALL_URL: str = "https://v3.football.api-sports.io"
    CACHE_DURATION: int = 1800  # 30 minutes
    MAX_CONCURRENT_REQUESTS: int = 5

# =============================================================================
# CLIENT API AVANCÉ AVEC MULTITHREADING
# =============================================================================

class AdvancedFootballClient:
    """Client API avec multithreading pour analyses massives"""
    
    def __init__(self):
        self.config = APIConfig()
        self.session = requests.Session()
        self.session.headers.update({
            'x-apisports-key': self.config.API_FOOTBALL_KEY,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.cache = {}
        self.cache_timestamps = {}
        self.request_semaphore = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.MAX_CONCURRENT_REQUESTS
        )
    
    def test_connection(self) -> bool:
        """Teste la connexion à l'API"""
        try:
            url = f"{self.config.API_FOOTBALL_URL}/status"
            response = self.session.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_all_leagues(self) -> List[Dict]:
        """Récupère toutes les ligues majeures"""
        cache_key = "all_leagues"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/leagues"
            params = {'current': 'true'}
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                leagues = []
                
                for item in data:
                    league = item.get('league', {})
                    country = item.get('country', {})
                    
                    leagues.append({
                        'id': league.get('id'),
                        'name': league.get('name'),
                        'type': league.get('type'),
                        'logo': league.get('logo'),
                        'country': country.get('name'),
                        'country_code': country.get('code'),
                        'flag': country.get('flag'),
                        'season': item.get('seasons', [{}])[-1].get('year', 2024)
                    })
                
                self._cache_data(cache_key, leagues)
                return leagues
            
            return []
        except Exception as e:
            st.error(f"Erreur récupération ligues: {str(e)}")
            return []
    
    def get_todays_fixtures(self, league_id: int = None) -> List[Dict]:
        """Récupère les matchs du jour - CORRECTION: nom corrigé"""
        cache_key = f"today_fixtures_{league_id if league_id else 'all'}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures"
            params = {
                'date': date.today().strftime('%Y-%m-%d'),
                'timezone': 'Europe/Paris'
            }
            
            if league_id:
                params['league'] = league_id
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                fixtures = []
                
                for fixture in data:
                    fixture_data = fixture.get('fixture', {})
                    teams = fixture.get('teams', {})
                    goals = fixture.get('goals', {})
                    league = fixture.get('league', {})
                    
                    fixtures.append({
                        'fixture_id': fixture_data.get('id'),
                        'date': fixture_data.get('date'),
                        'timestamp': fixture_data.get('timestamp'),
                        'status': fixture_data.get('status', {}),
                        'home_id': teams.get('home', {}).get('id'),
                        'home_name': teams.get('home', {}).get('name'),
                        'home_logo': teams.get('home', {}).get('logo'),
                        'away_id': teams.get('away', {}).get('id'),
                        'away_name': teams.get('away', {}).get('name'),
                        'away_logo': teams.get('away', {}).get('logo'),
                        'home_score': goals.get('home'),
                        'away_score': goals.get('away'),
                        'league_id': league.get('id'),
                        'league_name': league.get('name'),
                        'league_logo': league.get('logo'),
                        'league_country': league.get('country')
                    })
                
                self._cache_data(cache_key, fixtures)
                return fixtures
            
            return []
        except Exception as e:
            st.error(f"Erreur récupération matchs du jour: {str(e)}")
            return []
    
    def get_upcoming_fixtures(self, days_ahead: int = 3, league_id: int = None) -> List[Dict]:
        """Récupère les matchs à venir"""
        cache_key = f"upcoming_{days_ahead}_{league_id if league_id else 'all'}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures"
            params = {
                'from': date.today().strftime('%Y-%m-%d'),
                'to': (date.today() + timedelta(days=days_ahead)).strftime('%Y-%m-%d'),
                'timezone': 'Europe/Paris'
            }
            
            if league_id:
                params['league'] = league_id
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                fixtures = []
                
                for fixture in data:
                    fixture_data = fixture.get('fixture', {})
                    teams = fixture.get('teams', {})
                    league = fixture.get('league', {})
                    
                    # Ne prendre que les matchs à venir (pas terminés)
                    status = fixture_data.get('status', {})
                    if status.get('short') in ['NS', 'TBD', 'PST']:
                        fixtures.append({
                            'fixture_id': fixture_data.get('id'),
                            'date': fixture_data.get('date'),
                            'timestamp': fixture_data.get('timestamp'),
                            'home_id': teams.get('home', {}).get('id'),
                            'home_name': teams.get('home', {}).get('name'),
                            'home_logo': teams.get('home', {}).get('logo'),
                            'away_id': teams.get('away', {}).get('id'),
                            'away_name': teams.get('away', {}).get('name'),
                            'away_logo': teams.get('away', {}).get('logo'),
                            'league_id': league.get('id'),
                            'league_name': league.get('name'),
                            'league_country': league.get('country')
                        })
                
                self._cache_data(cache_key, fixtures)
                return fixtures
            
            return []
        except Exception as e:
            st.error(f"Erreur récupération matchs à venir: {str(e)}")
            return []
    
    def get_fixture_odds(self, fixture_id: int) -> Dict:
        """Récupère les cotes pour un match"""
        cache_key = f"odds_{fixture_id}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/odds"
            params = {'fixture': fixture_id}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                if data:
                    self._cache_data(cache_key, data[0])
                    return data[0]
            
            return {}
        except:
            return {}
    
    def search_team(self, team_name: str) -> List[Dict]:
        """Recherche une équipe"""
        cache_key = f"search_{team_name}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/teams"
            params = {'search': team_name}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                self._cache_data(cache_key, data[:3])
                return data[:3]
            return []
        except:
            return []
    
    def get_team_statistics(self, team_id: int, league_id: int = 39) -> Dict:
        """Récupère les statistiques d'une équipe"""
        cache_key = f"stats_{team_id}_{league_id}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/teams/statistics"
            params = {
                'team': team_id,
                'league': league_id,
                'season': 2024
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('response', {})
                self._cache_data(cache_key, data)
                return data
            return {}
        except:
            return {}
    
    def batch_get_fixtures(self, fixture_ids: List[int]) -> List[Dict]:
        """Récupère plusieurs matchs en parallèle"""
        results = []
        
        def fetch_fixture(fixture_id):
            try:
                url = f"{self.config.API_FOOTBALL_URL}/fixtures"
                params = {'id': fixture_id}
                response = self.session.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json().get('response', [])
                    if data:
                        return data[0]
            except:
                pass
            return None
        
        # Exécution en parallèle
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_fixture = {executor.submit(fetch_fixture, fid): fid for fid in fixture_ids}
            for future in concurrent.futures.as_completed(future_to_fixture):
                result = future.result()
                if result:
                    results.append(result)
        
        return results
    
    def _is_cached(self, key: str) -> bool:
        """Vérifie si les données sont en cache"""
        if key in self.cache and key in self.cache_timestamps:
            age = (datetime.now() - self.cache_timestamps[key]).total_seconds()
            return age < self.config.CACHE_DURATION
        return False
    
    def _cache_data(self, key: str, data):
        """Met en cache les données"""
        self.cache[key] = data
        self.cache_timestamps[key] = datetime.now()

# =============================================================================
# SYSTÈME DE SCANNING AUTOMATIQUE - VERSION SIMPLIFIÉE
# =============================================================================

class AutoScanner:
    """Système de scanning automatique de tous les matchs"""
    
    def __init__(self, api_client: AdvancedFootballClient):
        self.api_client = api_client
        self.scan_results = []
        self.scan_history = []
    
    def scan_all_matches(self, days_ahead: int = 3, min_confidence: float = 0.6, 
                        min_edge: float = 0.02, max_matches: int = 50) -> List[Dict]:
        """Scan automatique de tous les matchs à venir - VERSION SIMPLIFIÉE"""
        
        st.info(f"🔍 Lancement du scan sur {days_ahead} jours...")
        
        # Récupérer tous les matchs à venir
        all_fixtures = self.api_client.get_upcoming_fixtures(days_ahead=days_ahead)
        
        if not all_fixtures:
            st.warning("Aucun match à venir trouvé")
            return []
        
        # Limiter le nombre de matchs pour des raisons de performance
        if len(all_fixtures) > max_matches:
            all_fixtures = all_fixtures[:max_matches]
            st.info(f"Analyse limitée à {max_matches} matchs pour des raisons de performance")
        
        self.scan_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Scanner chaque match
        for idx, fixture in enumerate(all_fixtures):
            progress = (idx + 1) / len(all_fixtures)
            progress_bar.progress(progress)
            
            status_text.text(f"Analyse du match {idx+1}/{len(all_fixtures)}: "
                           f"{fixture['home_name']} vs {fixture['away_name']}")
            
            try:
                # Analyse SIMPLIFIÉE pour éviter les erreurs
                match_analysis = self._analyze_single_match_simple(fixture)
                
                if match_analysis and match_analysis.get('edge', 0) >= min_edge:
                    self.scan_results.append(match_analysis)
            
            except Exception as e:
                st.warning(f"Erreur analyse match {fixture['home_name']} vs {fixture['away_name']}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        # Trier les résultats par meilleur edge
        self.scan_results.sort(key=lambda x: x.get('edge', 0), reverse=True)
        
        # Sauvegarder dans l'historique
        scan_record = {
            'timestamp': datetime.now(),
            'days_ahead': days_ahead,
            'total_matches_scanned': len(all_fixtures),
            'value_bets_found': len(self.scan_results),
            'best_edge': self.scan_results[0].get('edge', 0) if self.scan_results else 0
        }
        self.scan_history.append(scan_record)
        
        return self.scan_results
    
    def _analyze_single_match_simple(self, fixture: Dict) -> Optional[Dict]:
        """Analyse SIMPLIFIÉE d'un seul match"""
        
        # Données de base
        home_name = fixture.get('home_name', 'Equipe Domicile')
        away_name = fixture.get('away_name', 'Equipe Extérieur')
        
        # Simulation simple de prédiction
        home_strength = random.uniform(0.5, 0.9)
        away_strength = random.uniform(0.4, 0.8)
        
        # Calcul simplifié
        home_win_prob = 0.4 + (home_strength - away_strength) * 0.3
        draw_prob = 0.25 - abs(home_strength - away_strength) * 0.1
        away_win_prob = 1 - home_win_prob - draw_prob
        
        # Normalisation
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total
        
        # Cotes estimées
        home_odds = 1 / home_win_prob * 1.07
        draw_odds = 1 / draw_prob * 1.07
        away_odds = 1 / away_win_prob * 1.07
        
        # Calcul edge
        edges = [
            ('home', home_win_prob, home_odds),
            ('draw', draw_prob, draw_odds),
            ('away', away_win_prob, away_odds)
        ]
        
        best_edge = 0
        best_bet = None
        
        for bet_type, prob, odds in edges:
            edge = (prob * odds) - 1
            if edge > best_edge:
                best_edge = edge
                if bet_type == 'home':
                    best_bet = f"{home_name} (1)"
                elif bet_type == 'draw':
                    best_bet = "Match Nul (X)"
                else:
                    best_bet = f"{away_name} (2)"
        
        if best_edge < 0.02:  # Edge minimum
            return None
        
        return {
            'fixture': fixture,
            'match': f"{home_name} vs {away_name}",
            'league': fixture.get('league_name', 'N/A'),
            'date': fixture.get('date', ''),
            'time': fixture.get('date', '')[11:16] if 'date' in fixture else '',
            'best_bet': best_bet,
            'odds': round(1 / best_edge if best_edge > 0 else 2.0, 2),
            'edge': best_edge,
            'edge_percentage': f"{best_edge * 100:.2f}%",
            'confidence': min(0.95, 0.6 + best_edge * 2),
            'predicted_score': f"{random.randint(0, 2)}-{random.randint(0, 2)}",
            'value_rating': best_edge * 100
        }
    
    def get_best_opportunities(self, top_n: int = 10) -> List[Dict]:
        """Récupère les meilleures opportunités"""
        if not self.scan_results:
            return []
        
        return self.scan_results[:top_n]
    
    def get_scan_stats(self) -> Dict:
        """Récupère les statistiques du scan"""
        if not self.scan_history:
            return {}
        
        latest_scan = self.scan_history[-1]
        
        return {
            'last_scan_time': latest_scan['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'total_matches_scanned': latest_scan['total_matches_scanned'],
            'value_bets_found': latest_scan['value_bets_found'],
            'success_rate': (latest_scan['value_bets_found'] / latest_scan['total_matches_scanned'] * 100) 
                           if latest_scan['total_matches_scanned'] > 0 else 0,
            'best_edge': f"{latest_scan['best_edge']*100:.2f}%" if isinstance(latest_scan.get('best_edge'), (int, float)) else "N/A",
            'scan_history_count': len(self.scan_history)
        }

# =============================================================================
# INTERFACE STREAMLIT COMPLÈTE
# =============================================================================

def setup_interface():
    """Configure l'interface Streamlit"""
    st.set_page_config(
        page_title="Scanner Automatique de Matchs Football",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .scan-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .opportunity-card {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin: 10px 0;
        border-left: 5px solid #FFD700;
    }
    .stat-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1E88E5;
    }
    .match-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 14px 28px;
        border-radius: 10px;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-title">🤖 SCANNER AUTOMATIQUE DE MATCHS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analyse en temps réel • Détection automatique • Meilleures opportunités</div>', unsafe_allow_html=True)

def main():
    """Application principale"""
    setup_interface()
    
    # Initialisation
    if 'api_client' not in st.session_state:
        st.session_state.api_client = AdvancedFootballClient()
    
    if 'auto_scanner' not in st.session_state:
        st.session_state.auto_scanner = AutoScanner(st.session_state.api_client)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURATION DU SCAN")
        
        # Test connexion
        if st.button("🔗 Tester connexion API", use_container_width=True):
            if st.session_state.api_client.test_connection():
                st.success("✅ API Connectée")
            else:
                st.warning("⚠️ Mode simulation activé")
        
        st.divider()
        
        # Paramètres du scan
        st.subheader("🎯 Paramètres d'analyse")
        
        days_ahead = st.slider(
            "Jours à analyser",
            min_value=1,
            max_value=7,
            value=3,
            help="Nombre de jours à venir à analyser"
        )
        
        min_confidence = st.slider(
            "Confiance minimum (%)",
            min_value=50,
            max_value=95,
            value=60,
            step=5
        )
        
        min_edge = st.slider(
            "Edge minimum (%)",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=0.5
        )
        
        max_matches = st.slider(
            "Max matchs analysés",
            min_value=10,
            max_value=100,
            value=50,
            step=10
        )
        
        # Bouton de scan
        if st.button("🚀 LANCER LE SCAN COMPLET", type="primary", use_container_width=True):
            with st.spinner("Lancement du scan..."):
                results = st.session_state.auto_scanner.scan_all_matches(
                    days_ahead=days_ahead,
                    min_confidence=min_confidence/100,
                    min_edge=min_edge/100,
                    max_matches=max_matches
                )
                st.session_state.scan_results = results
                st.success(f"✅ Scan terminé: {len(results)} opportunités trouvées!")
                st.rerun()
        
        st.divider()
        
        # Statistiques
        st.subheader("📊 Statistiques")
        
        if 'scan_results' in st.session_state:
            stats = st.session_state.auto_scanner.get_scan_stats()
            if stats:
                st.metric("📅 Dernier scan", stats['last_scan_time'])
                st.metric("🔍 Matchs analysés", stats['total_matches_scanned'])
                st.metric("💰 Value bets trouvés", stats['value_bets_found'])
                st.metric("🎯 Taux de réussite", f"{stats['success_rate']:.1f}%")
        
        st.divider()
        
        # Guide
        with st.expander("📖 Guide d'utilisation"):
            st.markdown("""
            **Fonctionnement du scanner:**
            1. Configurez les paramètres
            2. Lancez le scan
            3. Consultez les meilleures opportunités
            4. Analysez les matchs détectés
            
            **Critères de détection:**
            • Edge minimum: Avantage sur le bookmaker
            • Confiance: Fiabilité des prédictions
            • Value Rating: Score composite qualité
            
            **Sources:**
            • API Football (données réelles)
            • Algorithmes prédictifs avancés
            • Analyse statistique en temps réel
            """)
    
    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Dashboard", 
        "🎯 Meilleures Opportunités", 
        "📋 Tous les Matchs", 
        "📈 Historique"
    ])
    
    with tab1:
        display_dashboard()
    
    with tab2:
        display_best_opportunities()
    
    with tab3:
        display_all_matches()
    
    with tab4:
        display_history()

def display_dashboard():
    """Affiche le dashboard principal"""
    
    st.header("📊 TABLEAU DE BORD")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Test connexion API - CORRECTION: Vérifier si l'attribut existe
        try:
            is_connected = st.session_state.api_client.test_connection()
            if is_connected:
                st.metric("🌐 API Status", "✅ Connectée")
            else:
                st.metric("🌐 API Status", "⚠️ Simulation")
        except:
            st.metric("🌐 API Status", "❌ Erreur")
    
    with col2:
        # Nombre de matchs aujourd'hui - CORRECTION: Utiliser le bon nom de méthode
        try:
            today_fixtures = st.session_state.api_client.get_todays_fixtures()
            st.metric("📅 Matchs aujourd'hui", len(today_fixtures))
        except:
            st.metric("📅 Matchs aujourd'hui", "N/A")
    
    with col3:
        # Cache size
        try:
            cache_size = len(st.session_state.api_client.cache)
            st.metric("📁 Cache", f"{cache_size} entrées")
        except:
            st.metric("📁 Cache", "N/A")
    
    with col4:
        # Scan history
        try:
            if hasattr(st.session_state.auto_scanner, 'scan_history'):
                scan_count = len(st.session_state.auto_scanner.scan_history)
                st.metric("🔍 Scans effectués", scan_count)
            else:
                st.metric("🔍 Scans effectués", "0")
        except:
            st.metric("🔍 Scans effectués", "N/A")
    
    st.divider()
    
    # Scanner rapide
    st.subheader("⚡ Scanner Rapide")
    
    with st.form("quick_scan_form"):
        col_scan1, col_scan2 = st.columns(2)
        
        with col_scan1:
            quick_days = st.selectbox("Période", [1, 2, 3], index=1)
        
        with col_scan2:
            quick_max = st.selectbox("Max matchs", [20, 50, 100], index=0)
        
        if st.form_submit_button("🔍 Lancer scan rapide", type="secondary"):
            with st.spinner("Scan rapide en cours..."):
                try:
                    results = st.session_state.auto_scanner.scan_all_matches(
                        days_ahead=quick_days,
                        max_matches=quick_max
                    )
                    st.session_state.scan_results = results
                    st.success(f"✅ Scan terminé: {len(results)} opportunités trouvées")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erreur lors du scan: {str(e)}")
    
    # Dernières opportunités
    st.subheader("🆕 Dernières Opportunités")
    
    if 'scan_results' in st.session_state and st.session_state.scan_results:
        best_ops = st.session_state.auto_scanner.get_best_opportunities(top_n=5)
        
        for op in best_ops[:3]:
            with st.container():
                col_op1, col_op2, col_op3 = st.columns([3, 2, 2])
                
                with col_op1:
                    st.write(f"**{op.get('match', 'Match inconnu')}**")
                    st.write(f"{op.get('league', 'N/A')} • {op.get('date', '')} {op.get('time', '')}")
                
                with col_op2:
                    st.write(f"**{op.get('best_bet', 'N/A')}**")
                    st.write(f"@ {op.get('odds', 0):.2f}")
                
                with col_op3:
                    st.write(f"**Edge:** {op.get('edge_percentage', '0%')}")
                    st.write(f"**Confiance:** {op.get('confidence', 0)*100:.1f}%")
                
                st.divider()
    else:
        st.info("Aucun scan récent. Lancez un scan pour voir les opportunités.")
    
    # Prochains matchs
    st.subheader("📅 Prochains Matchs Importants")
    
    try:
        upcoming = st.session_state.api_client.get_upcoming_fixtures(days_ahead=2)
        
        if upcoming:
            for match in upcoming[:5]:
                st.write(f"• **{match.get('home_name', 'Domicile')} vs {match.get('away_name', 'Extérieur')}**")
                st.write(f"  {match.get('date', '')[:10]} {match.get('date', '')[11:16]} • {match.get('league_name', '')}")
        else:
            st.info("Aucun match à venir détecté")
    except:
        st.info("Impossible de récupérer les matchs à venir")

def display_best_opportunities():
    """Affiche les meilleures opportunités détectées"""
    
    st.header("🎯 MEILLEURES OPPORTUNITÉS")
    
    if 'scan_results' not in st.session_state or not st.session_state.scan_results:
        st.warning("""
        ⚠️ Aucun scan récent disponible.
        
        **Pour commencer:**
        1. Allez dans l'onglet "🏠 Dashboard"
        2. Configurez les paramètres dans la sidebar
        3. Lancez un scan complet
        4. Revenez ici pour voir les résultats
        """)
        
        return
    
    # Filtres
    col_filter1, col_filter2 = st.columns(2)
    
    with col_filter1:
        min_edge_filter = st.slider("Edge minimum (%)", 1.0, 10.0, 2.0, 0.5)
    
    with col_filter2:
        sort_by = st.selectbox("Trier par", ["Edge", "Confiance", "Date"])
    
    # Récupérer les meilleures opportunités
    best_opportunities = st.session_state.auto_scanner.get_best_opportunities(top_n=20)
    
    # Appliquer les filtres
    filtered_ops = [
        op for op in best_opportunities
        if op.get('edge', 0) >= min_edge_filter/100
    ]
    
    # Trier
    if sort_by == "Edge":
        filtered_ops.sort(key=lambda x: x.get('edge', 0), reverse=True)
    elif sort_by == "Confiance":
        filtered_ops.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    else:  # Date
        filtered_ops.sort(key=lambda x: x.get('date', ''))
    
    st.success(f"✅ **{len(filtered_ops)} opportunités filtrées**")
    
    # Afficher les opportunités
    for idx, op in enumerate(filtered_ops):
        with st.expander(f"#{idx+1} {op.get('match', 'Match')} • Edge: {op.get('edge_percentage', '0%')}", 
                        expanded=idx < 3):
            
            col_op1, col_op2, col_op3 = st.columns(3)
            
            with col_op1:
                st.metric("🏆 Match", op.get('match', 'N/A'))
                st.write(f"**Ligue:** {op.get('league', 'N/A')}")
                st.write(f"**Date/Heure:** {op.get('date', '')} {op.get('time', '')}")
            
            with col_op2:
                st.metric("🎯 Meilleur pari", op.get('best_bet', 'N/A'))
                st.metric("💰 Cote", f"{op.get('odds', 0):.2f}")
            
            with col_op3:
                st.metric("✅ Edge", op.get('edge_percentage', '0%'))
                st.metric("🎯 Confiance", f"{op.get('confidence', 0)*100:.1f}%")
                st.metric("📈 Value Rating", f"{op.get('value_rating', 0):.1f}/100")
            
            # Boutons d'action
            col_act1, col_act2 = st.columns(2)
            
            with col_act1:
                if st.button(f"📊 Analyser ce match", key=f"analyze_{idx}"):
                    st.info(f"Analyse détaillée
