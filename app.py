# tipser_pro_ai.py - Système Professionnel avec Machine Learning

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import requests
import json
import time
import random
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import xgboost as xgb

# Visualisation
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# CONFIGURATION AVANCÉE
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
    
    # Paramètres ML
    ML_PARAMS = {
        'test_size': 0.2,
        'random_state': 42,
        'n_estimators': 100
    }

# =============================================================================
# MODÈLE DE MACHINE LEARNING
# =============================================================================

class MatchPredictor:
    """Modèle de prédiction avec Machine Learning"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.features = [
            'home_form', 'away_form', 'home_goals_avg', 'away_goals_avg',
            'home_defense', 'away_defense', 'home_possession', 'away_possession',
            'home_shots', 'away_shots', 'h2h_home_wins', 'h2h_away_wins'
        ]
        
    def train_models(self):
        """Entraîne les modèles ML sur des données historiques"""
        # Générer des données d'entraînement simulées
        X_train, y_train = self._generate_training_data()
        
        # Normaliser les features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Modèle Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=Config.ML_PARAMS['n_estimators'],
            random_state=Config.ML_PARAMS['random_state']
        )
        rf_model.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf_model
        
        # Modèle XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=Config.ML_PARAMS['n_estimators'],
            random_state=Config.ML_PARAMS['random_state']
        )
        xgb_model.fit(X_train_scaled, y_train)
        self.models['xgboost'] = xgb_model
        
        # Modèle Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=Config.ML_PARAMS['n_estimators'],
            random_state=Config.ML_PARAMS['random_state']
        )
        gb_model.fit(X_train_scaled, y_train)
        self.models['gradient_boosting'] = gb_model
        
        return self._evaluate_models(X_train_scaled, y_train)
    
    def _generate_training_data(self, n_samples=1000):
        """Génère des données d'entraînement simulées"""
        np.random.seed(Config.ML_PARAMS['random_state'])
        
        X = np.zeros((n_samples, len(self.features)))
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Features
            X[i, 0] = np.random.uniform(0.3, 0.9)  # home_form
            X[i, 1] = np.random.uniform(0.3, 0.9)  # away_form
            X[i, 2] = np.random.uniform(1.0, 2.5)  # home_goals_avg
            X[i, 3] = np.random.uniform(1.0, 2.5)  # away_goals_avg
            X[i, 4] = np.random.uniform(0.3, 0.9)  # home_defense
            X[i, 5] = np.random.uniform(0.3, 0.9)  # away_defense
            X[i, 6] = np.random.uniform(40, 70)    # home_possession
            X[i, 7] = np.random.uniform(40, 70)    # away_possession
            X[i, 8] = np.random.uniform(10, 20)    # home_shots
            X[i, 9] = np.random.uniform(10, 20)    # away_shots
            X[i, 10] = np.random.uniform(0, 5)     # h2h_home_wins
            X[i, 11] = np.random.uniform(0, 5)     # h2h_away_wins
            
            # Target (0: home win, 1: draw, 2: away win)
            home_strength = X[i, 0] + X[i, 2] + X[i, 4] + X[i, 6]/100 + X[i, 10]/10
            away_strength = X[i, 1] + X[i, 3] + X[i, 5] + X[i, 7]/100 + X[i, 11]/10
            
            if abs(home_strength - away_strength) < 0.1:
                y[i] = 1  # Draw
            elif home_strength > away_strength:
                y[i] = 0  # Home win
            else:
                y[i] = 2  # Away win
        
        return X, y
    
    def _evaluate_models(self, X, y):
        """Évalue les performances des modèles"""
        results = {}
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.ML_PARAMS['test_size'],
            random_state=Config.ML_PARAMS['random_state']
        )
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted')
            }
        
        return results
    
    def predict_match(self, home_stats: Dict, away_stats: Dict, h2h_stats: Dict = None):
        """Prédit le résultat d'un match"""
        if not self.models:
            self.train_models()
        
        # Préparer les features
        features = self._extract_features(home_stats, away_stats, h2h_stats)
        features_scaled = self.scaler.transform([features])
        
        # Prédictions de chaque modèle
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]
            
            predictions[name] = pred
            probabilities[name] = {
                'home_win': proba[0],
                'draw': proba[1],
                'away_win': proba[2]
            }
        
        # Consenssus (vote majoritaire)
        final_prediction = self._consensus_prediction(predictions)
        final_probabilities = self._average_probabilities(probabilities)
        
        return {
            'prediction': final_prediction,
            'probabilities': final_probabilities,
            'model_predictions': predictions,
            'model_probabilities': probabilities,
            'confidence': self._calculate_confidence(final_probabilities)
        }
    
    def _extract_features(self, home_stats: Dict, away_stats: Dict, h2h_stats: Dict):
        """Extrait les features pour la prédiction"""
        # Valeurs par défaut
        h2h_stats = h2h_stats or {'home_wins': 0, 'away_wins': 0, 'draws': 0}
        
        return [
            home_stats.get('form_score', 0.5),      # home_form
            away_stats.get('form_score', 0.5),      # away_form
            home_stats.get('goals_for_avg', 1.5),   # home_goals_avg
            away_stats.get('goals_for_avg', 1.5),   # away_goals_avg
            home_stats.get('defense_score', 0.5),   # home_defense
            away_stats.get('defense_score', 0.5),   # away_defense
            home_stats.get('possession', 50),       # home_possession
            away_stats.get('possession', 50),       # away_possession
            home_stats.get('shots_per_game', 15),   # home_shots
            away_stats.get('shots_per_game', 15),   # away_shots
            h2h_stats.get('home_wins', 0),          # h2h_home_wins
            h2h_stats.get('away_wins', 0)           # h2h_away_wins
        ]
    
    def _consensus_prediction(self, predictions: Dict):
        """Calcule la prédiction par consensus"""
        votes = list(predictions.values())
        return max(set(votes), key=votes.count)
    
    def _average_probabilities(self, probabilities: Dict):
        """Calcule les probabilités moyennes"""
        avg_probs = {'home_win': 0, 'draw': 0, 'away_win': 0}
        
        for model_probs in probabilities.values():
            for key in avg_probs.keys():
                avg_probs[key] += model_probs[key]
        
        for key in avg_probs.keys():
            avg_probs[key] /= len(probabilities)
        
        return avg_probs
    
    def _calculate_confidence(self, probabilities: Dict):
        """Calcule le score de confiance"""
        max_prob = max(probabilities.values())
        return min(0.95, max(0.5, max_prob))

# =============================================================================
# DATA MANAGER
# =============================================================================

class DataManager:
    """Gestionnaire de données"""
    
    def __init__(self):
        self.matches = self._load_demo_data()
        self.teams = self._generate_teams_data()
        
    def _load_demo_data(self):
        """Charge les données de démonstration"""
        today = datetime.now()
        
        return [
            {
                'id': 1,
                'home': 'Paris SG',
                'away': 'Marseille',
                'league': 'Ligue 1',
                'date': today + timedelta(days=1),
                'time': '20:45',
                'venue': 'Parc des Princes',
                'odds': {'1': 1.65, 'N': 3.80, '2': 4.50, 'over_2.5': 1.85}
            },
            {
                'id': 2,
                'home': 'Lyon',
                'away': 'Monaco',
                'league': 'Ligue 1',
                'date': today + timedelta(days=2),
                'time': '17:00',
                'venue': 'Groupama Stadium',
                'odds': {'1': 2.10, 'N': 3.40, '2': 3.20, 'over_2.5': 1.75}
            },
            {
                'id': 3,
                'home': 'Real Madrid',
                'away': 'Barcelona',
                'league': 'La Liga',
                'date': today + timedelta(days=3),
                'time': '21:00',
                'venue': 'Santiago Bernabéu',
                'odds': {'1': 2.30, 'N': 3.50, '2': 2.90, 'over_2.5': 1.70}
            },
            {
                'id': 4,
                'home': 'Manchester City',
                'away': 'Liverpool',
                'league': 'Premier League',
                'date': today + timedelta(days=1),
                'time': '16:30',
                'venue': 'Etihad Stadium',
                'odds': {'1': 1.95, 'N': 3.60, '2': 3.40, 'over_2.5': 1.65}
            },
            {
                'id': 5,
                'home': 'Bayern Munich',
                'away': 'Borussia Dortmund',
                'league': 'Bundesliga',
                'date': today + timedelta(days=2),
                'time': '18:30',
                'venue': 'Allianz Arena',
                'odds': {'1': 1.75, 'N': 3.90, '2': 4.00, 'over_2.5': 1.55}
            }
        ]
    
    def _generate_teams_data(self):
        """Génère des données d'équipes"""
        teams = {}
        
        team_list = [
            'Paris SG', 'Marseille', 'Lyon', 'Monaco',
            'Real Madrid', 'Barcelona', 'Manchester City', 'Liverpool',
            'Bayern Munich', 'Borussia Dortmund'
        ]
        
        for team in team_list:
            teams[team] = {
                'form': ''.join(random.choices(['W', 'D', 'L'], k=5)),
                'form_score': random.uniform(0.4, 0.9),
                'goals_for_avg': round(random.uniform(1.2, 2.5), 1),
                'goals_against_avg': round(random.uniform(0.8, 1.8), 1),
                'defense_score': random.uniform(0.4, 0.9),
                'possession': random.randint(45, 65),
                'shots_per_game': random.randint(12, 20),
                'home_strength': random.uniform(0.6, 0.9),
                'away_strength': random.uniform(0.4, 0.8)
            }
        
        return teams
    
    def get_team_stats(self, team_name: str):
        """Récupère les stats d'une équipe"""
        return self.teams.get(team_name, {
            'form': 'DDDDD',
            'form_score': 0.5,
            'goals_for_avg': 1.5,
            'goals_against_avg': 1.5,
            'defense_score': 0.5,
            'possession': 50,
            'shots_per_game': 15
        })
    
    def get_h2h_stats(self, home_team: str, away_team: str):
        """Récupère les stats tête-à-tête"""
        # Simulé pour la démo
        return {
            'home_wins': random.randint(2, 5),
            'away_wins': random.randint(1, 4),
            'draws': random.randint(1, 3),
            'total_goals': random.randint(8, 20),
            'last_5': random.choices([0, 1, 2], k=5)  # 0: home win, 1: draw, 2: away win
        }

# =============================================================================
# VALUE BET ANALYZER
# =============================================================================

class ValueBetAnalyzer:
    """Analyseur de value bets"""
    
    @staticmethod
    def calculate_kelly_criterion(probability: float, odds: float, bankroll: float):
        """Calcule la fraction Kelly"""
        if odds <= 1:
            return 0
        
        fraction = (probability * (odds - 1) - (1 - probability)) / (odds - 1)
        return max(0, min(fraction, 0.5))  # Limiter à 50%
    
    @staticmethod
    def analyze_value(probability: float, odds: float):
        """Analyse la valeur d'un pari"""
        fair_odds = 1 / probability if probability > 0 else float('inf')
        value = (odds / fair_odds) - 1 if fair_odds > 0 else -1
        
        return {
            'fair_odds': round(fair_odds, 2),
            'value': round(value * 100, 1),
            'expected_value': round((probability * (odds - 1)) - (1 - probability), 3),
            'is_value_bet': value > 0.05  # 5% de valeur minimum
        }
    
    @staticmethod
    def recommend_stake(bankroll: float, kelly_fraction: float, risk_profile: str = 'moderate'):
        """Recommande une mise"""
        if risk_profile == 'conservative':
            multiplier = 0.5
        elif risk_profile == 'aggressive':
            multiplier = 1.5
        else:  # moderate
            multiplier = 1.0
        
        suggested_stake = bankroll * kelly_fraction * multiplier
        return max(1, min(suggested_stake, bankroll * 0.1))  # Max 10% du bankroll

# =============================================================================
# INTERFACE UTILISATEUR
# =============================================================================

class UIComponents:
    """Composants d'interface réutilisables"""
    
    @staticmethod
    def setup_page():
        """Configure la page Streamlit"""
        st.set_page_config(
            page_title="Tipser Pro AI | Pronostics Intelligents",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # CSS personnalisé
        st.markdown(f"""
        <style>
        /* Variables de couleurs */
        :root {{
            --primary: {Config.COLORS['primary']};
            --secondary: {Config.COLORS['secondary']};
            --success: {Config.COLORS['success']};
            --warning: {Config.COLORS['warning']};
            --danger: {Config.COLORS['danger']};
            --info: {Config.COLORS['info']};
        }}
        
        /* Styles principaux */
        .main-header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 20px;
        }}
        
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
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin: 2px;
        }}
        
        .badge-success {{ background: linear-gradient(135deg, var(--success) 0%, #2E7D32 100%); color: white; }}
        .badge-warning {{ background: linear-gradient(135deg, var(--warning) 0%, #F57C00 100%); color: white; }}
        .badge-danger {{ background: linear-gradient(135deg, var(--danger) 0%, #D32F2F 100%); color: white; }}
        .badge-info {{ background: linear-gradient(135deg, var(--info) 0%, #1976D2 100%); color: white; }}
        .badge-premium {{ background: linear-gradient(135deg, #FFD700 0%, #FFC107 100%); color: #333; font-weight: bold; }}
        
        /* Amélioration des progress bars */
        .stProgress > div > div > div {{
            background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        }}
        
        /* Boutons */
        .stButton > button {{
            border-radius: 10px;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            transform: scale(1.05);
        }}
        </style>
        """, unsafe_allow_html=True)

# =============================================================================
# APPLICATION PRINCIPALE
# =============================================================================

class TipserProApp:
    """Application principale Tipser Pro"""
    
    def __init__(self):
        self.ui = UIComponents()
        self.data_manager = DataManager()
        self.predictor = MatchPredictor()
        self.value_analyzer = ValueBetAnalyzer()
        
        # Initialiser session state
        self._init_session_state()
        
    def _init_session_state(self):
        """Initialise le session state"""
        if 'selected_match' not in st.session_state:
            st.session_state.selected_match = None
        if 'view_mode' not in st.session_state:
            st.session_state.view_mode = 'dashboard'
        if 'bankroll' not in st.session_state:
            st.session_state.bankroll = 1000
        if 'risk_profile' not in st.session_state:
            st.session_state.risk_profile = 'moderate'
        if 'ml_models_trained' not in st.session_state:
            st.session_state.ml_models_trained = False
    
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
                <h1>🤖 TIPSER PRO AI</h1>
                <h3>Machine Learning & Value Bet Detection</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.metric("Bankroll", f"€{st.session_state.bankroll}")
            st.caption("Powered by AI")
    
    def _display_sidebar(self):
        """Affiche la sidebar"""
        st.sidebar.title("🎯 Navigation")
        
        # Menu principal
        menu_options = {
            "📊 Dashboard": "dashboard",
            "🔍 Match Selection": "selection",
            "🤖 AI Analysis": "analysis",
            "💰 Value Bets": "value",
            "📈 Portfolio": "portfolio",
            "⚙️ Settings": "settings"
        }
        
        selected = st.sidebar.radio(
            "Go to",
            list(menu_options.keys())
        )
        
        st.session_state.view_mode = menu_options[selected]
        
        st.sidebar.divider()
        
        # Filtres rapides
        st.sidebar.subheader("🔎 Quick Filters")
        
        leagues = st.sidebar.multiselect(
            "Leagues",
            ["Ligue 1", "Premier League", "La Liga", "Bundesliga", "Serie A"],
            default=["Ligue 1", "Premier League"]
        )
        
        min_confidence = st.sidebar.slider(
            "Min Confidence",
            50, 95, 70
        )
        
        # Entraîner les modèles ML
        st.sidebar.divider()
        if st.sidebar.button("🤖 Train ML Models", type="secondary", use_container_width=True):
            with st.spinner("Training AI models..."):
                results = self.predictor.train_models()
                st.session_state.ml_models_trained = True
                st.sidebar.success("Models trained successfully!")
                
                # Afficher les résultats
                with st.sidebar.expander("Model Performance"):
                    for model, metrics in results.items():
                        st.write(f"**{model}**:")
                        st.write(f"Accuracy: {metrics['accuracy']:.3f}")
                        st.write(f"Precision: {metrics['precision']:.3f}")
        
        # Stats rapides
        st.sidebar.divider()
        st.sidebar.subheader("📊 Quick Stats")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("ROI", "+14.5%")
            st.metric("Tips", "24")
        with col2:
            st.metric("Hit Rate", "68%")
            st.metric("Value", "€145")
    
    def _display_main_content(self):
        """Affiche le contenu principal"""
        view_mode = st.session_state.view_mode
        
        if view_mode == 'dashboard':
            self._display_dashboard()
        elif view_mode == 'selection':
            self._display_selection()
        elif view_mode == 'analysis':
            self._display_analysis()
        elif view_mode == 'value':
            self._display_value_bets()
        elif view_mode == 'portfolio':
            self._display_portfolio()
        elif view_mode == 'settings':
            self._display_settings()
    
    def _display_dashboard(self):
        """Affiche le dashboard"""
        st.title("📊 AI Dashboard")
        
        # KPI Principaux
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🤖 AI Tips</h3>
                <h2>8</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📈 AI Accuracy</h3>
                <h2>72.4%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>💰 Value Detected</h3>
                <h2>+€245</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>🏆 ROI (30d)</h3>
                <h2>+16.8%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Graphiques avec Plotly
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 AI Model Performance")
            
            # Graphique de performance des modèles
            models = ['Random Forest', 'XGBoost', 'Gradient Boosting']
            accuracy = [0.724, 0.718, 0.731]
            precision = [0.728, 0.722, 0.735]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Accuracy',
                x=models,
                y=accuracy,
                marker_color=Config.COLORS['primary']
            ))
            fig.add_trace(go.Bar(
                name='Precision',
                x=models,
                y=precision,
                marker_color=Config.COLORS['secondary']
            ))
            
            fig.update_layout(
                barmode='group',
                height=300,
                template='plotly_white',
                margin=dict(l=20, r=20, t=30, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Bet Distribution")
            
            # Graphique en camembert
            labels = ['1N2', 'Over/Under', 'BTTS', 'Handicap', 'Others']
            values = [45, 25, 15, 10, 5]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.3,
                marker_colors=[Config.COLORS['primary'], Config.COLORS['secondary'],
                             Config.COLORS['success'], Config.COLORS['warning'],
                             Config.COLORS['info']]
            )])
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Derniers tips AI
        st.divider()
        st.subheader("🤖 Latest AI Predictions")
        
        ai_tips = [
            {"match": "PSG vs Marseille", "prediction": "1", "confidence": 72, "value": "+8.2%", "status": "✅ Won"},
            {"match": "Real vs Barca", "prediction": "Over 2.5", "confidence": 68, "value": "+7.1%", "status": "⏳ Live"},
            {"match": "City vs Liverpool", "prediction": "BTTS Yes", "confidence": 75, "value": "+6.5%", "status": "✅ Won"},
            {"match": "Bayern vs Dortmund", "prediction": "1 & Over 2.5", "confidence": 65, "value": "+5.3%", "status": "⏳ Pending"}
        ]
        
        for tip in ai_tips:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
                
                with col1:
                    st.write(f"**{tip['match']}**")
                
                with col2:
                    st.code(tip['prediction'])
                
                with col3:
                    st.progress(tip['confidence']/100)
                    st.caption(f"{tip['confidence']}%")
                
                with col4:
                    st.markdown(f"<span class='badge badge-premium'>{tip['value']}</span>", unsafe_allow_html=True)
                
                with col5:
                    if tip['status'] == '✅ Won':
                        st.success(tip['status'])
                    elif tip['status'] == '❌ Lost':
                        st.error(tip['status'])
                    else:
                        st.info(tip['status'])
                
                st.divider()
    
    def _display_selection(self):
        """Affiche la sélection des matchs"""
        st.title("🔍 Match Selection")
        
        # Filtres avancés
        with st.expander("🎯 Advanced Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                date_filter = st.selectbox(
                    "Time Period",
                    ["Today", "Tomorrow", "This Weekend", "Next 7 Days", "All Upcoming"]
                )
            
            with col2:
                league_filter = st.multiselect(
                    "Leagues",
                    ["Ligue 1", "Premier League", "La Liga", "Bundesliga", "Serie A",
                     "Champions League", "Europa League"],
                    default=["Ligue 1", "Premier League"]
                )
            
            with col3:
                col_a, col_b = st.columns(2)
                with col_a:
                    min_odds = st.number_input("Min Odds", 1.2, 5.0, 1.5, 0.1)
                with col_b:
                    max_odds = st.number_input("Max Odds", 1.5, 10.0, 3.0, 0.1)
        
        # Bouton de recherche
        if st.button("🔍 Search Matches", type="primary", use_container_width=True):
            with st.spinner("Analyzing matches with AI..."):
                time.sleep(1)
                self._display_match_cards()
    
    def _display_match_cards(self):
        """Affiche les cartes de match"""
        matches = self.data_manager.matches
        
        st.subheader(f"📋 Available Matches ({len(matches)})")
        
        for match in matches:
            # Récupérer les prédictions AI
            home_stats = self.data_manager.get_team_stats(match['home'])
            away_stats = self.data_manager.get_team_stats(match['away'])
            h2h_stats = self.data_manager.get_h2h_stats(match['home'], match['away'])
            
            prediction = self.predictor.predict_match(home_stats, away_stats, h2h_stats)
            
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
                    <span class="badge badge-info">AI Confidence: {prediction['confidence']*100:.1f}%</span>
                    <span class="badge badge-premium">Venue: {match['venue']}</span>
                </div>
            </div>
            
            <div style="display: flex; justify-content: space-around; margin: 20px 0;">
                <div style="text-align: center;">
                    <h5>1</h5>
                    <h3>{match['odds']['1']}</h3>
                    <p>{prediction['probabilities']['home_win']*100:.1f}%</p>
                </div>
                <div style="text-align: center;">
                    <h5>N</h5>
                    <h3>{match['odds']['N']}</h3>
                    <p>{prediction['probabilities']['draw']*100:.1f}%</p>
                </div>
                <div style="text-align: center;">
                    <h5>2</h5>
                    <h3>{match['odds']['2']}</h3>
                    <p>{prediction['probabilities']['away_win']*100:.1f}%</p>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 20px;">
                <h4>🤖 AI Prediction: {self._get_prediction_text(prediction['prediction'], match)}</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Boutons d'action
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"📊 Analyze with AI", key=f"analyze_{match['id']}", use_container_width=True):
                st.session_state.selected_match = match
                st.session_state.view_mode = 'analysis'
                st.rerun()
        
        with col2:
            if st.button(f"💰 Value Analysis", key=f"value_{match['id']}", use_container_width=True):
                st.session_state.selected_match = match
                st.session_state.view_mode = 'value'
                st.rerun()
        
        st.divider()
    
    def _get_prediction_text(self, prediction_code, match):
        """Convertit le code de prédiction en texte"""
        if prediction_code == 0:
            return f"{match['home']} Win"
        elif prediction_code == 1:
            return "Draw"
        else:
            return f"{match['away']} Win"
    
    def _display_analysis(self):
        """Affiche l'analyse AI"""
        if not st.session_state.selected_match:
            st.warning("Please select a match first")
            st.session_state.view_mode = 'selection'
            st.rerun()
            return
        
        match = st.session_state.selected_match
        
        # Bouton retour
        if st.button("← Back to Selection"):
            st.session_state.view_mode = 'selection'
            st.rerun()
        
        st.title(f"🤖 AI Analysis: {match['home']} vs {match['away']}")
        
        # Onglets d'analyse
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview",
            "🤖 ML Predictions",
            "📈 Statistics",
            "🎯 Betting Insights"
        ])
        
        with tab1:
            self._display_analysis_overview(match)
        
        with tab2:
            self._display_ml_predictions(match)
        
        with tab3:
            self._display_statistics(match)
        
        with tab4:
            self._display_betting_insights(match)
    
    def _display_analysis_overview(self, match):
        """Affiche la vue d'ensemble de l'analyse"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏟️ Match Info")
            
            info = {
                "League": match['league'],
                "Date": match['date'].strftime("%d/%m/%Y"),
                "Time": match['time'],
                "Venue": match['venue'],
                "Referee": "M. Turpin",
                "Weather": "☀️ 18°C, Clear"
            }
            
            for key, value in info.items():
                st.metric(key, value)
        
        with col2:
            st.subheader("📊 Market Odds")
            
            odds_data = {
                "Home Win": match['odds']['1'],
                "Draw": match['odds']['N'],
                "Away Win": match['odds']['2'],
                "Over 2.5 Goals": match['odds']['over_2.5']
            }
            
            df_odds = pd.DataFrame(list(odds_data.items()), columns=['Market', 'Odds'])
            st.dataframe(df_odds, use_container_width=True, hide_index=True)
        
        # Team Form
        st.divider()
        st.subheader("📈 Team Form Analysis")
        
        home_stats = self.data_manager.get_team_stats(match['home'])
        away_stats = self.data_manager.get_team_stats(match['away'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**{match['home']}**")
            self._display_team_form_chart(home_stats, match['home'])
        
        with col2:
            st.write(f"**{match['away']}**")
            self._display_team_form_chart(away_stats, match['away'])
    
    def _display_team_form_chart(self, stats, team_name):
        """Affiche le graphique de forme d'une équipe"""
        # Forme récente
        form_chars = list(stats['form'])
        form_values = []
        
        for char in form_chars:
            if char == 'W':
                form_values.append(1)
            elif char == 'D':
                form_values.append(0.5)
            else:  # L
                form_values.append(0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=form_values,
            mode='lines+markers',
            line=dict(color=Config.COLORS['primary'], width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(range=[0, 1.1]),
            showlegend=False,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Métriques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Goals For", stats['goals_for_avg'])
        with col2:
            st.metric("Goals Against", stats['goals_against_avg'])
        with col3:
            st.metric("Form Score", f"{stats['form_score']:.2f}")
    
    def _display_ml_predictions(self, match):
        """Affiche les prédictions ML"""
        st.subheader("🤖 Machine Learning Predictions")
        
        # Obtenir les prédictions
        home_stats = self.data_manager.get_team_stats(match['home'])
        away_stats = self.data_manager.get_team_stats(match['away'])
        h2h_stats = self.data_manager.get_h2h_stats(match['home'], match['away'])
        
        prediction = self.predictor.predict_match(home_stats, away_stats, h2h_stats)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Prediction Probabilities")
            
            # Barres de probabilité
            prob_data = {
                f"✅ {match['home']} Win": prediction['probabilities']['home_win'],
                f"⚪ Draw": prediction['probabilities']['draw'],
                f"✅ {match['away']} Win": prediction['probabilities']['away_win']
            }
            
            for label, prob in prob_data.items():
                st.write(f"**{label}**")
                st.progress(prob)
                st.caption(f"{prob*100:.1f}%")
        
        with col2:
            st.markdown("### 🎯 Model Consensus")
            
            # Graphique radar des modèles
            models = list(prediction['model_probabilities'].keys())
            home_probs = [prediction['model_probabilities'][m]['home_win'] for m in models]
            draw_probs = [prediction['model_probabilities'][m]['draw'] for m in models]
            away_probs = [prediction['model_probabilities'][m]['away_win'] for m in models]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=home_probs,
                theta=models,
                fill='toself',
                name=f'{match["home"]} Win',
                line_color=Config.COLORS['success']
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=draw_probs,
                theta=models,
                fill='toself',
                name='Draw',
                line_color=Config.COLORS['warning']
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=away_probs,
                theta=models,
                fill='toself',
                name=f'{match["away"]} Win',
                line_color=Config.COLORS['danger']
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                height=350,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommandation finale
        st.divider()
        st.markdown("### 🏆 AI Recommendation")
        
        final_pred = self._get_prediction_text(prediction['prediction'], match)
        confidence = prediction['confidence'] * 100
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {Config.COLORS['primary']} 0%, {Config.COLORS['secondary']} 100%);
                    color: white;
                    padding: 25px;
                    border-radius: 15px;
                    text-align: center;
                    margin: 20px 0;">
            <h2>{final_pred}</h2>
            <h3>Confidence: {confidence:.1f}%</h3>
            <p>Based on ensemble of {len(prediction['model_predictions'])} ML models</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_statistics(self, match):
        """Affiche les statistiques détaillées"""
        st.subheader("📈 Advanced Statistics")
        
        home_stats = self.data_manager.get_team_stats(match['home'])
        away_stats = self.data_manager.get_team_stats(match['away'])
        
        # Tableau comparatif
        comparison_data = {
            'Statistic': [
                'Recent Form',
                'Form Score',
                'Goals For (avg)',
                'Goals Against (avg)',
                'Possession %',
                'Shots per Game',
                'Defense Score',
                'Expected Goals (xG)'
            ],
            match['home']: [
                home_stats['form'],
                f"{home_stats['form_score']:.2f}",
                home_stats['goals_for_avg'],
                home_stats['goals_against_avg'],
                f"{home_stats['possession']}%",
                home_stats['shots_per_game'],
                f"{home_stats['defense_score']:.2f}",
                f"{home_stats.get('xg', round(random.uniform(1.5, 2.3), 1))}"
            ],
            match['away']: [
                away_stats['form'],
                f"{away_stats['form_score']:.2f}",
                away_stats['goals_for_avg'],
                away_stats['goals_against_avg'],
                f"{away_stats['possession']}%",
                away_stats['shots_per_game'],
                f"{away_stats['defense_score']:.2f}",
                f"{away_stats.get('xg', round(random.uniform(1.5, 2.3), 1))}"
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df.set_index('Statistic'), use_container_width=True)
        
        # Graphiques comparatifs
        st.divider()
        st.subheader("📊 Statistical Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique à barres comparatif
            categories = ['Attack', 'Defense', 'Form', 'Possession']
            home_values = [
                home_stats['goals_for_avg'] / 3,
                (3 - home_stats['goals_against_avg']) / 3,
                home_stats['form_score'],
                home_stats['possession'] / 100
            ]
            away_values = [
                away_stats['goals_for_avg'] / 3,
                (3 - away_stats['goals_against_avg']) / 3,
                away_stats['form_score'],
                away_stats['possession'] / 100
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name=match['home'],
                x=categories,
                y=home_values,
                marker_color=Config.COLORS['primary']
            ))
            fig.add_trace(go.Bar(
                name=match['away'],
                x=categories,
                y=away_values,
                marker_color=Config.COLORS['secondary']
            ))
            
            fig.update_layout(
                barmode='group',
                height=300,
                title="Team Strength Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Graphique radar
            categories = ['Attack', 'Defense', 'Midfield', 'Form', 'Consistency']
            home_radar = [
                home_stats['goals_for_avg'] / 2.5,
                (2.5 - home_stats['goals_against_avg']) / 2.5,
                home_stats['possession'] / 70,
                home_stats['form_score'],
                random.uniform(0.6, 0.9)
            ]
            away_radar = [
                away_stats['goals_for_avg'] / 2.5,
                (2.5 - away_stats['goals_against_avg']) / 2.5,
                away_stats['possession'] / 70,
                away_stats['form_score'],
                random.uniform(0.6, 0.9)
            ]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=home_radar,
                theta=categories,
                fill='toself',
                name=match['home']
            ))
            fig.add_trace(go.Scatterpolar(
                r=away_radar,
                theta=categories,
                fill='toself',
                name=match['away']
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                height=300,
                title="Radar Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _display_betting_insights(self, match):
        """Affiche les insights de pari"""
        st.subheader("💰 Betting Insights")
        
        # Analyse de valeur
        home_stats = self.data_manager.get_team_stats(match['home'])
        away_stats = self.data_manager.get_team_stats(match['away'])
        h2h_stats = self.data_manager.get_h2h_stats(match['home'], match['away'])
        
        prediction = self.predictor.predict_match(home_stats, away_stats, h2h_stats)
        
        # Analyser chaque marché
        markets = [
            {'name': 'Home Win', 'probability': prediction['probabilities']['home_win'], 'odds': match['odds']['1']},
            {'name': 'Draw', 'probability': prediction['probabilities']['draw'], 'odds': match['odds']['N']},
            {'name': 'Away Win', 'probability': prediction['probabilities']['away_win'], 'odds': match['odds']['2']},
            {'name': 'Over 2.5', 'probability': 0.45, 'odds': match['odds']['over_2.5']}
        ]
        
        value_bets = []
        
        for market in markets:
            analysis = self.value_analyzer.analyze_value(market['probability'], market['odds'])
            
            if analysis['is_value_bet']:
                value_bets.append({
                    'market': market['name'],
                    'odds': market['odds'],
                    'probability': market['probability'],
                    'value': analysis['value'],
                    'expected_value': analysis['expected_value'],
                    'fair_odds': analysis['fair_odds']
                })
        
        if value_bets:
            st.success(f"🎯 {len(value_bets)} Value Bets Detected!")
            
            for bet in value_bets:
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{bet['market']}**")
                        st.caption(f"Fair Odds: {bet['fair_odds']}")
                    
                    with col2:
                        st.metric("Odds", bet['odds'])
                    
                    with col3:
                        st.metric("Value", f"+{bet['value']}%")
                    
                    with col4:
                        st.metric("EV", f"{bet['expected_value']:.3f}")
                    
                    st.divider()
        else:
            st.info("ℹ️ No significant value bets detected for this match.")
        
        # Calculateur de mise Kelly
        st.divider()
        st.subheader("🏦 Kelly Calculator")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bankroll = st.number_input(
                "Bankroll (€)",
                min_value=10,
                max_value=100000,
                value=st.session_state.bankroll
            )
        
        with col2:
            selected_market = st.selectbox(
                "Select Market",
                [m['name'] for m in markets]
            )
            
            # Trouver la probabilité correspondante
            selected_prob = next((m['probability'] for m in markets if m['name'] == selected_market), 0.5)
            selected_odds = next((m['odds'] for m in markets if m['name'] == selected_market), 2.0)
        
        with col3:
            probability = st.slider(
                "Your Probability Estimate (%)",
                0, 100,
                int(selected_prob * 100)
            ) / 100
            
            odds = st.number_input(
                "Odds",
                value=float(selected_odds),
                min_value=1.1,
                max_value=100.0,
                step=0.1
            )
        
        # Calculs
        kelly_fraction = self.value_analyzer.calculate_kelly_criterion(probability, odds, bankroll)
        suggested_stake = self.value_analyzer.recommend_stake(bankroll, kelly_fraction, st.session_state.risk_profile)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Kelly Fraction", f"{kelly_fraction*100:.1f}%")
        
        with col2:
            st.metric("Suggested Stake", f"€{suggested_stake:.2f}")
        
        with col3:
            st.metric("Max Exposure", f"€{bankroll * 0.1:.2f}")
    
    def _display_value_bets(self):
        """Affiche la page des value bets"""
        st.title("💰 Value Bet Detection")
        
        # Filtrer les value bets
        st.subheader("🔍 Scan for Value Bets")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_value = st.slider("Min Value (%)", 0, 50, 5)
        
        with col2:
            min_confidence = st.slider("Min AI Confidence", 50, 95, 70)
        
        with col3:
            max_stake = st.slider("Max Stake (%)", 1, 20, 10)
        
        if st.button("🔍 Scan All Matches", type="primary", use_container_width=True):
            with st.spinner("Scanning for value bets..."):
                time.sleep(2)
                self._display_value_bets_results(min_value, min_confidence)
    
    def _display_value_bets_results(self, min_value, min_confidence):
        """Affiche les résultats des value bets"""
        # Simuler la détection de value bets
        value_bets = []
        
        for match in self.data_manager.matches[:3]:  # Analyser 3 matchs
            home_stats = self.data_manager.get_team_stats(match['home'])
            away_stats = self.data_manager.get_team_stats(match['away'])
            h2h_stats = self.data_manager.get_h2h_stats(match['home'], match['away'])
            
            prediction = self.predictor.predict_match(home_stats, away_stats, h2h_stats)
            
            # Vérifier la confiance
            if prediction['confidence'] * 100 >= min_confidence:
                # Analyser les marchés
                markets = [
                    {'name': f"{match['home']} Win", 'probability': prediction['probabilities']['home_win'], 'odds': match['odds']['1']},
                    {'name': "Draw", 'probability': prediction['probabilities']['draw'], 'odds': match['odds']['N']},
                    {'name': f"{match['away']} Win", 'probability': prediction['probabilities']['away_win'], 'odds': match['odds']['2']}
                ]
                
                for market in markets:
                    analysis = self.value_analyzer.analyze_value(market['probability'], market['odds'])
                    
                    if analysis['value'] >= min_value:
                        value_bets.append({
                            'match': f"{match['home']} vs {match['away']}",
                            'market': market['name'],
                            'odds': market['odds'],
                            'probability': market['probability'] * 100,
                            'value': analysis['value'],
                            'confidence': prediction['confidence'] * 100,
                            'league': match['league']
                        })
        
        if value_bets:
            st.success(f"🎯 Found {len(value_bets)} Value Bets!")
            
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
                                <span class="badge badge-premium">+{bet['value']}% Value</span>
                                <span class="badge badge-info">{bet['confidence']:.1f}% AI Confidence</span>
                            </div>
                        </div>
                        
                        <div style="display: flex; justify-content: space-around; margin: 20px 0;">
                            <div style="text-align: center;">
                                <h5>Odds</h5>
                                <h3>{bet['odds']}</h3>
                            </div>
                            <div style="text-align: center;">
                                <h5>Probability</h5>
                                <h3>{bet['probability']:.1f}%</h3>
                            </div>
                            <div style="text-align: center;">
                                <h5>Value</h5>
                                <h3>+{bet['value']}%</h3>
                            </div>
                        </div>
                        
                        <div style="text-align: center;">
                            <h4>💰 Expected Value: {(bet['probability']/100 * (bet['odds'] - 1) - (1 - bet['probability']/100)):.3f}</h4>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Bouton pour placer le pari
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📊 Analyze", key=f"analyze_val_{bet['match']}", use_container_width=True):
                            match_to_select = next((m for m in self.data_manager.matches 
                                                  if f"{m['home']} vs {m['away']}" == bet['match']), None)
                            if match_to_select:
                                st.session_state.selected_match = match_to_select
                                st.session_state.view_mode = 'analysis'
                                st.rerun()
                    
                    with col2:
                        if st.button(f"💰 Place Bet", key=f"bet_{bet['match']}", use_container_width=True, type="primary"):
                            st.success(f"Bet placed on {bet['match']} - {bet['market']}")
                    
                    st.divider()
        else:
            st.info("ℹ️ No value bets found with current filters.")
    
    def _display_portfolio(self):
        """Affiche le portfolio"""
        st.title("💰 My Portfolio")
        
        # Résumé
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Bets", "3", "+1")
        
        with col2:
            st.metric("Total Investment", "€150")
        
        with col3:
            st.metric("Potential Profit", "€285")
        
        with col4:
            st.metric("ROI", "+90%", "+5.2%")
        
        st.divider()
        
        # Graphique de performance
        st.subheader("📈 Performance Chart")
        
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        portfolio_value = [1000 + i * 10 + random.randint(-20, 30) for i in range(30)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_value,
            mode='lines',
            line=dict(color=Config.COLORS['success'], width=3),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        
        fig.update_layout(
            height=300,
            template='plotly_white',
            xaxis_title="Date",
            yaxis_title="Portfolio Value (€)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Paris actifs
        st.divider()
        st.subheader("📊 Active Bets")
        
        active_bets = [
            {"match": "PSG vs Marseille", "type": "Over 2.5", "odds": 1.85, "stake": "€50", "status": "⏳ Pending"},
            {"match": "Real Madrid vs Barca", "type": "1", "odds": 2.10, "stake": "€60", "status": "⏳ Pending"},
            {"match": "Liverpool vs City", "type": "BTTS Yes", "odds": 1.65, "stake": "€40", "status": "⏳ Live"}
        ]
        
        for bet in active_bets:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
                with col1:
                    st.write(f"**{bet['match']}**")
                with col2:
                    st.code(bet['type'])
                with col3:
                    st.metric("Odds", bet['odds'])
                with col4:
                    st.write(bet['stake'])
                with col5:
                    if bet['status'] == '⏳ Live':
                        st.warning(bet['status'])
                    else:
                        st.info(bet['status'])
                st.divider()
    
    def _display_settings(self):
        """Affiche les paramètres"""
        st.title("⚙️ Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 User Profile")
            
            st.selectbox(
                "Subscription Level",
                ["Free", "Basic", "Pro", "Enterprise"],
                index=2,
                key="subscription_level"
            )
            
            st.number_input(
                "Bankroll (€)",
                min_value=100,
                max_value=100000,
                value=st.session_state.bankroll,
                step=100,
                key="bankroll_input"
            )
            
            st.selectbox(
                "Risk Profile",
                ["Conservative", "Moderate", "Aggressive"],
                index=1,
                key="risk_profile_select"
            )
        
        with col2:
            st.subheader("🔧 Betting Settings")
            
            st.number_input(
                "Max Stake (%)",
                min_value=1,
                max_value=20,
                value=5,
                key="max_stake_percent"
            )
            
            st.slider(
                "Min Value Bet Threshold (%)",
                0, 20, 5,
                key="min_value_threshold"
            )
            
            st.checkbox("Email Notifications", value=True, key="email_notifications")
            st.checkbox("Value Bet Alerts", value=True, key="value_alerts")
            st.checkbox("Dark Mode", value=False, key="dark_mode")
        
        # Boutons d'action
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Settings", type="primary", use_container_width=True):
                st.session_state.bankroll = st.session_state.bankroll_input
                st.session_state.risk_profile = st.session_state.risk_profile_select.lower()
                st.success("Settings saved successfully!")
        
        with col2:
            if st.button("🔄 Reset to Default", use_container_width=True):
                st.session_state.bankroll = 1000
                st.session_state.risk_profile = 'moderate'
                st.rerun()
        
        with col3:
            if st.button("📤 Export Data", use_container_width=True):
                st.info("Export feature coming soon...")

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main():
    """Fonction principale"""
    app = TipserProApp()
    app.run()

if __name__ == "__main__":
    main()
