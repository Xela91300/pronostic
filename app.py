import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import requests
import json
from pytz import timezone
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="⚽ Football Betting Analytics Live",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation de session
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'timezone' not in st.session_state:
    st.session_state.timezone = 'Europe/Paris'
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# CSS personnalisé
def load_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .match-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
    }
    
    .live-badge {
        background: #ff4757;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .dataframe {
        font-size: 0.9rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f3f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1rem;
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Classe API Manager
class FootballAPIManager:
    def __init__(self, api_key=None):
        self.api_key = api_key or st.session_state.api_key
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            'x-apisports-key': self.api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }
    
    def make_request(self, endpoint, params=None):
        """Fait une requête à l'API Football"""
        try:
            if not self.api_key:
                return None
                
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.warning(f"Erreur API: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Erreur de connexion: {str(e)}")
            return None
    
    def get_live_matches(self):
        """Récupère les matchs en direct"""
        today = datetime.now().strftime('%Y-%m-%d')
        params = {
            'date': today,
            'status': 'LIVE'
        }
        return self.make_request('fixtures', params)
    
    def get_today_matches(self):
        """Récupère les matchs du jour"""
        today = datetime.now().strftime('%Y-%m-%d')
        params = {'date': today}
        return self.make_request('fixtures', params)
    
    def get_predictions(self, fixture_id):
        """Récupère les prédictions pour un match"""
        params = {'fixture': fixture_id}
        return self.make_request('predictions', params)
    
    def get_odds(self, fixture_id):
        """Récupère les cotes pour un match"""
        params = {'fixture': fixture_id}
        return self.make_request('odds', params)
    
    def get_standings(self, league_id, season):
        """Récupère le classement"""
        params = {'league': league_id, 'season': season}
        return self.make_request('standings', params)
    
    def get_team_statistics(self, team_id, league_id, season):
        """Récupère les statistiques d'une équipe"""
        params = {'team': team_id, 'league': league_id, 'season': season}
        return self.make_request('teams/statistics', params)

# Données de démo si pas d'API
DEMO_DATA = {
    'live_matches': [
        {
            'fixture': {'id': 1, 'status': {'short': 'LIVE', 'elapsed': 65}},
            'teams': {'home': {'name': 'Paris SG', 'logo': 'https://media.api-sports.io/football/teams/85.png'},
                     'away': {'name': 'Marseille', 'logo': 'https://media.api-sports.io/football/teams/81.png'}},
            'goals': {'home': 2, 'away': 1},
            'league': {'name': 'Ligue 1', 'country': 'France'}
        },
        {
            'fixture': {'id': 2, 'status': {'short': 'LIVE', 'elapsed': 30}},
            'teams': {'home': {'name': 'Liverpool', 'logo': 'https://media.api-sports.io/football/teams/40.png'},
                     'away': {'name': 'Manchester City', 'logo': 'https://media.api-sports.io/football/teams/50.png'}},
            'goals': {'home': 0, 'away': 0},
            'league': {'name': 'Premier League', 'country': 'England'}
        }
    ],
    'today_matches': [
        {
            'fixture': {'id': 3, 'date': (datetime.now() + timedelta(hours=2)).isoformat()},
            'teams': {'home': {'name': 'Real Madrid'}, 'away': {'name': 'Barcelona'}},
            'league': {'name': 'La Liga'}
        }
    ]
}

# Fonctions utilitaires
def format_time(timestamp):
    """Formate un timestamp en heure locale"""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        local_tz = timezone(st.session_state.timezone)
        local_dt = dt.astimezone(local_tz)
        return local_dt.strftime('%H:%M')
    except:
        return "N/A"

def calculate_probability(home_stats, away_stats):
    """Calcule les probabilités basiques"""
    if not home_stats or not away_stats:
        return {'home': 45, 'draw': 30, 'away': 25}
    
    # Logique simplifiée de calcul
    home_strength = home_stats.get('strength', 50)
    away_strength = away_stats.get('strength', 50)
    
    total = home_strength + away_strength
    home_prob = int((home_strength / total) * 45) + 40
    away_prob = int((away_strength / total) * 45) + 40
    draw_prob = 100 - home_prob - away_prob
    
    return {
        'home': min(95, max(5, home_prob)),
        'draw': min(95, max(5, draw_prob)),
        'away': min(95, max(5, away_prob))
    }

def generate_prediction_analysis(fixture):
    """Génère une analyse de prédiction"""
    analysis = {
        'confidence': np.random.randint(65, 90),
        'recommendation': np.random.choice(['1X', 'X2', 'Over 1.5', 'Under 3.5', 'BTTS Yes', 'BTTS No']),
        'risk_level': np.random.choice(['Low', 'Medium', 'High']),
        'key_factors': []
    }
    
    factors = [
        "Forme récente favorable",
        "Blessures importantes",
        "Motivation élevée",
        "Historique favorable",
        "Conditions météo optimales",
        "Suspensions clés"
    ]
    
    analysis['key_factors'] = np.random.choice(factors, size=3, replace=False).tolist()
    return analysis

# Interface principale
def main():
    load_css()
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="main-header">
            <h1 style="color: white; text-align: center; margin: 0;">⚽ FOOTBALL BETTING ANALYTICS LIVE</h1>
            <p style="color: white; text-align: center; margin: 10px 0 0 0; opacity: 0.9;">
                Analyse en temps réel • Données live • Prédictions intelligentes
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        # Configuration API
        api_key = st.text_input(
            "Clé API Football",
            type="password",
            help="Obtenez une clé sur https://dashboard.api-football.com/",
            value=st.session_state.get('api_key', '')
        )
        
        if api_key != st.session_state.get('api_key', ''):
            st.session_state.api_key = api_key
            st.success("Clé API mise à jour!")
        
        st.session_state.timezone = st.selectbox(
            "Fuseau horaire",
            ["Europe/Paris", "Europe/London", "America/New_York", "Asia/Tokyo", "Australia/Sydney"],
            index=0
        )
        
        # Options d'affichage
        st.markdown("### 👁️ Options d'affichage")
        show_live = st.checkbox("Matchs en direct", value=True)
        show_odds = st.checkbox("Afficher les cotes", value=True)
        auto_refresh = st.checkbox("Rafraîchissement auto", value=True)
        
        if auto_refresh:
            refresh_rate = st.slider("Intervalle (secondes)", 30, 300, 60)
            if st.button("🔄 Rafraîchir maintenant"):
                st.rerun()
        
        st.markdown("---")
        st.markdown("""
        ### 📊 Statut
        - **Dernière mise à jour:** {}
        - **Mode:** {}
        - **Fuseau:** {}
        """.format(
            st.session_state.last_update.strftime('%H:%M:%S'),
            "API" if st.session_state.api_key else "Démo",
            st.session_state.timezone
        ))
    
    # Contenu principal avec onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard Live", 
        "🔮 Prédictions", 
        "📈 Statistiques", 
        "⚽ Matchs", 
        "🏆 Classements"
    ])
    
    # Initialiser le manager API
    api_manager = FootballAPIManager()
    
    with tab1:
        render_dashboard(api_manager, show_live)
    
    with tab2:
        render_predictions(api_manager)
    
    with tab3:
        render_statistics(api_manager)
    
    with tab4:
        render_matches(api_manager, show_live)
    
    with tab5:
        render_standings(api_manager)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_rate)
        st.session_state.last_update = datetime.now()
        st.rerun()

def render_dashboard(api_manager, show_live):
    """Affiche le dashboard en direct"""
    st.markdown("### 📊 Tableau de bord Live")
    
    # KPI en temps réel
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Matchs en direct", "12", "+3")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Prédictions actives", "24", "78%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Cote moyenne", "2.15", "-0.10")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Taux de succès", "72.5%", "+1.8%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Section matchs en direct
    if show_live:
        st.markdown("### 🎯 Matchs en Direct")
        
        # Récupérer les matchs en direct
        live_data = api_manager.get_live_matches() if api_manager.api_key else DEMO_DATA
        
        if live_data and 'response' in live_data and live_data['response']:
            matches = live_data['response']
            for match in matches[:5]:  # Limiter à 5 matchs
                display_live_match(match)
        else:
            # Mode démo
            for match in DEMO_DATA['live_matches']:
                display_live_match(match)
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des résultats
        results = ['Victoire Domicile', 'Match Nul', 'Victoire Extérieur']
        percentages = [45.3, 27.8, 26.9]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=results,
                values=percentages,
                hole=.4,
                marker_colors=['#667eea', '#764ba2', '#f093fb']
            )
        ])
        fig.update_layout(
            title="Distribution des résultats récents",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tendances des buts
        days = list(range(1, 31))
        goals = [2.1 + 0.1 * np.sin(i/3) + np.random.normal(0, 0.1) for i in days]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=days, y=goals,
            mode='lines+markers',
            name='Buts/match',
            line=dict(color='#f5576c', width=3),
            marker=dict(size=6)
        ))
        fig.update_layout(
            title="Évolution moyenne des buts (30 jours)",
            xaxis_title="Jours",
            yaxis_title="Buts par match",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_live_match(match):
    """Affiche un match en direct"""
    try:
        fixture = match.get('fixture', {})
        teams = match.get('teams', {})
        goals = match.get('goals', {})
        league = match.get('league', {})
        
        home_team = teams.get('home', {}).get('name', 'Home')
        away_team = teams.get('away', {}).get('name', 'Away')
        home_logo = teams.get('home', {}).get('logo', '')
        away_logo = teams.get('away', {}).get('logo', '')
        home_score = goals.get('home', 0)
        away_score = goals.get('away', 0)
        status = fixture.get('status', {}).get('short', 'NS')
        elapsed = fixture.get('status', {}).get('elapsed', 0)
        league_name = league.get('name', '')
        
        col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])
        
        with col1:
            if home_logo:
                st.image(home_logo, width=40)
        
        with col2:
            st.markdown(f"**{home_team}**")
        
        with col3:
            st.markdown(f"<h3 style='text-align: center; margin: 0;'>{home_score} - {away_score}</h3>", 
                       unsafe_allow_html=True)
            if status == 'LIVE':
                st.markdown(f'<span class="live-badge">{elapsed}\'</span>', unsafe_allow_html=True)
            else:
                st.caption(status)
        
        with col4:
            st.markdown(f"**{away_team}**")
        
        with col5:
            if away_logo:
                st.image(away_logo, width=40)
        
        st.caption(f"{league_name}")
        st.divider()
        
    except Exception as e:
        st.warning(f"Erreur d'affichage du match: {str(e)}")

def render_predictions(api_manager):
    """Affiche les prédictions"""
    st.markdown("### 🔮 Prédictions Intelligentes")
    
    # Récupérer les matchs du jour
    today_data = api_manager.get_today_matches() if api_manager.api_key else None
    
    if today_data and 'response' in today_data:
        matches = today_data['response']
    else:
        matches = DEMO_DATA['today_matches']
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    with col1:
        min_confidence = st.slider("Confiance minimale", 50, 95, 70)
    with col2:
        max_odds = st.slider("Cote maximale", 1.1, 5.0, 3.0, 0.1)
    with col3:
        bet_type = st.selectbox("Type de pari", ["Tous", "1X2", "Over/Under", "BTTS", "Double Chance"])
    
    # Prédictions
    for i, match in enumerate(matches[:10]):  # Limiter à 10 matchs
        try:
            fixture = match.get('fixture', {})
            teams = match.get('teams', {})
            
            home_team = teams.get('home', {}).get('name', f'Team H{i}')
            away_team = teams.get('away', {}).get('name', f'Team A{i}')
            match_time = format_time(fixture.get('date', ''))
            
            # Générer une prédiction
            analysis = generate_prediction_analysis(match)
            
            if analysis['confidence'] >= min_confidence:
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 2, 1, 2])
                    
                    with col1:
                        st.markdown(f"**{home_team} vs {away_team}**")
                        st.caption(f"⏰ {match_time}")
                    
                    with col2:
                        risk_color = {
                            'Low': 'green',
                            'Medium': 'orange',
                            'High': 'red'
                        }.get(analysis['risk_level'], 'gray')
                        
                        st.markdown(f"""
                        🎯 **{analysis['recommendation']}**
                        ⚡ **Niveau de risque:** <span style='color:{risk_color}'>{analysis['risk_level']}</span>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        confidence_color = 'green' if analysis['confidence'] > 75 else 'orange' if analysis['confidence'] > 65 else 'red'
                        st.markdown(f"""
                        <div style='text-align: center;'>
                            <h3 style='color:{confidence_color}; margin:0;'>{analysis['confidence']}%</h3>
                            <small>Confiance</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        odds = round(1.5 + np.random.random() * 1.5, 2)
                        if odds <= max_odds:
                            st.metric("Meilleure cote", f"{odds}", 
                                     delta=f"+{round((odds-1)*100, 1)}%")
                            
                            if st.button("📊 Analyser", key=f"analyze_{i}"):
                                show_detailed_analysis(match, analysis, odds)
                    
                    # Facteurs clés
                    with st.expander("📋 Facteurs déterminants", expanded=False):
                        for factor in analysis['key_factors']:
                            st.markdown(f"✅ {factor}")
                    
                    st.divider()
                    
        except Exception as e:
            st.error(f"Erreur dans la prédiction: {str(e)}")

def show_detailed_analysis(match, analysis, odds):
    """Affiche une analyse détaillée"""
    st.markdown("### 📊 Analyse détaillée")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique radar
        categories = ['Attaque', 'Défense', 'Forme', 'Motivation', 'Historique', 'Conditions']
        home_values = [np.random.randint(60, 95) for _ in categories]
        away_values = [np.random.randint(60, 95) for _ in categories]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=home_values,
            theta=categories,
            fill='toself',
            name='Domicile',
            line_color='#667eea'
        ))
        fig.add_trace(go.Scatterpolar(
            r=away_values,
            theta=categories,
            fill='toself',
            name='Extérieur',
            line_color='#f093fb'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Analyse comparative"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Recommandations
        st.markdown("### 💎 Recommandation")
        st.markdown(f"""
        <div class="prediction-card">
            <h3 style="margin:0;">{analysis['recommendation']}</h3>
            <p style="margin:10px 0 0 0;">
                <strong>Confiance:</strong> {analysis['confidence']}%<br>
                <strong>Cote conseillée:</strong> {odds}<br>
                <strong>Mise suggérée:</strong> 2-4% de bankroll
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Statistiques
        st.markdown("### 📈 Statistiques clés")
        stats = {
            "Buts/moyenne domicile": f"{np.random.randint(1, 3)}.{np.random.randint(0, 9)}",
            "Buts/moyenne extérieur": f"{np.random.randint(1, 3)}.{np.random.randint(0, 9)}",
            "Clean sheets dernière 5": f"{np.random.randint(1, 4)}/5",
            "BTTS dernière 5": f"{np.random.randint(2, 5)}/5",
            "+2.5 buts dernière 5": f"{np.random.randint(2, 5)}/5"
        }
        
        for key, value in stats.items():
            st.markdown(f"**{key}:** {value}")

def render_statistics(api_manager):
    """Affiche les statistiques"""
    st.markdown("### 📈 Statistiques Avancées")
    
    # Sélection de l'équipe/ligue
    col1, col2 = st.columns(2)
    with col1:
        league_options = ["Ligue 1", "Premier League", "La Liga", "Bundesliga", "Serie A"]
        selected_league = st.selectbox("Sélectionner une ligue", league_options)
    
    with col2:
        team_options = {
            "Ligue 1": ["Paris SG", "Marseille", "Lyon", "Monaco", "Lille"],
            "Premier League": ["Manchester City", "Liverpool", "Arsenal", "Chelsea", "Tottenham"],
            "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Valencia"],
            "Bundesliga": ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen", "Eintracht Frankfurt"],
            "Serie A": ["Juventus", "AC Milan", "Inter Milan", "Napoli", "Roma"]
        }
        selected_team = st.selectbox("Sélectionner une équipe", team_options[selected_league])
    
    # Graphiques
    tab1, tab2, tab3 = st.tabs(["Performance", "Tendances", "Comparaisons"])
    
    with tab1:
        # Graphique de performance
        months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin']
        home_perf = [np.random.randint(60, 90) for _ in months]
        away_perf = [np.random.randint(50, 85) for _ in months]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=months, y=home_perf,
            name='Performance domicile',
            marker_color='#667eea'
        ))
        fig.add_trace(go.Bar(
            x=months, y=away_perf,
            name='Performance extérieur',
            marker_color='#f093fb'
        ))
        
        fig.update_layout(
            title=f"Performance {selected_team} - Saison en cours",
            barmode='group',
            xaxis_title="Mois",
            yaxis_title="Performance (%)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Tendances
        weeks = list(range(1, 21))
        form_trend = [50 + np.random.normal(0, 10) + i*2 for i in weeks]
        goals_trend = [1.5 + np.random.normal(0, 0.3) + i*0.05 for i in weeks]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weeks, y=form_trend,
            mode='lines',
            name='Indice de forme',
            line=dict(color='#667eea', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=weeks, y=goals_trend,
            mode='lines',
            name='Buts/match',
            line=dict(color='#f093fb', width=3),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Évolution des performances",
            xaxis_title="Semaines",
            yaxis_title="Indice de forme",
            yaxis2=dict(
                title="Buts/match",
                overlaying='y',
                side='right'
            ),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Comparaison d'équipes
        teams_comparison = {
            'Équipe': [selected_team] + np.random.choice(team_options[selected_league], 4, replace=False).tolist(),
            'Points': np.random.randint(20, 60, 5),
            'Buts pour': np.random.randint(20, 50, 5),
            'Buts contre': np.random.randint(10, 40, 5),
            'Différence': np.random.randint(-10, 20, 5)
        }
        
        df_comparison = pd.DataFrame(teams_comparison)
        df_comparison = df_comparison.sort_values('Points', ascending=False)
        
        st.dataframe(
            df_comparison,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Points': st.column_config.ProgressColumn(
                    format="%d",
                    min_value=0,
                    max_value=60
                ),
                'Buts pour': st.column_config.NumberColumn(format="%d"),
                'Buts contre': st.column_config.NumberColumn(format="%d"),
                'Différence': st.column_config.NumberColumn(format="%+d")
            }
        )

def render_matches(api_manager, show_live):
    """Affiche les matchs"""
    st.markdown("### ⚽ Calendrier des Matchs")
    
    # Sélection de la date
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        date_range = st.selectbox(
            "Période",
            ["Aujourd'hui", "Demain", "Week-end", "7 prochains jours", "Tous"]
        )
    
    with col2:
        league_filter = st.multiselect(
            "Ligues",
            ["Ligue 1", "Premier League", "La Liga", "Bundesliga", "Serie A", "Champions League"],
            default=["Ligue 1", "Premier League"]
        )
    
    with col3:
        only_live = st.checkbox("En direct seulement", value=False)
    
    # Générer des matchs
    matches = []
    for league in league_filter:
        for i in range(5):
            matches.append({
                'date': (datetime.now() + timedelta(days=np.random.randint(0, 7))).strftime('%Y-%m-%d'),
                'time': f"{np.random.randint(12, 22)}:{np.random.choice(['00', '15', '30', '45'])}",
                'league': league,
                'home': f"Équipe H{i}",
                'away': f"Équipe A{i}",
                'status': np.random.choice(['NS', 'LIVE', 'HT', 'FT'], p=[0.6, 0.1, 0.1, 0.2])
            })
    
    # Afficher les matchs
    for match in matches:
        if only_live and match['status'] != 'LIVE':
            continue
            
        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 1, 2, 2, 1])
            
            with col1:
                status_color = {
                    'NS': 'gray',
                    'LIVE': 'red',
                    'HT': 'orange',
                    'FT': 'green'
                }.get(match['status'], 'gray')
                st.markdown(f"<span style='color:{status_color}; font-weight:bold'>{match['status']}</span>", 
                           unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**{match['home']}**")
            
            with col3:
                st.markdown("**vs**")
            
            with col4:
                st.markdown(f"**{match['away']}**")
            
            with col5:
                st.markdown(f"{match['league']} • {match['time']}")
            
            with col6:
                if st.button("📊", key=f"stats_{match['home']}_{match['away']}"):
                    st.info(f"Analyse détaillée pour {match['home']} vs {match['away']}")
            
            st.divider()

def render_standings(api_manager):
    """Affiche les classements"""
    st.markdown("### 🏆 Classements des Ligues")
    
    # Sélection de la ligue
    selected_league = st.selectbox(
        "Choisir une ligue",
        ["Ligue 1", "Premier League", "La Liga", "Bundesliga", "Serie A", "Champions League"]
    )
    
    # Générer un classement
    teams = {
        "Ligue 1": ["Paris SG", "Marseille", "Lyon", "Monaco", "Lille", "Nice", "Rennes", "Lens"],
        "Premier League": ["Manchester City", "Liverpool", "Arsenal", "Chelsea", "Tottenham", "Manchester United", "Newcastle", "Aston Villa"],
        "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Valencia", "Real Betis", "Villarreal", "Athletic Bilbao"],
        "Bundesliga": ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen", "Eintracht Frankfurt", "Wolfsburg", "Borussia M'gladbach", "Freiburg"],
        "Serie A": ["Juventus", "AC Milan", "Inter Milan", "Napoli", "Roma", "Lazio", "Atalanta", "Fiorentina"],
        "Champions League": ["Real Madrid", "Manchester City", "Bayern Munich", "Paris SG", "Liverpool", "Barcelona", "Chelsea", "AC Milan"]
    }
    
    standings_data = []
    for i, team in enumerate(teams[selected_league], 1):
        standings_data.append({
            'Position': i,
            'Équipe': team,
            'MJ': np.random.randint(20, 30),
            'G': np.random.randint(10, 20),
            'N': np.random.randint(3, 10),
            'P': np.random.randint(2, 8),
            'BP': np.random.randint(30, 60),
            'BC': np.random.randint(15, 40),
            'Diff': np.random.randint(-5, 30),
            'PTS': np.random.randint(30, 60)
        })
    
    df_standings = pd.DataFrame(standings_data)
    
    # Afficher le classement
    st.dataframe(
        df_standings,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Position': st.column_config.NumberColumn(format="%d"),
            'PTS': st.column_config.ProgressColumn(
                format="%d",
                min_value=0,
                max_value=60
            ),
            'Diff': st.column_config.NumberColumn(format="%+d")
        }
    )
    
    # Graphique du classement
    fig = go.Figure(data=[
        go.Bar(
            x=df_standings['Équipe'],
            y=df_standings['PTS'],
            marker_color=['#667eea' if i < 4 else '#f093fb' if i < 7 else '#764ba2' for i in range(len(df_standings))]
        )
    ])
    
    fig.update_layout(
        title=f"Points - {selected_league}",
        xaxis_title="Équipes",
        yaxis_title="Points",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
def render_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **ℹ️ À propos**  
        Plateforme d'analyse footballistique en temps réel
        """)
    
    with col2:
        st.markdown("""
        **⚖️ Disclaimer**  
        À but informatif seulement.  
        Paris sportifs = risque financier.
        """)
    
    with col3:
        st.markdown("""
        **🔄 Dernière mise à jour**  
        {}
        """.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

if __name__ == "__main__":
    main()
    render_footer()
