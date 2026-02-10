import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Import des composants
from components.header import render_header, render_sidebar
from components.dashboard import render_dashboard
from components.predictions import render_predictions
from components.statistics import render_statistics

# Configuration de la page
st.set_page_config(
    page_title="Football Betting Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
def load_css():
    with open('assets/css/custom.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialisation de session
if 'timezone' not in st.session_state:
    st.session_state.timezone = 'Europe/Paris'
if 'language' not in st.session_state:
    st.session_state.language = 'fr'
if 'show_tutorial' not in st.session_state:
    st.session_state.show_tutorial = True

# Charger le CSS
load_css()

# Titre principal
st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #38a9ff, #0066cc); border-radius: 10px; margin-bottom: 30px;">
        <h1 style="color: white; margin: 0;">⚽ Football Betting Analytics</h1>
        <p style="color: white; opacity: 0.9; margin-top: 10px;">Outils d'analyse automatisés pour les paris sportifs</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar avec navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2784/2784403.png", width=100)
    
    st.markdown("## Navigation")
    
    # Sélection de la page
    page = st.radio(
        "Menu principal",
        ["Dashboard", "Prédictions CIA", "Statistiques", "Top 50", "Alertes", "Historique"]
    )
    
    # Options de configuration
    st.markdown("---")
    st.markdown("### Configuration")
    
    timezone = st.selectbox(
        "Fuseau horaire",
        ["Europe/Paris", "Europe/London", "America/New_York", "Asia/Tokyo", "Australia/Sydney"],
        index=0
    )
    
    language = st.selectbox(
        "Langue",
        ["Français", "English", "Español"],
        index=0
    )
    
    if st.button("Appliquer"):
        st.session_state.timezone = timezone
        st.session_state.language = language
        st.success("Configuration mise à jour!")
        time.sleep(1)
        st.rerun()

# Contenu principal basé sur la page sélectionnée
if page == "Dashboard":
    render_dashboard()
    
elif page == "Prédictions CIA":
    render_predictions()
    
elif page == "Statistiques":
    render_statistics()
    
elif page == "Top 50":
    st.markdown("## Top 50 des meilleures statistiques")
    
    # Sélection du type de statistiques
    stat_type = st.selectbox(
        "Type de statistiques",
        ["Lay the Draw", "+1.5 buts", "+2.5 buts", "BTTS", "No Clean Sheet"]
    )
    
    # Simulation de données
    data = {
        "Position": list(range(1, 51)),
        "Équipe": [f"Équipe {i}" for i in range(1, 51)],
        "Championnat": ["Ligue 1", "Premier League", "La Liga", "Bundesliga", "Serie A"] * 10,
        "Taux historique": [f"{round(75 + i*0.5, 1)}%" for i in range(50)],
        "Taux saison actuelle": [f"{round(70 + i*0.8, 1)}%" for i in range(50)],
        "Série en cours": ["W W W W W", "D D D D", "W D W D", "L L L", "W W W"] * 10,
        "Prochain match": [f"{(datetime.now() + timedelta(days=i)).strftime('%d/%m/%Y %H:%M')}" for i in range(50)]
    }
    
    df = pd.DataFrame(data)
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    with col1:
        min_games = st.slider("Matchs minimum", 1, 100, 20)
    with col2:
        sort_by = st.selectbox("Trier par", ["Taux historique", "Taux saison", "Position"])
    with col3:
        show_today = st.checkbox("Aujourd'hui seulement", value=True)
    
    # Affichage du tableau
    st.dataframe(
        df,
        use_container_width=True,
        height=600,
        column_config={
            "Position": st.column_config.NumberColumn(format="%d"),
            "Taux historique": st.column_config.ProgressColumn(
                format="%s",
                min_value=0,
                max_value=100
            ),
            "Taux saison actuelle": st.column_config.ProgressColumn(
                format="%s",
                min_value=0,
                max_value=100
            )
        }
    )
    
elif page == "Alertes":
    st.markdown("## Alertes automatiques")
    
    # Configuration des alertes
    with st.expander("Configurer les alertes", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.checkbox("Lay the Draw", value=True)
            st.checkbox("+1.5 buts", value=True)
            st.checkbox("BTTS", value=False)
            
        with col2:
            st.checkbox("No Clean Sheet", value=True)
            st.checkbox("+2.5 buts", value=False)
            st.checkbox("-0.5 MT", value=True)
            
        with col3:
            min_odds = st.number_input("Cote minimum", 1.0, 10.0, 1.5, 0.1)
            min_success_rate = st.number_input("% réussite min", 0, 100, 70)
    
    # Liste des alertes
    st.markdown("### Alertes actives")
    
    alerts = [
        {"time": "14:30", "match": "PSG vs Marseille", "type": "+1.5 buts", "odds": "1.65", "confidence": "82%"},
        {"time": "15:00", "match": "Real Madrid vs Barcelona", "type": "Lay the Draw", "odds": "1.45", "confidence": "78%"},
        {"time": "17:00", "match": "Bayern vs Dortmund", "type": "BTTS", "odds": "1.80", "confidence": "75%"},
        {"time": "20:00", "match": "Liverpool vs Man City", "type": "+2.5 buts", "odds": "1.90", "confidence": "70%"},
    ]
    
    for alert in alerts:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 2, 2])
            with col1:
                st.markdown(f"**{alert['time']}**")
            with col2:
                st.markdown(f"**{alert['match']}**")
            with col3:
                st.markdown(f"`{alert['type']}`")
            with col4:
                st.markdown(f"📈 {alert['odds']}")
            with col5:
                confidence_color = "green" if float(alert['confidence'].strip('%')) > 75 else "orange" if float(alert['confidence'].strip('%')) > 65 else "red"
                st.markdown(f"<span style='color: {confidence_color}; font-weight: bold'>{alert['confidence']}</span>", unsafe_allow_html=True)
            st.divider()
    
elif page == "Historique":
    st.markdown("## Historique des prédictions")
    
    # Sélection de la période
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Date début", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("Date fin", datetime.now())
    with col3:
        result_filter = st.selectbox("Résultat", ["Tous", "Gagnants", "Perdants"])
    
    # Graphique de performance
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    success_rates = [65 + i*0.5 + (i%7)*2 for i in range(30)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=success_rates,
        mode='lines+markers',
        name='Taux de réussite',
        line=dict(color='#38a9ff', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Évolution du taux de réussite",
        xaxis_title="Date",
        yaxis_title="Taux de réussite (%)",
        template="plotly_white",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau des résultats
    st.markdown("### Détail des prédictions")
    
    history_data = {
        "Date": [(datetime.now() - timedelta(days=i)).strftime('%d/%m/%Y') for i in range(30)],
        "Match": [f"Match {i}" for i in range(30)],
        "Prédiction": ["+1.5", "Lay Draw", "BTTS", "+2.5", "1X"] * 6,
        "Cote": [round(1.3 + i*0.02, 2) for i in range(30)],
        "Résultat": ["✅", "❌", "✅", "✅", "❌"] * 6,
        "Gains": [f"+{round(10 * (1.3 + i*0.02 - 1), 2)}€" for i in range(30)]
    }
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True, height=400)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em; padding: 20px;">
        <p>© 2024 Football Betting Analytics | Développé avec Streamlit</p>
        <p style="margin-top: 10px;">
            <a href="#" style="color: #38a9ff; text-decoration: none; margin: 0 10px;">Contact</a> |
            <a href="#" style="color: #38a9ff; text-decoration: none; margin: 0 10px;">Documentation</a> |
            <a href="#" style="color: #38a9ff; text-decoration: none; margin: 0 10px;">GitHub</a>
        </p>
    </div>
""", unsafe_allow_html=True)
