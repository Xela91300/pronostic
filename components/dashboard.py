import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

def render_dashboard():
    st.markdown("## 📊 Tableau de bord principal")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Taux de réussite",
            value="72.5%",
            delta="+2.3%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="Prédictions aujourd'hui",
            value="17",
            delta="+3",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="Gain moyen",
            value="+28.5€",
            delta="-3.2€",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="Séries actives",
            value="8",
            delta="+1",
            delta_color="normal"
        )
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique de distribution des paris
        bet_types = ['+1.5 buts', 'Lay Draw', 'BTTS', '+2.5 buts', '1X']
        success_rates = [78, 72, 65, 70, 68]
        
        fig = go.Figure(data=[
            go.Bar(
                x=bet_types,
                y=success_rates,
                marker_color=['#38a9ff', '#4CAF50', '#FF9800', '#9C27B0', '#E91E63']
            )
        ])
        
        fig.update_layout(
            title="Taux de réussite par type de pari",
            xaxis_title="Type de pari",
            yaxis_title="Taux de réussite (%)",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Graphique circulaire des ligues
        leagues = ['Ligue 1', 'Premier League', 'La Liga', 'Bundesliga', 'Serie A']
        matches = [45, 38, 35, 32, 30]
        
        fig = go.Figure(data=[go.Pie(
            labels=leagues,
            values=matches,
            hole=.3,
            marker_colors=px.colors.qualitative.Set3
        )])
        
        fig.update_layout(
            title="Répartition par championnat",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Matchs du jour
    st.markdown("### ⚽ Matchs du jour")
    
    matches_today = [
        {
            "time": "15:00",
            "home": "Paris SG",
            "away": "Marseille",
            "prediction": "+1.5 buts",
            "odds": "1.65",
            "confidence": "82%"
        },
        {
            "time": "16:30",
            "home": "Real Madrid",
            "away": "Barcelona",
            "prediction": "Lay the Draw",
            "odds": "1.45",
            "confidence": "78%"
        },
        {
            "time": "18:00",
            "home": "Bayern Munich",
            "away": "Borussia Dortmund",
            "prediction": "BTTS Oui",
            "odds": "1.80",
            "confidence": "75%"
        },
        {
            "time": "20:45",
            "home": "Liverpool",
            "away": "Manchester City",
            "prediction": "+2.5 buts",
            "odds": "1.90",
            "confidence": "70%"
        }
    ]
    
    for match in matches_today:
        with st.container():
            cols = st.columns([1, 2, 2, 2, 2, 1])
            
            with cols[0]:
                st.markdown(f"**{match['time']}**")
            
            with cols[1]:
                st.markdown(f"**{match['home']}**")
            
            with cols[2]:
                st.markdown(f"**VS**")
            
            with cols[3]:
                st.markdown(f"**{match['away']}**")
            
            with cols[4]:
                st.markdown(f"`{match['prediction']}`")
            
            with cols[5]:
                confidence_color = "green" if float(match['confidence'].strip('%')) > 75 else "orange" if float(match['confidence'].strip('%')) > 65 else "red"
                st.markdown(f"<span style='color: {confidence_color}; font-weight: bold'>{match['confidence']}</span>", unsafe_allow_html=True)
            
            st.divider()
    
    # Informations rapides
    with st.expander("ℹ️ Informations sur la plateforme", expanded=False):
        st.markdown("""
        **Statut :** ✅ En ligne  
        **Dernière mise à jour :** {}  
        **Ligues couvertes :** 23 championnats  
        **Base de données :** 10,000+ matchs analysés
        
        ### 📈 Méthodologie :
        1. **Analyse des séries statistiques**
        2. **Calcul des probabilités**
        3. **Vérification des cotes**
        4. **Recommandations personnalisées**
        
        ### ⚠️ Disclaimer :
        Les informations sont fournies à titre indicatif seulement.
        Les paris sportifs comportent des risques de perte financière.
        """.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
