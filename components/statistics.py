import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

def render_statistics():
    st.markdown("## 📈 Statistiques avancées")
    
    # Sélection du type de statistiques
    stat_type = st.selectbox(
        "Type de statistiques",
        ["Lay the Draw", "+1.5 buts", "BTTS (Both Teams To Score)", 
         "No Clean Sheet", "+2.5 buts", "-0.5 MT", "+0.5 MT", "-1.5 MT"]
    )
    
    # Filtres
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        min_matches = st.selectbox(
            "Matchs minimum",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            index=3
        )
    
    with col2:
        sort_by = st.selectbox(
            "Trier par",
            ["Prochain match", "Série max", "Taux réussite"]
        )
    
    with col3:
        sort_order = st.radio(
            "Ordre",
            ["Croissant", "Décroissant"],
            horizontal=True
        )
    
    with col4:
        show_records = st.checkbox("Records seulement", value=True)
        show_today = st.checkbox("Aujourd'hui seulement")
    
    # Données simulées
    teams = [
        "Paris SG", "Manchester City", "Bayern Munich", "Real Madrid", 
        "Liverpool", "Barcelona", "Juventus", "AC Milan", 
        "Atletico Madrid", "Chelsea", "Arsenal", "Tottenham",
        "Borussia Dortmund", "Inter Milan", "Napoli", "Lyon"
    ]
    
    leagues = ["Ligue 1", "Premier League", "Bundesliga", "La Liga"] * 4
    
    # Générer des données aléatoires mais réalistes
    import random
    
    data = []
    for i, (team, league) in enumerate(zip(teams, leagues)):
        current_series = random.randint(2, 8)
        max_series = random.randint(5, 15)
        success_rate = random.randint(65, 90)
        
        # Générer une série de résultats
        results = []
        for _ in range(current_series):
            r = random.random()
            if r > 0.7:
                results.append("W")
            elif r > 0.4:
                results.append("D")
            else:
                results.append("L")
        
        # Prochain match (aujourd'hui ou futur)
        if random.random() > 0.3:
            match_time = (datetime.now() + timedelta(days=random.randint(0, 7))).strftime("%d/%m/%Y %H:%M")
            is_today = "today" if random.random() > 0.7 else ""
        else:
            match_time = (datetime.now() - timedelta(days=random.randint(1, 3))).strftime("%d/%m/%Y %H:%M")
            is_today = ""
        
        opponent = random.choice([t for t in teams if t != team])
        odds = round(1.3 + random.random() * 0.7, 2)
        
        data.append({
            "Championnat": league,
            "Équipe": team,
            "Série en cours": " ".join(results),
            "Série max": f"{max_series} (2016+)",
            "Prochain match": f"{match_time} {is_today}",
            "Adversaire": opponent,
            "Cote": odds
        })
    
    df_stats = pd.DataFrame(data)
    
    # Affichage
    st.dataframe(
        df_stats,
        use_container_width=True,
        height=500,
        column_config={
            "Cote": st.column_config.NumberColumn(
                format="%.2f",
                help="Meilleure cote disponible"
            )
        }
    )
    
    # Visualisations
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des séries
        series_lengths = [len(team["Série en cours"].split()) for team in data]
        
        fig = go.Figure(data=[
            go.Histogram(
                x=series_lengths,
                nbinsx=10,
                marker_color='#38a9ff',
                opacity=0.7
            )
        ])
        
        fig.update_layout(
            title="Distribution des séries en cours",
            xaxis_title="Longueur de la série",
            yaxis_title="Nombre d'équipes",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Taux de réussite par championnat
        league_data = {}
        for team in data:
            league = team["Championnat"]
            if league not in league_data:
                league_data[league] = []
            # Simuler un taux de réussite basé sur la longueur de série
            success_rate = min(95, 70 + len(team["Série en cours"].split()) * 3)
            league_data[league].append(success_rate)
        
        avg_success = {league: sum(rates)/len(rates) for league, rates in league_data.items()}
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(avg_success.keys()),
                y=list(avg_success.values()),
                marker_color=['#38a9ff', '#4CAF50', '#FF9800', '#9C27B0']
            )
        ])
        
        fig.update_layout(
            title="Taux de réussite moyen par championnat",
            xaxis_title="Championnat",
            yaxis_title="Taux de réussite (%)",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Légende
    with st.expander("📖 Légende", expanded=False):
        st.markdown("""
        ### Symboles et significations :
        
        **Résultats des matchs :**
        - **W** = Win (Victoire)
        - **D** = Draw (Match nul)
        - **L** = Loss (Défaite)
        
        **Annotations :**
        - **(2016+)** = Données disponibles depuis 2016
        - ***Série max*** = Plus grande série historique
        - **•2020•** = Année du record
        
        **Couleurs des records :**
        - 🟢 **Vert** = Record > 8 ans (recommandé)
        - 🟡 **Orange** = Record 4-8 ans (à surveiller)
        - 🔴 **Rouge** = Record < 4 ans (déconseillé)
        
        **Indicateurs :**
        - 📈 = Série en hausse
        - 📉 = Série en baisse
        - ⭐ = Équipe favorite
        - 🔔 = Alerte activée
        """)
    
    # Boutons d'action
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Actualiser les données"):
            st.success("Données actualisées!")
            st.rerun()
    
    with col2:
        if st.button("🔔 Configurer alertes"):
            st.info("Configuration des alertes")
    
    with col3:
        if st.button("📊 Exporter données"):
            csv = df_stats.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name=f"statistics_{stat_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
