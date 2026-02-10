import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def render_predictions():
    st.markdown("## 🔮 Prédictions CIA (Conseils Intelligents d'Analyse)")
    
    # Sélecteur de date
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_option = st.radio(
            "Sélectionner la date",
            ["Aujourd'hui", "Hier", "Demain"],
            horizontal=True
        )
    
    with col2:
        min_success = st.slider("% réussite minimum", 0, 100, 70)
    
    with col3:
        min_matches = st.slider("Matchs analysés min", 1, 100, 15)
    
    # Filtres avancés
    with st.expander("Filtres avancés", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_odds = st.number_input("Cote minimum", 1.0, 10.0, 1.4, 0.05)
        
        with col2:
            max_odds = st.number_input("Cote maximum", 1.0, 10.0, 2.5, 0.05)
        
        with col3:
            leagues = st.multiselect(
                "Championnats",
                ["Ligue 1", "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 2"],
                default=["Ligue 1", "Premier League"]
            )
    
    # Tableau des prédictions
    predictions_data = {
        "Championnat": ["Ligue 1", "Premier League", "La Liga", "Bundesliga", "Serie A"] * 3,
        "Match": ["PSG vs Marseille", "Liverpool vs Man City", "Real vs Barca", 
                  "Bayern vs Dortmund", "Juventus vs Milan", "Lyon vs Monaco",
                  "Arsenal vs Chelsea", "Atletico vs Sevilla", "Leipzig vs Leverkusen",
                  "Inter vs Napoli", "Lille vs Nice", "Tottenham vs Man United",
                  "Valencia vs Betis", "Frankfurt vs Wolfsburg", "Roma vs Lazio"],
        "Prédiction": ["+1.5 buts", "Lay Draw", "BTTS Oui", "+2.5 buts", "1X",
                      "+1.5 buts", "Lay Draw", "BTTS Oui", "+2.5 buts", "1X",
                      "+1.5 buts", "Lay Draw", "BTTS Oui", "+2.5 buts", "1X"],
        "% Réussite": ["82%", "78%", "75%", "70%", "68%",
                      "85%", "80%", "77%", "72%", "69%",
                      "81%", "79%", "76%", "71%", "67%"],
        "Cote": ["1.65", "1.45", "1.80", "1.90", "1.55",
                "1.60", "1.48", "1.75", "1.85", "1.52",
                "1.63", "1.47", "1.78", "1.88", "1.50"],
        "Heure": ["15:00", "16:30", "18:00", "19:45", "21:00",
                 "14:00", "15:30", "17:15", "20:00", "21:30",
                 "13:45", "15:15", "16:45", "19:30", "22:00"]
    }
    
    df_predictions = pd.DataFrame(predictions_data)
    
    # Filtrer selon les paramètres
    df_filtered = df_predictions.copy()
    
    # Affichage
    st.dataframe(
        df_filtered,
        use_container_width=True,
        height=400,
        column_config={
            "% Réussite": st.column_config.ProgressColumn(
                format="%s",
                min_value=0,
                max_value=100,
                help="Taux de réussite historique"
            ),
            "Cote": st.column_config.NumberColumn(
                format="%.2f",
                help="Meilleure cote disponible"
            )
        }
    )
    
    # Graphique radar pour une prédiction
    st.markdown("### 📊 Analyse détaillée")
    
    selected_match = st.selectbox(
        "Sélectionner un match pour l'analyse",
        df_filtered["Match"].tolist()
    )
    
    if selected_match:
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique radar
            categories = ['Forme actuelle', 'Attaque', 'Défense', 
                         'Loi de poisson', 'H2H', 'Buts H2H', 'Prévision']
            
            team_home = [85, 78, 72, 80, 75, 82, 88]
            team_away = [70, 82, 68, 65, 60, 58, 65]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=team_home,
                theta=categories,
                fill='toself',
                name='Équipe Domicile',
                line_color='#38a9ff'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=team_away,
                theta=categories,
                fill='toself',
                name='Équipe Extérieur',
                line_color='#FF9800'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Analyse comparative"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Statistiques détaillées
            st.markdown("#### 📈 Statistiques clés")
            
            stats = {
                "Forme dernière 5 matchs": "3V, 1N, 1D",
                "Buts marqués/moyenne": "12 / 2.4 par match",
                "Buts encaissés/moyenne": "6 / 1.2 par match",
                "BTTS dernière 5": "4 sur 5",
                "+1.5 buts dernière 5": "5 sur 5",
                "+2.5 buts dernière 5": "3 sur 5",
                "Clean sheets": "2 sur 5"
            }
            
            for key, value in stats.items():
                st.markdown(f"**{key}:** {value}")
            
            # Recommandation
            st.markdown("---")
            st.markdown("#### 🎯 Recommandation")
            
            confidence = 82  # Exemple
            if confidence >= 80:
                color = "green"
                emoji = "✅"
            elif confidence >= 70:
                color = "orange"
                emoji = "⚠️"
            else:
                color = "red"
                emoji = "❌"
            
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {color};">
                <h4 style="color: {color}; margin-top: 0;">{emoji} Confiance: <strong>{confidence}%</strong></h4>
                <p><strong>Pari recommandé:</strong> +1.5 buts</p>
                <p><strong>Cote optimale:</strong> 1.65</p>
                <p><strong>Bankroll suggérée:</strong> 3-5%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Bouton d'export
    if st.button("📥 Exporter les prédictions (CSV)"):
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="Télécharger CSV",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
