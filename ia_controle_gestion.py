import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="IA Contrôle de Gestion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    h1, h2, h3 {
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: white !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.2);
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
    }
    .stTabs [data-baseweb="tab"] button {
        color: white !important;
    }
    .stTabs [data-baseweb="tab"] button p {
        color: white !important;
    }
</style>


""", unsafe_allow_html=True)

# CSS personnalisé
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# Génération de données financières
@st.cache_data
def generate_financial_data():
    np.random.seed(42)
    months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
    
    data = []
    for i, month in enumerate(months):
        base = 100000 + i * 5000
        variance = np.random.randn() * 10000
        actual_spend = base + variance
        budgeted = base + 5000
        historical_spend = base - 5000 + np.random.randn() * 5000
        
        data.append({
            'Mois': month,
            'Mois_Num': i,
            'Dépenses_Réelles': round(actual_spend),
            'Budget': round(budgeted),
            'Historique': round(historical_spend),
            'Variance': round(actual_spend - budgeted)
        })
    
    df = pd.DataFrame(data)
    return df

# Détection d'anomalies avec Z-Score
def detect_anomalies(df, column='Variance', threshold=2):
    mean = df[column].mean()
    std = df[column].std()
    df['Z_Score'] = (df[column] - mean) / std
    df['Anomalie'] = np.abs(df['Z_Score']) > threshold
    return df

# Modèle de régression linéaire
def train_linear_regression(df):
    X = df[['Mois_Num']].values
    y = df['Dépenses_Réelles'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    predictions = model.predict(X)
    df['Prédiction_Linéaire'] = predictions
    
    # Intervalle de confiance
    residuals = y - predictions
    std_residuals = np.std(residuals)
    df['Conf_Sup'] = predictions + 1.96 * std_residuals
    df['Conf_Inf'] = predictions - 1.96 * std_residuals
    
    return model, df

# Modèle Random Forest
def train_random_forest(df):
    X = df[['Mois_Num', 'Budget', 'Historique']].values
    y = df['Dépenses_Réelles'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    predictions = rf_model.predict(X)
    df['Prédiction_RF'] = predictions
    
    # Métriques
    y_pred_test = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    return rf_model, df, mse, r2

# Titre principal
st.title("🧠 IA Contrôle de Gestion")
st.markdown("### Tableau de bord analytique avec Machine Learning et détection d'anomalies")

# Génération des données
df = generate_financial_data()
df = detect_anomalies(df)
lr_model, df = train_linear_regression(df)
rf_model, df, rf_mse, rf_r2 = train_random_forest(df)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("⚙️ Configuration")
    
    anomaly_threshold = st.slider(
        "Seuil de détection (Z-Score)",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.1
    )
    
    df = detect_anomalies(df, threshold=anomaly_threshold)
    
    st.markdown("---")
    st.markdown("### 📊 Modèles Disponibles")
    st.markdown("""
    - ✅ Régression Linéaire
    - ✅ Random Forest
    - ✅ Détection Anomalies
    - ✅ ARIMA (conceptuel)
    """)
    
    st.markdown("---")
    if st.button("🔄 Régénérer les données"):
        st.cache_data.clear()
        st.rerun()

# KPIs
col1, col2, col3, col4 = st.columns(4)

total_spend = df['Dépenses_Réelles'].sum()
total_budget = df['Budget'].sum()
avg_variance = df['Variance'].abs().mean()
anomaly_count = df['Anomalie'].sum()

with col1:
    st.metric(
        label="💰 Dépenses Totales",
        value=f"{total_spend/1000000:.1f}M€",
        delta=f"{((total_spend - total_budget)/total_budget*100):.1f}%"
    )

with col2:
    st.metric(
        label="📊 Budget Total",
        value=f"{total_budget/1000000:.1f}M€"
    )

with col3:
    st.metric(
        label="📈 Écart Moyen",
        value=f"{avg_variance/1000:.0f}K€"
    )

with col4:
    st.metric(
        label="⚠️ Anomalies",
        value=f"{anomaly_count}",
        delta="Alertes détectées",
        delta_color="inverse"
    )

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Vue d'ensemble", 
    "🤖 Prédictions ML", 
    "⚠️ Anomalies", 
    "🏗️ Architecture",
    "📥 Données"
])

# TAB 1: Vue d'ensemble
with tab1:
    st.header("Vue d'ensemble des finances")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique ligne - Dépenses vs Budget
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=df['Mois'], y=df['Dépenses_Réelles'],
            mode='lines+markers',
            name='Dépenses Réelles',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8)
        ))
        
        fig1.add_trace(go.Scatter(
            x=df['Mois'], y=df['Budget'],
            mode='lines+markers',
            name='Budget',
            line=dict(color='#10b981', width=3),
            marker=dict(size=8)
        ))
        
        fig1.add_trace(go.Scatter(
            x=df['Mois'], y=df['Historique'],
            mode='lines',
            name='Historique',
            line=dict(color='#64748b', width=2, dash='dash')
        ))
        
        fig1.update_layout(
            title="Dépenses vs Budget",
            xaxis_title="Mois",
            yaxis_title="Montant (€)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Graphique barres - Variances
        fig2 = go.Figure()
        
        colors = ['#ef4444' if v > 0 else '#10b981' for v in df['Variance']]
        
        fig2.add_trace(go.Bar(
            x=df['Mois'],
            y=df['Variance'],
            marker_color=colors,
            name='Variance'
        ))
        
        fig2.add_hline(y=0, line_dash="dash", line_color="white")
        
        fig2.update_layout(
            title="Variances Mensuelles",
            xaxis_title="Mois",
            yaxis_title="Écart (€)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)

# TAB 2: Prédictions ML
with tab2:
    st.header("Modèles de Machine Learning")
    
    # Informations sur les modèles
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Régression Linéaire**
        - Équation: y = {lr_model.coef_[0]:.2f}x + {lr_model.intercept_:.2f}
        - Tendance: +{lr_model.coef_[0]/1000:.1f}K€ par mois
        """)
    
    with col2:
        st.success(f"""
        **Random Forest**
        - MSE: {rf_mse:.2f}
        - R² Score: {rf_r2:.4f}
        - 100 arbres de décision
        """)
    
    # Graphique comparatif des modèles
    fig3 = go.Figure()
    
    fig3.add_trace(go.Scatter(
        x=df['Mois'], y=df['Dépenses_Réelles'],
        mode='lines+markers',
        name='Réel',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=10)
    ))
    
    fig3.add_trace(go.Scatter(
        x=df['Mois'], y=df['Prédiction_Linéaire'],
        mode='lines',
        name='Prédiction Linéaire',
        line=dict(color='#ef4444', width=2, dash='dash')
    ))
    
    fig3.add_trace(go.Scatter(
        x=df['Mois'], y=df['Prédiction_RF'],
        mode='lines',
        name='Prédiction Random Forest',
        line=dict(color='#8b5cf6', width=2, dash='dot')
    ))
    
    # Intervalle de confiance
    fig3.add_trace(go.Scatter(
        x=df['Mois'], y=df['Conf_Sup'],
        mode='lines',
        name='Conf. Sup (95%)',
        line=dict(color='rgba(100, 116, 139, 0.3)', width=1),
        showlegend=False
    ))
    
    fig3.add_trace(go.Scatter(
        x=df['Mois'], y=df['Conf_Inf'],
        mode='lines',
        name='Conf. Inf (95%)',
        line=dict(color='rgba(100, 116, 139, 0.3)', width=1),
        fill='tonexty',
        fillcolor='rgba(100, 116, 139, 0.2)',
        showlegend=True
    ))
    
    fig3.update_layout(
        title="Comparaison des Modèles de Prédiction",
        xaxis_title="Mois",
        yaxis_title="Montant (€)",
        template="plotly_dark",
        height=500
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Scatter plot - Réel vs Prédit
    col1, col2 = st.columns(2)
    
    with col1:
        fig4 = px.scatter(
            df, x='Dépenses_Réelles', y='Prédiction_Linéaire',
            title="Régression Linéaire - Réel vs Prédit",
            labels={'Dépenses_Réelles': 'Réel (€)', 'Prédiction_Linéaire': 'Prédit (€)'},
            template="plotly_dark"
        )
        fig4.add_trace(go.Scatter(
            x=[df['Dépenses_Réelles'].min(), df['Dépenses_Réelles'].max()],
            y=[df['Dépenses_Réelles'].min(), df['Dépenses_Réelles'].max()],
            mode='lines',
            name='Parfait',
            line=dict(color='#10b981', dash='dash')
        ))
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        fig5 = px.scatter(
            df, x='Dépenses_Réelles', y='Prédiction_RF',
            title="Random Forest - Réel vs Prédit",
            labels={'Dépenses_Réelles': 'Réel (€)', 'Prédiction_RF': 'Prédit (€)'},
            template="plotly_dark",
            color='Anomalie',
            color_discrete_map={True: '#ef4444', False: '#8b5cf6'}
        )
        fig5.add_trace(go.Scatter(
            x=[df['Dépenses_Réelles'].min(), df['Dépenses_Réelles'].max()],
            y=[df['Dépenses_Réelles'].min(), df['Dépenses_Réelles'].max()],
            mode='lines',
            name='Parfait',
            line=dict(color='#10b981', dash='dash')
        ))
        st.plotly_chart(fig5, use_container_width=True)

# TAB 3: Anomalies
with tab3:
    st.header("Détection d'Anomalies (Z-Score)")
    
    st.warning(f"⚠️ **{anomaly_count} anomalies détectées** avec un seuil |Z-Score| > {anomaly_threshold}")
    
    # Distribution des Z-Scores
    fig6 = go.Figure()
    
    colors_z = ['#ef4444' if a else '#10b981' for a in df['Anomalie']]
    
    fig6.add_trace(go.Bar(
        x=df['Mois'],
        y=df['Z_Score'],
        marker_color=colors_z,
        name='Z-Score'
    ))
    
    fig6.add_hline(y=anomaly_threshold, line_dash="dash", line_color="red", 
                   annotation_text=f"Seuil +{anomaly_threshold}")
    fig6.add_hline(y=-anomaly_threshold, line_dash="dash", line_color="red",
                   annotation_text=f"Seuil -{anomaly_threshold}")
    
    fig6.update_layout(
        title="Distribution des Z-Scores",
        xaxis_title="Mois",
        yaxis_title="Z-Score",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig6, use_container_width=True)
    
    # Tableau des anomalies
    st.subheader("Détail des Anomalies")
    
    anomaly_df = df[df['Anomalie']].copy()
    
    if len(anomaly_df) > 0:
        anomaly_display = anomaly_df[['Mois', 'Dépenses_Réelles', 'Budget', 'Variance', 'Z_Score']].copy()
        anomaly_display['Dépenses_Réelles'] = anomaly_display['Dépenses_Réelles'].apply(lambda x: f"{x/1000:.0f}K€")
        anomaly_display['Budget'] = anomaly_display['Budget'].apply(lambda x: f"{x/1000:.0f}K€")
        anomaly_display['Variance'] = anomaly_display['Variance'].apply(lambda x: f"{x/1000:+.0f}K€")
        anomaly_display['Z_Score'] = anomaly_display['Z_Score'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(anomaly_display, use_container_width=True, hide_index=True)
    else:
        st.success("✅ Aucune anomalie détectée avec ce seuil!")
    
    # Tableau complet
    st.subheader("Vue Complète")
    
    display_df = df[['Mois', 'Dépenses_Réelles', 'Budget', 'Variance', 'Z_Score', 'Anomalie']].copy()
    
    def color_anomaly(val):
        color = 'background-color: rgba(239, 68, 68, 0.3)' if val else ''
        return color
    
    styled_df = display_df.style.applymap(color_anomaly, subset=['Anomalie'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

# TAB 4: Architecture
with tab4:
    st.header("Architecture Technique du Système")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🗄️ Couche Données
        
        **Stockage:**
        - AWS S3 / Azure Blob Storage
        - PostgreSQL / MongoDB
        - Data Lake (Delta Lake)
        
        **ETL:**
        - Apache Airflow
        - Apache Spark
        - dbt (Data Build Tool)
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Couche ML
        
        **Frameworks:**
        - Python + Scikit-learn
        - TensorFlow / PyTorch
        - XGBoost
        
        **MLOps:**
        - MLflow (tracking)
        - Kubeflow (pipelines)
        - FastAPI (API REST)
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Couche Présentation
        
        **Visualisation:**
        - Streamlit Dashboard
        - Tableau / Power BI
        - Plotly / Matplotlib
        
        **Communication:**
        - WebSockets (temps réel)
        - Alertes email/Slack
        - API GraphQL
        """)
    
    st.markdown("---")
    
    # Modèles implémentés
    st.subheader("🧠 Modèles ML Implémentés")
    
    model_col1, model_col2 = st.columns(2)
    
    with model_col1:
        st.info("""
        **Régression Linéaire**
        - Prédiction des tendances
        - Intervalle de confiance 95%
        - Rapide et interprétable
        """)
        
        st.success("""
        **Random Forest**
        - Ensemble de 100 arbres
        - Capture non-linéarités
        - Feature importance
        """)
    
    with model_col2:
        st.warning("""
        **Détection Anomalies (Z-Score)**
        - Seuil paramétrable
        - Temps réel
        - Statistiquement robuste
        """)
        
        st.error("""
        **ARIMA (conceptuel)**
        - Séries temporelles
        - Saisonnalité
        - Auto-régression
        """)
    
    st.markdown("---")
    
    # Workflow
    st.subheader("🔄 Flux de Travail")
    
    workflow = """
    ```
    1. COLLECTE       →  2. NETTOYAGE    →  3. ENTRAÎNEMENT
           ↓                   ↓                    ↓
    (Sources multiples)  (ETL Pipeline)      (ML Models)
           ↓                   ↓                    ↓
    4. PRÉDICTION     →  5. VISUALISATION →  6. ALERTE
           ↓                   ↓                    ↓
    (API REST)         (Dashboard Streamlit)  (Email/Slack)
    ```
    """
    st.code(workflow, language='text')

# TAB 5: Données
with tab5:
    st.header("📥 Données Brutes et Export")
    
    # Afficher le DataFrame complet
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Statistiques descriptives
    st.subheader("📊 Statistiques Descriptives")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dépenses Réelles:**")
        st.write(df['Dépenses_Réelles'].describe())
    
    with col2:
        st.write("**Variances:**")
        st.write(df['Variance'].describe())
    
    # Export CSV
    st.subheader("💾 Export des Données")
    
    csv = df.to_csv(index=False).encode('utf-8')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name="donnees_financieres.csv",
            mime="text/csv"
        )
    
    with col2:
        st.download_button(
            label="📥 Télécharger Anomalies",
            data=df[df['Anomalie']].to_csv(index=False).encode('utf-8'),
            file_name="anomalies.csv",
            mime="text/csv"
        )
    
    with col3:
        # Résumé JSON
        summary = {
            'total_depenses': int(total_spend),
            'total_budget': int(total_budget),
            'ecart_moyen': int(avg_variance),
            'nb_anomalies': int(anomaly_count)
        }
        import json
        st.download_button(
            label="📥 Télécharger Résumé JSON",
            data=json.dumps(summary, indent=2),
            file_name="resume.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.6);'>
    <p>🧠 Tableau de Bord IA Contrôle de Gestion | Développé avec Streamlit & Python</p>
    <p>📊 Machine Learning | 🔍 Détection d'Anomalies | 📈 Analyse Prédictive</p>
</div>
""", unsafe_allow_html=True)
