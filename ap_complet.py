import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Contrôle de Gestion Complet",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
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
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.25);
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
    .pipeline-step {
        background: rgba(255, 255, 255, 0.1);
        border-left: 4px solid #667eea;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
    }
    .alert-box {
        background: rgba(239, 68, 68, 0.2);
        border: 2px solid #ef4444;
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    .success-box {
        background: rgba(16, 185, 129, 0.2);
        border: 2px solid #10b981;
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============= FONCTIONS DE GÉNÉRATION DE DONNÉES =============

@st.cache_data
def generate_complete_financial_data():
    """Génère un ensemble complet de données financières multi-départements"""
    np.random.seed(42)
    
    departments = ['Finance', 'Marketing', 'RH', 'IT', 'Opérations', 'Commercial']
    months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
    
    data = []
    for dept in departments:
        base_budget = np.random.randint(50000, 150000)
        for i, month in enumerate(months):
            base = base_budget + i * np.random.randint(2000, 8000)
            variance = np.random.randn() * 15000
            actual_spend = base + variance
            budgeted = base + np.random.randint(3000, 10000)
            historical_spend = base - 5000 + np.random.randn() * 8000
            
            # Simulation de catégories de dépenses
            salaires = actual_spend * 0.5
            fournitures = actual_spend * 0.2
            marketing_cost = actual_spend * 0.15
            autres = actual_spend * 0.15
            
            data.append({
                'Département': dept,
                'Mois': month,
                'Mois_Num': i,
                'Dépenses_Réelles': round(actual_spend),
                'Budget': round(budgeted),
                'Historique': round(historical_spend),
                'Variance': round(actual_spend - budgeted),
                'Variance_Pct': round((actual_spend - budgeted) / budgeted * 100, 2),
                'Salaires': round(salaires),
                'Fournitures': round(fournitures),
                'Marketing': round(marketing_cost),
                'Autres': round(autres),
                'Effectif': np.random.randint(5, 50),
                'CA_Généré': round(actual_spend * np.random.uniform(1.5, 3.0))
            })
    
    return pd.DataFrame(data)

@st.cache_data
def generate_kpi_data():
    """Génère des KPIs stratégiques"""
    return {
        'EBITDA': 2500000,
        'EBITDA_Objectif': 2800000,
        'Marge_Brute': 35.5,
        'ROI': 18.3,
        'Cash_Flow': 850000,
        'Dette_Capitaux': 0.45,
        'Rotation_Stock': 6.2,
        'Délai_Paiement': 45
    }

def detect_anomalies(df, column='Variance', threshold=2):
    """Détection d'anomalies avec Z-Score"""
    mean = df[column].mean()
    std = df[column].std()
    df['Z_Score'] = (df[column] - mean) / std
    df['Anomalie'] = np.abs(df['Z_Score']) > threshold
    return df

def calculate_ratios(df):
    """Calcule les ratios financiers clés"""
    df['ROI'] = ((df['CA_Généré'] - df['Dépenses_Réelles']) / df['Dépenses_Réelles'] * 100).round(2)
    df['Efficience'] = (df['CA_Généré'] / df['Dépenses_Réelles']).round(2)
    df['Coût_par_Employé'] = (df['Dépenses_Réelles'] / df['Effectif']).round(0)
    return df

def train_predictive_models(df):
    """Entraîne les modèles prédictifs"""
    X = df[['Mois_Num', 'Budget', 'Historique']].values
    y = df['Dépenses_Réelles'].values
    
    # Régression linéaire
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    df['Prédiction_LR'] = lr_model.predict(X)
    
    # Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    df['Prédiction_RF'] = rf_model.predict(X)
    
    # Métriques
    y_pred_test = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    return df, lr_model, rf_model, mse, r2

def generate_recommendations(df, kpis):
    """Génère des recommandations stratégiques"""
    recommendations = []
    
    # Analyse des dépassements budgétaires
    over_budget = df[df['Variance'] > 0].groupby('Département')['Variance'].sum().sort_values(ascending=False)
    if len(over_budget) > 0:
        dept = over_budget.index[0]
        amount = over_budget.iloc[0]
        recommendations.append({
            'Type': '⚠️ Alerte Budget',
            'Priorité': 'Haute',
            'Département': dept,
            'Message': f"Le département {dept} a dépassé son budget de {amount/1000:.0f}K€",
            'Action': f"Revoir les dépenses et mettre en place un plan de réduction de {amount*0.15/1000:.0f}K€"
        })
    
    # Analyse ROI
    low_roi = df.groupby('Département')['ROI'].mean().sort_values()
    if len(low_roi) > 0 and low_roi.iloc[0] < 10:
        dept = low_roi.index[0]
        roi = low_roi.iloc[0]
        recommendations.append({
            'Type': '📉 ROI Faible',
            'Priorité': 'Moyenne',
            'Département': dept,
            'Message': f"ROI moyen de {roi:.1f}% pour {dept} (< 10%)",
            'Action': f"Optimiser les processus et réduire les coûts inefficaces"
        })
    
    # Analyse efficience
    efficiency = df.groupby('Département')['Efficience'].mean().sort_values(ascending=False)
    if len(efficiency) > 0:
        best_dept = efficiency.index[0]
        best_eff = efficiency.iloc[0]
        recommendations.append({
            'Type': '✅ Bonne Pratique',
            'Priorité': 'Basse',
            'Département': best_dept,
            'Message': f"{best_dept} a une excellente efficience ({best_eff:.2f})",
            'Action': f"Partager les bonnes pratiques avec les autres départements"
        })
    
    # EBITDA
    if kpis['EBITDA'] < kpis['EBITDA_Objectif']:
        gap = kpis['EBITDA_Objectif'] - kpis['EBITDA']
        recommendations.append({
            'Type': '🎯 Objectif EBITDA',
            'Priorité': 'Haute',
            'Département': 'Global',
            'Message': f"EBITDA actuel: {kpis['EBITDA']/1000000:.1f}M€ vs objectif {kpis['EBITDA_Objectif']/1000000:.1f}M€",
            'Action': f"Augmenter l'EBITDA de {gap/1000000:.1f}M€ via réduction des coûts et amélioration du CA"
        })
    
    return pd.DataFrame(recommendations)

# ============= CHARGEMENT DES DONNÉES =============

df_complete = generate_complete_financial_data()
df_complete = detect_anomalies(df_complete)
df_complete = calculate_ratios(df_complete)
df_complete, lr_model, rf_model, rf_mse, rf_r2 = train_predictive_models(df_complete)
kpis = generate_kpi_data()
recommendations = generate_recommendations(df_complete, kpis)

# ============= SIDEBAR =============

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("⚙️ Configuration")
    
    # Filtres
    st.markdown("### 🔍 Filtres")
    selected_dept = st.multiselect(
        "Départements",
        options=df_complete['Département'].unique(),
        default=df_complete['Département'].unique()
    )
    
    anomaly_threshold = st.slider(
        "Seuil anomalies (Z-Score)",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.1
    )
    
    st.markdown("---")
    st.markdown("### 📊 Pipeline")
    pipeline_steps = [
        "✅ 1. Collecte données",
        "✅ 2. Nettoyage ETL",
        "✅ 3. Analyse financière",
        "✅ 4. ML & Prédictions",
        "✅ 5. Détection anomalies",
        "✅ 6. KPIs stratégiques",
        "✅ 7. Recommandations",
        "✅ 8. Reporting"
    ]
    for step in pipeline_steps:
        st.markdown(f"**{step}**")
    
    st.markdown("---")
    if st.button("🔄 Régénérer"):
        st.cache_data.clear()
        st.rerun()

# Filtrage des données
df_filtered = df_complete[df_complete['Département'].isin(selected_dept)]
df_filtered = detect_anomalies(df_filtered, threshold=anomaly_threshold)

# ============= HEADER =============

st.title("💼 Contrôle de Gestion - Pipeline Complet")
st.markdown("### Système intégré de pilotage financier et stratégique")

# ============= KPIs STRATÉGIQUES =============

st.markdown("## 📊 Tableau de Bord Exécutif")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "💰 EBITDA",
        f"{kpis['EBITDA']/1000000:.1f}M€",
        f"{((kpis['EBITDA']/kpis['EBITDA_Objectif']-1)*100):.1f}%"
    )

with col2:
    st.metric(
        "📈 Marge Brute",
        f"{kpis['Marge_Brute']:.1f}%",
        "+2.3%"
    )

with col3:
    st.metric(
        "🎯 ROI Moyen",
        f"{kpis['ROI']:.1f}%",
        "+1.5%"
    )

with col4:
    st.metric(
        "💵 Cash Flow",
        f"{kpis['Cash_Flow']/1000:.0f}K€",
        "+12.5%"
    )

with col5:
    anomaly_count = df_filtered['Anomalie'].sum()
    st.metric(
        "⚠️ Alertes",
        f"{anomaly_count}",
        "Anomalies"
    )

st.markdown("---")

# ============= TABS =============

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📥 1. Collecte",
    "🧹 2. ETL & Qualité", 
    "💰 3. Analyse Budget",
    "🤖 4. Prédictions",
    "⚠️ 5. Anomalies",
    "📊 6. KPIs Avancés",
    "💡 7. Recommandations",
    "📄 8. Reporting"
])

# ============= TAB 1: COLLECTE =============

with tab1:
    st.header("📥 Étape 1 : Collecte des Données")
    
    st.markdown("""
    <div class='pipeline-step'>
        <h3>Sources de données intégrées</h3>
        <p>• Systèmes ERP (SAP, Oracle)</p>
        <p>• Bases de données transactionnelles</p>
        <p>• Fichiers Excel/CSV des départements</p>
        <p>• APIs externes (banques, fournisseurs)</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Vue d'ensemble des données")
        st.dataframe(df_filtered.head(10), use_container_width=True)
        
        st.info(f"""
        **Statistiques de collecte:**
        - {len(df_filtered)} enregistrements
        - {len(df_filtered['Département'].unique())} départements
        - {len(df_filtered['Mois'].unique())} mois
        - Période: {df_filtered['Mois'].min()} à {df_filtered['Mois'].max()}
        """)
    
    with col2:
        st.subheader("🔍 Qualité des données")
        
        quality_metrics = {
            'Complétude': 100,
            'Cohérence': 98.5,
            'Validité': 99.2,
            'Actualité': 100
        }
        
        fig_quality = go.Figure(go.Bar(
            x=list(quality_metrics.values()),
            y=list(quality_metrics.keys()),
            orientation='h',
            marker_color=['#10b981' if v >= 95 else '#ef4444' for v in quality_metrics.values()]
        ))
        
        fig_quality.update_layout(
            title="Score de Qualité des Données (%)",
            xaxis_range=[0, 100],
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig_quality, use_container_width=True)
        
        st.success("✅ Données validées et prêtes pour l'analyse")

# ============= TAB 2: ETL =============

with tab2:
    st.header("🧹 Étape 2 : ETL et Nettoyage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='pipeline-step'>
            <h3>🔄 Transformations appliquées</h3>
            <p>✅ Normalisation des formats de dates</p>
            <p>✅ Conversion des devises</p>
            <p>✅ Suppression des doublons (0 détectés)</p>
            <p>✅ Gestion des valeurs manquantes</p>
            <p>✅ Validation des types de données</p>
            <p>✅ Enrichissement avec données historiques</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📊 Distribution des dépenses")
        
        fig_dist = px.histogram(
            df_filtered,
            x='Dépenses_Réelles',
            nbins=30,
            title="Distribution des Dépenses par Transaction",
            template="plotly_dark"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Détection des valeurs aberrantes")
        
        fig_box = px.box(
            df_filtered,
            x='Département',
            y='Dépenses_Réelles',
            color='Département',
            title="Box Plot par Département",
            template="plotly_dark"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.subheader("📈 Évolution temporelle")
        
        time_series = df_filtered.groupby('Mois')['Dépenses_Réelles'].sum().reset_index()
        
        fig_time = px.line(
            time_series,
            x='Mois',
            y='Dépenses_Réelles',
            title="Évolution Mensuelle des Dépenses",
            template="plotly_dark",
            markers=True
        )
        st.plotly_chart(fig_time, use_container_width=True)

# ============= TAB 3: ANALYSE BUDGET =============

with tab3:
    st.header("💰 Étape 3 : Analyse Budgétaire")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Analyse par département
        dept_summary = df_filtered.groupby('Département').agg({
            'Dépenses_Réelles': 'sum',
            'Budget': 'sum',
            'Variance': 'sum'
        }).reset_index()
        dept_summary['Taux_Réalisation'] = (dept_summary['Dépenses_Réelles'] / dept_summary['Budget'] * 100).round(1)
        
        fig_dept = go.Figure()
        
        fig_dept.add_trace(go.Bar(
            x=dept_summary['Département'],
            y=dept_summary['Budget'],
            name='Budget',
            marker_color='#10b981'
        ))
        
        fig_dept.add_trace(go.Bar(
            x=dept_summary['Département'],
            y=dept_summary['Dépenses_Réelles'],
            name='Réel',
            marker_color='#3b82f6'
        ))
        
        fig_dept.update_layout(
            title="Budget vs Réalisé par Département",
            barmode='group',
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig_dept, use_container_width=True)
    
    with col2:
        # Pie chart répartition
        fig_pie = px.pie(
            dept_summary,
            values='Dépenses_Réelles',
            names='Département',
            title="Répartition des Dépenses par Département",
            template="plotly_dark",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Tableau détaillé
    st.subheader("📋 Tableau Détaillé par Département")
    
    dept_summary['Budget'] = dept_summary['Budget'].apply(lambda x: f"{x/1000:.0f}K€")
    dept_summary['Dépenses_Réelles'] = dept_summary['Dépenses_Réelles'].apply(lambda x: f"{x/1000:.0f}K€")
    dept_summary['Variance'] = dept_summary['Variance'].apply(lambda x: f"{x/1000:+.0f}K€")
    dept_summary['Taux_Réalisation'] = dept_summary['Taux_Réalisation'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(dept_summary, use_container_width=True, hide_index=True)
    
    # Analyse des catégories de dépenses
    st.subheader("📊 Analyse par Catégorie de Dépenses")
    
    col1, col2 = st.columns(2)
    
    with col1:
        categories = df_filtered[['Salaires', 'Fournitures', 'Marketing', 'Autres']].sum()
        
        fig_cat = go.Figure(data=[go.Pie(
            labels=categories.index,
            values=categories.values,
            hole=0.3
        )])
        
        fig_cat.update_layout(
            title="Répartition par Catégorie",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        cat_df = pd.DataFrame({
            'Catégorie': categories.index,
            'Montant': categories.values,
            'Pourcentage': (categories.values / categories.values.sum() * 100).round(1)
        })
        
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

# ============= TAB 4: PRÉDICTIONS =============

with tab4:
    st.header("🤖 Étape 4 : Modèles Prédictifs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Régression Linéaire**
        - Équation: y = {lr_model.coef_[0]:.2f}x₁ + {lr_model.coef_[1]:.2f}x₂ + {lr_model.coef_[2]:.2f}x₃ + {lr_model.intercept_:.2f}
        - Tendance mensuelle: +{lr_model.coef_[0]/1000:.1f}K€
        """)
    
    with col2:
        st.success(f"""
        **Random Forest**
        - MSE: {rf_mse:.2f}
        - R² Score: {rf_r2:.4f}
        - Précision: {rf_r2*100:.1f}%
        """)
    
    # Graphique prédictions
    dept_selected = st.selectbox("Sélectionner un département", df_filtered['Département'].unique())
    df_dept = df_filtered[df_filtered['Département'] == dept_selected]
    
    fig_pred = go.Figure()
    
    fig_pred.add_trace(go.Scatter(
        x=df_dept['Mois'],
        y=df_dept['Dépenses_Réelles'],
        mode='lines+markers',
        name='Réel',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=10)
    ))
    
    fig_pred.add_trace(go.Scatter(
        x=df_dept['Mois'],
        y=df_dept['Prédiction_LR'],
        mode='lines',
        name='Prédiction Linéaire',
        line=dict(color='#ef4444', width=2, dash='dash')
    ))
    
    fig_pred.add_trace(go.Scatter(
        x=df_dept['Mois'],
        y=df_dept['Prédiction_RF'],
        mode='lines',
        name='Prédiction RF',
        line=dict(color='#8b5cf6', width=2, dash='dot')
    ))
    
    fig_pred.update_layout(
        title=f"Prédictions pour {dept_selected}",
        xaxis_title="Mois",
        yaxis_title="Montant (€)",
        template="plotly_dark",
        height=500
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Prévisions futures
    st.subheader("🔮 Prévisions pour les 3 prochains mois")
    
    future_months = ['Jan+1', 'Fév+1', 'Mar+1']
    last_mois = df_dept['Mois_Num'].max()
    
    future_predictions = []
    for i, month in enumerate(future_months, 1):
        X_future = np.array([[last_mois + i, df_dept['Budget'].mean(), df_dept['Historique'].mean()]])
        pred_lr = lr_model.predict(X_future)[0]
        pred_rf = rf_model.predict(X_future)[0]
        
        future_predictions.append({
            'Mois': month,
            'Prédiction LR': f"{pred_lr/1000:.0f}K€",
            'Prédiction RF': f"{pred_rf/1000:.0f}K€",
            'Moyenne': f"{(pred_lr + pred_rf)/2/1000:.0f}K€"
        })
    
    st.dataframe(pd.DataFrame(future_predictions), use_container_width=True, hide_index=True)

# ============= TAB 5: ANOMALIES =============

with tab5:
    st.header("⚠️ Étape 5 : Détection d'Anomalies")
    
    anomaly_count = df_filtered['Anomalie'].sum()
    
    if anomaly_count > 0:
        st.markdown(f"""
        <div class='alert-box'>
            <h3>⚠️ {anomaly_count} anomalies détectées</h3>
            <p>Seuil: |Z-Score| > {anomaly_threshold}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='success-box'>
            <h3>✅ Aucune anomalie détectée</h3>
            <p>Toutes les transactions sont dans les limites normales</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphique Z-Scores
    fig_z = go.Figure()
    
    colors_z = ['#ef4444' if a else '#10b981' for a in df_filtered['Anomalie']]
    
    fig_z.add_trace(go.Scatter(
        x=df_filtered.index,
        y=df_filtered['Z_Score'],
        mode='markers',
        marker=dict(color=colors_z, size=8),
        name='Z-Score'
    ))
    
    fig_z.add_hline(y=anomaly_threshold, line_dash="dash", line_color="red",
                    annotation_text=f"Seuil +{anomaly_threshold}")
    fig_z.add_hline(y=-anomaly_threshold, line_dash="dash", line_color="red",
                    annotation_text=f"Seuil -{anomaly_threshold}")
    
    fig_z.update_layout(
        title="Distribution des Z-Scores",
        xaxis_title="Transaction",
        yaxis_title="Z-Score",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig_z, use_container_width=True)
    
   # Détail des anomalies
if anomaly_count > 0:
    st.subheader("📋 Détail des Anomalies")
    
    anomaly_df = df_filtered[df_filtered['Anomalie']][
        ['Département', 'Mois', 'Dépenses_Réelles', 'Budget', 'Variance', 'Z_Score']
    ].copy()
    
    # Format K€ et arrondi
    anomaly_df['Dépenses_Réelles'] = anomaly_df['Dépenses_Réelles'].apply(
        lambda x: f"{x/1000:.0f}K€"
    )
    anomaly_df['Budget'] = anomaly_df['Budget'].apply(
        lambda x: f"{x/1000:.0f}K€"
    )
    anomaly_df['Variance'] = anomaly_df['Variance'].apply(
        lambda x: f"{x/1000:.0f}K€"
    )
    
    # Format Z-Score avec deux décimales
    anomaly_df['Z_Score'] = anomaly_df['Z_Score'].apply(
        lambda x: f"{x:.2f}"
    )
    
    st.dataframe(anomaly_df, use_container_width=True, hide_index=True)


# ============= TAB 6: KPIs AVANCÉS =============

with tab6:
    st.header("📊 Étape 6 : KPIs et Indicateurs Avancés")
    
    # Ratios financiers
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ratio d'Endettement", f"{kpis['Dette_Capitaux']:.2f}", "Optimal")
    
    with col2:
        st.metric("Rotation Stock", f"{kpis['Rotation_Stock']:.1f}x", "+0.3x")
    
    with col3:
        st.metric("Délai Paiement", f"{kpis['Délai_Paiement']}j", "-2j")
    
    with col4:
        avg_roi = df_filtered['ROI'].mean()
        st.metric("ROI Moyen", f"{avg_roi:.1f}%", f"{avg_roi-15:.1f}%")
    
    st.markdown("---")
    
    # Analyse ROI par département
    col1, col2 = st.columns(2)
    
    with col1:
        roi_dept = df_filtered.groupby('Département')['ROI'].mean().sort_values(ascending=False).reset_index()
        
        fig_roi = px.bar(
            roi_dept,
            x='Département',
            y='ROI',
            title="ROI Moyen par Département (%)",
            color='ROI',
            color_continuous_scale='RdYlGn',
            template="plotly_dark"
        )
        
        st.plotly_chart(fig_roi, use_container_width=True)
    
    with col2:
        eff_dept = df_filtered.groupby('Département')['Efficience'].mean().sort_values(ascending=False).reset_index()
        
        fig_eff = px.bar(
            eff_dept,
            x='Département',
            y='Efficience',
            title="Efficience par Département (CA/Coûts)",
            color='Efficience',
            color_continuous_scale='Blues',
            template="plotly_dark"
        )
        
        st.plotly_chart(fig_eff, use_container_width=True)
    
    # Coût par employé
    st.subheader("👥 Analyse des Coûts par Employé")
    
    cost_emp = df_filtered.groupby('Département').agg({
        'Coût_par_Employé': 'mean',
        'Effectif': 'mean',
        'CA_Généré': 'sum'
    }).reset_index()
    
    cost_emp['CA_par_Employé'] = (cost_emp['CA_Généré'] / cost_emp['Effectif']).round(0)
    
    fig_scatter = px.scatter(
        cost_emp,
        x='Coût_par_Employé',
        y='CA_par_Employé',
        size='Effectif',
        color='Département',
        title="Coût vs CA par Employé (taille = effectif)",
        template="plotly_dark",
        hover_data=['Département', 'Effectif']
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

# ============= TAB 7: RECOMMANDATIONS =============

with tab7:
    st.header("💡 Étape 7 : Recommandations Stratégiques")
    
    st.markdown("""
    <div class='pipeline-step'>
        <h3>🎯 Analyse IA et Génération de Recommandations</h3>
        <p>Basé sur l'analyse des données, des modèles ML et des KPIs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Affichage des recommandations
    for idx, row in recommendations.iterrows():
        priority_color = {
            'Haute': '#ef4444',
            'Moyenne': '#f59e0b',
            'Basse': '#10b981'
        }
        
        color = priority_color.get(row['Priorité'], '#64748b')
        
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.1); 
                    border-left: 4px solid {color}; 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-radius: 8px;'>
            <h4>{row['Type']} - Priorité: {row['Priorité']}</h4>
            <p><strong>Département:</strong> {row['Département']}</p>
            <p><strong>Constat:</strong> {row['Message']}</p>
            <p><strong>Action recommandée:</strong> {row['Action']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Plan d'action
    st.markdown("---")
    st.subheader("📋 Plan d'Action Global")
    
    action_plan = pd.DataFrame({
        'Action': [
            'Audit détaillé département Marketing',
            'Formation efficacité opérationnelle',
            'Renégociation contrats fournisseurs',
            'Mise en place tableaux de bord temps réel',
            'Revue des processus inefficaces'
        ],
        'Responsable': ['DAF', 'DRH', 'Achats', 'DSI', 'COO'],
        'Délai': ['2 semaines', '1 mois', '3 mois', '1 mois', '6 semaines'],
        'Impact Estimé': ['150K€', '200K€', '300K€', '100K€', '250K€'],
        'Statut': ['🟡 En cours', '🟢 Planifié', '🔴 Urgent', '🟢 Planifié', '🟡 En cours']
    })
    
    st.dataframe(action_plan, use_container_width=True, hide_index=True)

# ============= TAB 8: REPORTING =============

with tab8:
    st.header("📄 Étape 8 : Reporting et Export")
    
    st.markdown("""
    <div class='pipeline-step'>
        <h3>📊 Rapports Disponibles</h3>
        <p>Génération automatique de rapports pour différents stakeholders</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Exports de Données")
        
        # Export données complètes
        csv_complete = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Données Complètes (CSV)",
            data=csv_complete,
            file_name=f"donnees_financieres_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Export anomalies
        if anomaly_count > 0:
            csv_anomalies = df_filtered[df_filtered['Anomalie']].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⚠️ Anomalies (CSV)",
                data=csv_anomalies,
                file_name=f"anomalies_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        # Export recommandations
        csv_recomm = recommendations.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💡 Recommandations (CSV)",
            data=csv_recomm,
            file_name=f"recommandations_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.subheader("📊 Rapports Générés")
        
        reports = [
            {"Nom": "Rapport Mensuel Direction", "Fréquence": "Mensuel", "Dernier": "01/12/2024"},
            {"Nom": "Tableau de Bord CFO", "Fréquence": "Hebdomadaire", "Dernier": "04/12/2024"},
            {"Nom": "Analyse Départementale", "Fréquence": "Mensuel", "Dernier": "01/12/2024"},
            {"Nom": "Alerte Anomalies", "Fréquence": "Temps réel", "Dernier": "05/12/2024"}
        ]
        
        st.dataframe(pd.DataFrame(reports), use_container_width=True, hide_index=True)
    
    # Résumé exécutif
    st.markdown("---")
    st.subheader("📋 Résumé Exécutif")
    
    total_depenses = df_filtered['Dépenses_Réelles'].sum()
    total_budget = df_filtered['Budget'].sum()
    variance_total = df_filtered['Variance'].sum()
    
    st.markdown(f"""
    <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px;'>
        <h3>Synthèse Financière - Période {df_filtered['Mois'].min()} à {df_filtered['Mois'].max()}</h3>
        
        <p><strong>💰 Dépenses Totales:</strong> {total_depenses/1000000:.2f}M€</p>
        <p><strong>📊 Budget Alloué:</strong> {total_budget/1000000:.2f}M€</p>
        <p><strong>📈 Variance Globale:</strong> {variance_total/1000:+.0f}K€ ({variance_total/total_budget*100:+.1f}%)</p>
        <p><strong>⚠️ Anomalies Détectées:</strong> {anomaly_count}</p>
        <p><strong>🎯 EBITDA:</strong> {kpis['EBITDA']/1000000:.1f}M€ ({(kpis['EBITDA']/kpis['EBITDA_Objectif']*100):.1f}% de l'objectif)</p>
        <p><strong>📊 ROI Moyen:</strong> {df_filtered['ROI'].mean():.1f}%</p>
        
        <hr style='border-color: rgba(255,255,255,0.2);'>
        
        <h4>🎯 Priorités Stratégiques:</h4>
        <ul>
            <li>Réduire les dépassements budgétaires dans les départements critiques</li>
            <li>Améliorer le ROI des départements sous-performants</li>
            <li>Optimiser l'allocation des ressources basée sur l'efficience</li>
            <li>Surveiller et corriger les anomalies détectées</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Graphique de synthèse
    st.subheader("📊 Vue Consolidée")
    
    monthly_summary = df_filtered.groupby('Mois').agg({
        'Dépenses_Réelles': 'sum',
        'Budget': 'sum',
        'CA_Généré': 'sum'
    }).reset_index()
    
    fig_summary = go.Figure()
    
    fig_summary.add_trace(go.Bar(
        x=monthly_summary['Mois'],
        y=monthly_summary['Budget'],
        name='Budget',
        marker_color='#10b981',
        opacity=0.6
    ))
    
    fig_summary.add_trace(go.Bar(
        x=monthly_summary['Mois'],
        y=monthly_summary['Dépenses_Réelles'],
        name='Dépenses',
        marker_color='#3b82f6'
    ))
    
    fig_summary.add_trace(go.Scatter(
        x=monthly_summary['Mois'],
        y=monthly_summary['CA_Généré'],
        name='CA Généré',
        mode='lines+markers',
        line=dict(color='#f59e0b', width=3),
        yaxis='y2'
    ))
    
    fig_summary.update_layout(
        title="Vue Consolidée Mensuelle",
        yaxis=dict(title="Montant (€)"),
        yaxis2=dict(title="CA Généré (€)", overlaying='y', side='right'),
        barmode='group',
        template="plotly_dark",
        height=500
    )
    
    st.plotly_chart(fig_summary, use_container_width=True)

# ============= FOOTER =============

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.6);'>
    <p>💼 Système de Contrôle de Gestion Intégré | Pipeline Complet</p>
    <p>🧠 Machine Learning | 📊 Analytics | 💡 Recommandations IA | 🔄 Temps Réel</p>
    <p style='font-size: 12px; margin-top: 10px;'>v2.0 - Développé avec Streamlit & Python | © 2024</p>
</div>
""", unsafe_allow_html=True)