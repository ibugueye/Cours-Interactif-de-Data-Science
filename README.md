https://claude.ai/public/artifacts/39f1caca-b961-4a9c-a7d8-97bfbf8c574a

📚 Documentation Complète - Système de Contrôle de Gestion IA
Table des Matières
Vue d'ensemble
Installation et Configuration
Architecture Technique
Guide Utilisateur
Modules et Fonctionnalités
API et Fonctions
Modèles de Machine Learning
Guide de Personnalisation
Résolution de Problèmes
Feuille de Route

1. Vue d'ensemble
🎯 Objectif
Le Système de Contrôle de Gestion IA est une application web complète développée avec Streamlit qui automatise et optimise l'ensemble du pipeline de contrôle de gestion financier, de la collecte des données jusqu'à la génération de recommandations stratégiques.
✨ Caractéristiques Principales
Pipeline complet en 8 étapes couvrant tout le cycle de contrôle de gestion
Intelligence Artificielle pour les prédictions et recommandations
Détection automatique d'anomalies via analyse statistique
Visualisations interactives avec Plotly
Multi-départements avec filtrage dynamique
Exports de données en CSV pour analyse externe
Interface moderne avec design responsive
👥 Public Cible
Contrôleurs de Gestion : Pilotage quotidien des budgets
Directeurs Financiers (DAF/CFO) : Vision stratégique et KPIs
Directeurs de Département : Suivi de leurs budgets
Direction Générale : Rapports exécutifs et synthèses

2. Installation et Configuration
📋 Prérequis
Python : Version 3.8 ou supérieure
pip : Gestionnaire de paquets Python
Système d'exploitation : Windows, macOS, ou Linux
🔧 Installation Étape par Étape
1. Cloner ou créer le projet
# Créer un répertoire pour le projet
mkdir ai_controle_gestion
cd ai_controle_gestion

# Créer le fichier principal
touch app.py

2. Créer un environnement virtuel (recommandé)
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement
# Sur Windows :
venv\Scripts\activate
# Sur macOS/Linux :
source venv/bin/activate

3. Installer les dépendances
pip install streamlit pandas numpy plotly scikit-learn

Ou créer un fichier requirements.txt :
streamlit==1.29.0
pandas==2.1.4
numpy==1.26.2
plotly==5.18.0
scikit-learn==1.3.2

Puis installer :
pip install -r requirements.txt

4. Lancer l'application
streamlit run app.py

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse : http://localhost:8501
⚙️ Configuration Avancée
Modifier le port d'exécution
streamlit run app.py --server.port 8080

Configuration du fichier .streamlit/config.toml
Créez le dossier .streamlit et le fichier config.toml :
[theme]
primaryColor = "#667eea"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true


3. Architecture Technique
🏗️ Structure du Projet
ai_controle_gestion/
│
├── app.py                    # Application principale
├── requirements.txt          # Dépendances Python
├── README.md                # Documentation générale
├── .streamlit/              # Configuration Streamlit
│   └── config.toml
│
├── data/                    # Données (si connecté à source externe)
│   ├── raw/                # Données brutes
│   └── processed/          # Données traitées
│
├── models/                  # Modèles ML sauvegardés
│   ├── linear_regression.pkl
│   └── random_forest.pkl
│
├── exports/                 # Exports générés
│   ├── reports/
│   └── data/
│
└── docs/                    # Documentation
    ├── user_guide.md
    └── technical_doc.md

🔄 Flux de Données
┌─────────────────┐
│  Génération     │
│  Données        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validation &   │
│  Nettoyage      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Calculs &      │
│  Enrichissement │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Training &  │
│  Prédictions    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Détection      │
│  Anomalies      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Génération     │
│  Recommandations│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Visualisation  │
│  & Export       │
└─────────────────┘

💾 Modèle de Données
DataFrame Principal
Colonne
Type
Description
Département
string
Nom du département
Mois
string
Mois (Jan-Déc)
Mois_Num
int
Numéro du mois (0-11)
Dépenses_Réelles
float
Montant réel dépensé
Budget
float
Budget alloué
Historique
float
Dépenses année précédente
Variance
float
Écart Budget - Réel
Variance_Pct
float
Écart en pourcentage
Salaires
float
Coût salarial
Fournitures
float
Coût fournitures
Marketing
float
Coût marketing
Autres
float
Autres coûts
Effectif
int
Nombre d'employés
CA_Généré
float
Chiffre d'affaires généré
Z_Score
float
Score statistique
Anomalie
bool
Indicateur d'anomalie
ROI
float
Retour sur investissement
Efficience
float
Ratio CA/Coûts
Coût_par_Employé
float
Coût moyen par employé
Prédiction_LR
float
Prédiction régression linéaire
Prédiction_RF
float
Prédiction Random Forest


4. Guide Utilisateur
🚀 Démarrage Rapide
1. Lancement de l'Application
streamlit run app.py

2. Interface Principale
L'application s'ouvre sur le Tableau de Bord Exécutif avec 5 KPIs principaux :
💰 EBITDA : Résultat opérationnel
📈 Marge Brute : Rentabilité globale
🎯 ROI Moyen : Retour sur investissement
💵 Cash Flow : Trésorerie disponible
⚠️ Alertes : Nombre d'anomalies
3. Navigation par Onglets
L'application est organisée en 8 onglets correspondant aux étapes du pipeline.
📊 Utilisation de la Sidebar
Filtres Disponibles
1. Départements
Sélection multiple
Filtre toutes les visualisations
Par défaut : tous les départements sélectionnés
2. Seuil Anomalies
Slider de 1.0 à 3.0
Valeur par défaut : 2.0
Impact : modifie la sensibilité de détection
3. Pipeline Tracker
Affiche les 8 étapes validées
Indicateur visuel de progression
4. Régénérer
Bouton pour rafraîchir les données
Génère un nouveau jeu de données aléatoire
📋 Guide par Onglet
📥 Onglet 1 : Collecte
Objectif : Visualiser les données brutes collectées
Fonctionnalités :
Aperçu des 10 premières lignes
Statistiques de collecte (nombre d'enregistrements, départements, mois)
Score de qualité des données (4 métriques)
Graphique de qualité des données
Actions utilisateur :
Consulter la qualité des données
Vérifier la complétude des informations

🧹 Onglet 2 : ETL & Qualité
Objectif : Comprendre les transformations appliquées
Fonctionnalités :
Liste des transformations ETL
Histogramme de distribution des dépenses
Box plot par département (détection valeurs aberrantes)
Graphique d'évolution temporelle
Actions utilisateur :
Identifier les outliers visuellement
Comprendre la distribution des dépenses
Analyser les tendances temporelles

💰 Onglet 3 : Analyse Budget
Objectif : Comparer budget vs réalisé
Visualisations :
Graphique en barres groupées : Budget vs Réel par département
Graphique circulaire : Répartition des dépenses
Tableau détaillé : Taux de réalisation par département
Analyse par catégorie : Salaires, Fournitures, Marketing, Autres
Indicateurs clés :
Taux de réalisation budgétaire
Variance absolue et relative
Répartition par poste de dépense
Actions utilisateur :
Identifier les départements en dépassement
Analyser la structure des coûts
Comparer les performances départementales

🤖 Onglet 4 : Prédictions
Objectif : Visualiser les prédictions ML
Fonctionnalités :
Sélection d'un département spécifique
Comparaison des 2 modèles (LR et RF)
Métriques de performance (MSE, R²)
Scatter plots Réel vs Prédit
Prévisions sur 3 mois futurs
Interprétation :
Ligne bleue : Dépenses réelles
Ligne rouge pointillée : Régression linéaire
Ligne violette pointillée : Random Forest
Points proches de la diagonale verte : Bonnes prédictions
Actions utilisateur :
Comparer la précision des modèles
Anticiper les dépenses futures
Planifier les budgets des mois suivants

⚠️ Onglet 5 : Anomalies
Objectif : Détecter et analyser les anomalies
Méthode : Z-Score (écart à la moyenne en unités d'écart-type)
Visualisations :
Alerte globale (nombre d'anomalies)
Graphique scatter des Z-Scores
Lignes de seuil (configurable)
Tableau détaillé des anomalies
Interprétation :
Points rouges : Anomalies détectées
Points verts : Valeurs normales
|Z-Score| > seuil : Anomalie
Actions utilisateur :
Investiguer les anomalies détectées
Ajuster le seuil de sensibilité
Exporter la liste des anomalies

📊 Onglet 6 : KPIs Avancés
Objectif : Analyser les ratios financiers avancés
KPIs Disponibles :
Ratio d'Endettement : Dette / Capitaux propres
Rotation Stock : Nombre de renouvellements
Délai Paiement : DSO (Days Sales Outstanding)
ROI Moyen : Retour sur investissement global
Analyses :
ROI par département (graphique en barres coloré)
Efficience par département (CA/Coûts)
Scatter plot Coût vs CA par employé
Interprétation :
ROI > 15% : Performance excellente
Efficience > 2 : Département très rentable
CA/Employé élevé : Productivité forte
Actions utilisateur :
Identifier les départements les plus rentables
Benchmarker les performances
Optimiser l'allocation des ressources

💡 Onglet 7 : Recommandations
Objectif : Obtenir des recommandations stratégiques générées par IA
Types de recommandations :
⚠️ Alerte Budget (Priorité Haute)


Dépassements budgétaires significatifs
Actions correctives suggérées
📉 ROI Faible (Priorité Moyenne)


Départements sous-performants
Optimisations proposées
✅ Bonne Pratique (Priorité Basse)


Départements exemplaires
Partage de pratiques
🎯 Objectif EBITDA (Priorité Haute)


Écart vs objectif
Plan d'amélioration
Plan d'Action Global :
Liste des 5 actions prioritaires
Responsables désignés
Délais et impacts estimés
Statut de suivi
Actions utilisateur :
Prioriser les actions selon l'urgence
Affecter les responsabilités
Suivre l'avancement du plan

📄 Onglet 8 : Reporting
Objectif : Générer et exporter des rapports
Exports Disponibles :
Données Complètes (CSV) : Tous les enregistrements filtrés
Anomalies (CSV) : Uniquement les transactions anormales
Recommandations (CSV) : Liste des actions suggérées
Rapports Automatiques :
Rapport Mensuel Direction
Tableau de Bord CFO (hebdomadaire)
Analyse Départementale
Alertes Anomalies (temps réel)
Résumé Exécutif :
Synthèse financière période complète
KPIs consolidés
Priorités stratégiques
Graphique consolidé mensuel (Budget + Dépenses + CA)
Actions utilisateur :
Télécharger les données pour analyse externe
Partager les rapports avec la direction
Archiver les résultats mensuels

5. Modules et Fonctionnalités
📦 Fonctions Principales
generate_complete_financial_data()
Description : Génère un jeu de données financières complet pour tous les départements
Paramètres : Aucun (utilise @st.cache_data pour mise en cache)
Retour : pd.DataFrame avec 72 lignes (6 départements × 12 mois)
Champs générés :
Dépenses réelles (avec variance aléatoire)
Budget alloué
Historique année précédente
Répartition par catégorie (Salaires, Fournitures, etc.)
Effectifs et CA généré
Exemple d'utilisation :
df = generate_complete_financial_data()
print(df.head())


detect_anomalies(df, column='Variance', threshold=2)
Description : Détecte les anomalies via la méthode du Z-Score
Paramètres :
df : DataFrame à analyser
column : Colonne sur laquelle calculer le Z-Score (défaut: 'Variance')
threshold : Seuil de détection (défaut: 2.0)
Formule Z-Score :
Z = (X - μ) / σ

Où :
- X = valeur observée
- μ = moyenne
- σ = écart-type

Retour : DataFrame enrichi avec colonnes Z_Score et Anomalie
Exemple :
df = detect_anomalies(df, column='Variance', threshold=2.5)
anomalies = df[df['Anomalie'] == True]
print(f"Nombre d'anomalies : {len(anomalies)}")


calculate_ratios(df)
Description : Calcule les ratios financiers clés
Ratios calculés :
ROI : (CA_Généré - Dépenses_Réelles) / Dépenses_Réelles × 100
Efficience : CA_Généré / Dépenses_Réelles
Coût par Employé : Dépenses_Réelles / Effectif
Exemple :
df = calculate_ratios(df)
print(df[['Département', 'ROI', 'Efficience']].head())


train_predictive_models(df)
Description : Entraîne les modèles de Machine Learning
Modèles créés :
Régression Linéaire : Prédiction basique avec tendance linéaire
Random Forest : Modèle avancé (100 arbres)
Features utilisées :
Mois_Num : Position temporelle
Budget : Budget alloué
Historique : Dépenses année précédente
Target : Dépenses_Réelles
Retour :
DataFrame enrichi avec prédictions
Modèles entraînés (lr_model, rf_model)
Métriques (MSE, R²)
Exemple :
df, lr_model, rf_model, mse, r2 = train_predictive_models(df)
print(f"R² Score : {r2:.4f}")


generate_recommendations(df, kpis)
Description : Génère des recommandations stratégiques automatiques
Logique de génération :
Analyse des dépassements budgétaires
Identification des ROI faibles (< 10%)
Reconnaissance des bonnes pratiques (efficience élevée)
Vérification de l'objectif EBITDA
Retour : DataFrame avec colonnes :
Type : Type de recommandation
Priorité : Haute / Moyenne / Basse
Département : Département concerné
Message : Description du constat
Action : Action recommandée
Exemple :
recommendations = generate_recommendations(df, kpis)
for _, rec in recommendations.iterrows():
    print(f"{rec['Type']} : {rec['Message']}")


🎨 Composants Visuels
Métriques (KPIs)
st.metric(
    label="💰 EBITDA",
    value="2.5M€",
    delta="+12.5%"
)

Interprétation des couleurs :
Vert : Delta positif
Rouge : Delta négatif

Graphiques Plotly
1. Line Chart (Évolution temporelle)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['Mois'],
    y=df['Dépenses_Réelles'],
    mode='lines+markers',
    name='Réel'
))
st.plotly_chart(fig, use_container_width=True)

2. Bar Chart (Comparaisons)
fig = px.bar(
    df,
    x='Département',
    y='ROI',
    color='ROI',
    color_continuous_scale='RdYlGn'
)

3. Pie Chart (Répartitions)
fig = px.pie(
    df,
    values='Montant',
    names='Catégorie',
    hole=0.4  # Donut chart
)

4. Scatter Plot (Corrélations)
fig = px.scatter(
    df,
    x='Coût_par_Employé',
    y='CA_par_Employé',
    size='Effectif',
    color='Département'
)


6. API et Fonctions
🔌 Fonctions Utilitaires
Export de Données
# Export CSV
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Télécharger CSV",
    data=csv,
    file_name="donnees.csv",
    mime="text/csv"
)

Filtrage de Données
# Filtrage multi-sélection
selected_depts = st.multiselect(
    "Départements",
    options=df['Département'].unique(),
    default=df['Département'].unique()
)

df_filtered = df[df['Département'].isin(selected_depts)]

Agrégations
# Agrégation par département
summary = df.groupby('Département').agg({
    'Dépenses_Réelles': 'sum',
    'Budget': 'sum',
    'Variance': 'sum'
}).reset_index()


🔄 Gestion du Cache
Streamlit utilise @st.cache_data pour optimiser les performances :
@st.cache_data
def load_data():
    # Opération coûteuse
    return df

# Vider le cache
st.cache_data.clear()
st.rerun()


7. Modèles de Machine Learning
🤖 Régression Linéaire
Principe
Modèle prédictif simple basé sur une relation linéaire entre variables.
Équation :
y = β₀ + β₁x₁ + β₂x₂ + β₃x₃

Où :
- y = Dépenses prédites
- x₁ = Mois_Num
- x₂ = Budget
- x₃ = Historique
- β = Coefficients

Avantages
✅ Rapide à entraîner
✅ Facile à interpréter
✅ Fonctionne bien avec tendances linéaires
Limites
❌ Ne capture pas les non-linéarités
❌ Sensible aux outliers
Code d'entraînement
from sklearn.linear_model import LinearRegression

X = df[['Mois_Num', 'Budget', 'Historique']].values
y = df['Dépenses_Réelles'].values

model = LinearRegression()
model.fit(X, y)

# Coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")


🌳 Random Forest
Principe
Ensemble de 100 arbres de décision qui votent pour la prédiction finale.
Fonctionnement :
Création de 100 sous-échantillons du dataset
Entraînement d'un arbre sur chaque échantillon
Prédiction = moyenne des prédictions des 100 arbres
Avantages
✅ Capture les non-linéarités
✅ Robuste aux outliers
✅ Importance des features
✅ Peu de surapprentissage
Paramètres
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,      # Nombre d'arbres
    max_depth=None,        # Profondeur max (None = illimitée)
    min_samples_split=2,   # Min échantillons pour split
    random_state=42        # Reproductibilité
)

Feature Importance
importance = rf_model.feature_importances_
features = ['Mois_Num', 'Budget', 'Historique']

for feat, imp in zip(features, importance):
    print(f"{feat}: {imp:.3f}")


📊 Métriques d'Évaluation
1. Mean Squared Error (MSE)
Formule :
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²

Où :
- yᵢ = valeur réelle
- ŷᵢ = valeur prédite
- n = nombre d'observations

Interprétation :
Plus le MSE est faible, meilleure est la prédiction
Sensible aux grandes erreurs (au carré)

2. R² Score (Coefficient de Détermination)
Formule :
R² = 1 - (SS_res / SS_tot)

Où :
- SS_res = Σ(yᵢ - ŷᵢ)²  (somme carrés résidus)
- SS_tot = Σ(yᵢ - ȳ)²   (somme carrés totale)

Interprétation :
R² = 1 : Prédiction parfaite
R² = 0.8 : 80% de la variance expliquée (bon)
R² < 0.5 : Modèle peu performant
R² < 0 : Modèle pire qu'une moyenne simple

🔮 Prédictions Futures
Générer des Prédictions
# Prédiction pour les 3 prochains mois
future_months = ['Jan+1', 'Fév+1', 'Mar+1']
last_month = df['Mois_Num'].max()

predictions = []
for i in range(1, 4):
    X_future = np.array([[
        last_month + i,
        df['Budget'].mean(),
        df['Historique'].mean()
    ]])
    
    pred_lr = lr_model.predict(X_future)[0]
    pred_rf = rf_model.predict(X_future)[0]
    
    predictions.append({
        'Mois': future_months[i-1],
        'LR': pred_lr,
        'RF': pred_rf,
        'Moyenne': (pred_lr + pred_rf) / 2
    })


8. Guide de Personnalisation
🎨 Modifier le Design
Couleurs du Thème
Éditer .streamlit/config.toml :
[theme]
primaryColor = "#667eea"        # Bleu principal
backgroundColor = "#0e1117"      # Fond noir
secondaryBackgroundColor = "#262730"  # Fond cartes
textColor = "#fafafa"           # Texte blanc
font = "sans serif"             # Police

CSS Personnalisé
Dans app.py, modifier la section CSS :
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #votre_couleur1, #votre_couleur2);
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)


📊 Ajouter de Nouveaux Départements
departments = [
    'Finance', 
    'Marketing', 
    'RH', 
    'IT', 
    'Opérations', 
    'Commercial',
    'Logistique',      # Nouveau
    'R&D'              # Nouveau
]


💾 Connecter à une Base de Données Réelle
PostgreSQL
import psycopg2
import pandas as pd

@st.cache_data
def load_data_from_db():
    conn = psycopg2.connect(
        host="localhost",
        database="finance_db",
        user="user",
        password="password"
    )
    
    query = """
    SELECT 
        departement,
        mois,
        depenses_reelles,
        budget,
        effectif
    FROM finances
    WHERE annee = 2024
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    return df

MySQL
import mysql.connector
import pandas as pd

@st.cache_data
def load_data_from_mysql():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="finance_db"
    )
    
 

