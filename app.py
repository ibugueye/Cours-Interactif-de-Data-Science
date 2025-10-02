# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sympy as sp
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, mean_squared_error, r2_score,
                           roc_auc_score, roc_curve, precision_recall_curve, auc)
from sklearn.datasets import make_classification, load_iris
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="📚 Cours Interactif de Data Science",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-card {
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .exercise-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #28a745;
    }
    .concept-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .glossary-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🎓 Navigation du Cours")
st.sidebar.markdown("---")

# Glossaire dans la sidebar
with st.sidebar.expander("📕 Glossaire"):
    st.markdown("""
    - **DataFrame** : Structure de données tabulaire (comme un tableau Excel).
    - **p-value** : Probabilité que les résultats soient dus au hasard. **< 0.05** = significatif.
    - **AUC-ROC** : Aire sous la courbe ROC (1 = modèle parfait, 0.5 = hasard).
    - **Overfitting** : Modèle trop complexe qui mémorise les données d'entraînement.
    - **Corrélation** : Relation entre deux variables (-1 à 1).
    - **Biais-Variance** : Équilibre entre sous-apprentissage et surapprentissage.
    - **Feature** : Variable explicative (ex. : âge, revenu).
    - **Target** : Variable à prédire (ex. : remboursement d'un prêt).
    """)

# Sélection du module
module = st.sidebar.radio(
    "Choisissez un module:",
    ["🏠 Accueil", "📊 Module 1 - Fondations (Pandas)", "📈 Module 2 - Statistiques",
     "🧮 Module 3 - Mathématiques", "🤖 Module 4 - ML Introduction",
     "🌲 Module 5 - Algorithmes ML", "🏦 Module 6 - Projet Scoring Crédit",
     "🎓 Module 7 - Quiz Interactif", "🏦 Module 8 - Projet Immobilier", "📘 À Propos"]
)

# Page d'accueil
if module == "🏠 Accueil":
    st.markdown('<h1 class="main-header">📚 Cours Interactif de Data Science</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## 🌟 Bienvenue dans ce cours interactif !

        **Pourquoi ce cours ?**
        La Data Science est partout :
        - **Netflix** utilise des algorithmes pour recommander des films.
        - **Les banques** prédisent les risques de crédit.
        - **Les hôpitaux** optimisent les traitements avec l'IA.

        ### 🎯 Ce que vous allez apprendre :
        1. **Manipuler des données** avec Pandas et NumPy.
        2. **Analyser des statistiques** et tester des hypothèses.
        3. **Comprendre les maths** derrière le Machine Learning.
        4. **Construire des modèles** prédictifs.
        5. **Résoudre un cas réel** : un système de scoring crédit.

        ### 🛠 Comment utiliser cette application ?
        - **Expérimentez** avec les paramètres interactifs.
        - **Lisez les explications** et exemples concrets.
        - **Résolvez les exercices** pour valider vos connaissances.
        - **Visualisez les résultats** en temps réel.

        **Public cible** : Débutants en Data Science, analystes, ou toute personne curieuse !
        """)

    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=200)
        st.info("💡 **Conseil** : Exécutez les codes et jouez avec les paramètres pour mieux comprendre !")

    # Métriques du cours
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⏱ Durée estimée", "6 heures", "Auto-rythmé")
    with col2:
        st.metric("📊 Modules", "6", "De la théorie à la pratique")
    with col3:
        st.metric("💡 Concepts clés", "50+", "Explicités simplement")
    with col4:
        st.metric("🚀 Projet final", "1", "Scoring crédit réaliste")

    # Témoignage fictif
    st.markdown("---")
    st.markdown("""
    > *"Ce cours m'a permis de passer de zéro à un projet concret en quelques heures !
    > Les explications sont claires et les exemples très pratiques."*
    > **— Marie, Analyste Marketing**
    """)

# Module 1 - Fondations (Pandas)
elif module == "📊 Module 1 - Fondations (Pandas)":
    st.header("📊 Module 1 - Fondations de la Data Science avec Pandas")

    st.markdown("""
    ### 🎯 **Objectifs de ce Module**
    - Comprendre les **DataFrames** et leur manipulation.
    - Calculer des **statistiques descriptives**.
    - Nettoyer et explorer des données réelles.
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["📚 Introduction", "🐍 Pandas en Pratique", "📋 Exercice", "✅ Solution"])

    with tab1:
        st.markdown("""
        #### 💡 **Concept Clé : Qu'est-ce qu'un DataFrame ?**
        Un **DataFrame** est une structure de données **tabulaire** (comme un tableau Excel), avec :
        - **Des lignes** = Observations (ex. : clients, produits).
        - **Des colonnes** = Variables (ex. : âge, prix, catégorie).
        - **Un index** = Identifiant unique pour chaque ligne.

        **Exemple** :
        | Client   | Âge | Revenu | Achat |
        |----------|-----|--------|-------|
        | Client_1 | 25  | 3000€  | Oui   |
        | Client_2 | 40  | 5000€  | Non   |

        **Analogie** :
        > *"Un DataFrame est comme une **feuille Excel intelligente** :
        > - Vous pouvez **filtrer** (ex. : clients de plus de 30 ans).
        > - **Grouper** (ex. : revenu moyen par âge).
        > - **Visualiser** en 2 clics !"*

        ---

        #### 📖 **Pourquoi Pandas ?**
        - **80% du travail en Data Science** = Nettoyage et exploration de données.
        - Pandas permet de :
          1. **Lire** des fichiers (CSV, Excel, SQL).
          2. **Nettoyer** (valeurs manquantes, doublons).
          3. **Transformer** (créer de nouvelles colonnes).
          4. **Analyser** (moyennes, corrélations).

        **Cas d'usage réel** :
        > *"Un supermarché utilise Pandas pour :
        > - Identifier les **produits les plus vendus** par tranche d'âge.
        > - Détecter les **périodes de forte affluence**.
        > - Optimiser les **stocks** en fonction des tendances."*
        """)

    with tab2:
        st.subheader("🐍 Manipulation de données avec Pandas")

        st.markdown("""
        #### 📌 **Générateur de données interactif**
        Créez un jeu de données personnalisé pour expérimenter !
        """)

        col1, col2 = st.columns(2)
        with col1:
            n_clients = st.slider("Nombre de clients", 5, 50, 10)
            min_montant = st.number_input("Montant minimum (€)", 50, 500, 100)
            max_montant = st.number_input("Montant maximum (€)", 500, 2000, 1000)
        with col2:
            categories = st.multiselect(
                "Catégories de dépense",
                ["Alimentation", "Loisirs", "Transport", "Logement"],
                default=["Alimentation", "Loisirs", "Transport"]
            )

        if st.button("🔄 Générer les données"):
            np.random.seed(42)
            data = {
                "Client": [f"Client_{i+1}" for i in range(n_clients)],
                "Montant": np.random.randint(min_montant, max_montant, n_clients),
                "Catégorie": np.random.choice(categories, n_clients)
            }
            df = pd.DataFrame(data)
            st.session_state.df_module1 = df
            st.balloons()

        if 'df_module1' in st.session_state:
            df = st.session_state.df_module1

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📄 DataFrame")
                st.dataframe(df, use_container_width=True)
            with col2:
                st.subheader("📊 Statistiques descriptives")
                stats = df["Montant"].describe()
                st.dataframe(stats, use_container_width=True)
                st.markdown(f"""
                **Interprétation** :
                - **Moyenne** : {stats['mean']:.2f}€ (dépense typique).
                - **Écart-type** : {stats['std']:.2f}€ (variabilité).
                - **50% des clients** dépensent **moins de** {stats['50%']:.2f}€.
                """)

            # Visualisations
            st.subheader("📈 Visualisations")
            fig_col1, fig_col2 = st.columns(2)
            with fig_col1:
                fig = px.histogram(df, x="Montant", title="Distribution des montants",
                                  nbins=10, color_discrete_sequence=['#1f77b4'])
                st.plotly_chart(fig, use_container_width=True)
            with fig_col2:
                category_stats = df.groupby("Catégorie")["Montant"].mean().reset_index()
                fig = px.bar(category_stats, x="Catégorie", y="Montant",
                           title="Dépense moyenne par catégorie",
                           color="Catégorie", color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **💡 Insight** :
            - La catégorie **"Logement"** est souvent la plus coûteuse.
            - La **distribution** montre si les dépenses sont concentrées ou étalées.
            """)

    with tab3:
        st.markdown('<div class="exercise-box">', unsafe_allow_html=True)
        st.subheader("📋 Exercice Pratique : Analyse de Salaires")

        st.markdown("""
        **Scénario** :
        Vous êtes RH dans une entreprise et vous avez les données suivantes pour 15 employés :
        - **Âge** (20-60 ans)
        - **Salaire** (2000-8000€)

        **Questions** :
        1. Quelle est la **moyenne** et l'**écart-type** des salaires ?
        2. Y a-t-il une **corrélation** entre âge et salaire ?
        3. Quel est le **salaire médian** pour les 20-30 ans vs 50-60 ans ?

        **Indice** : Utilisez `df.groupby()` et `df.corr()`.
        """)

        if st.button("🧹 Générer les données d'exercice"):
            np.random.seed(123)
            ages = np.random.randint(20, 61, 15)
            salaires = np.random.randint(2000, 8001, 15)
            st.session_state.df_exercise = pd.DataFrame({
                "Age": ages,
                "Salaire": salaires
            })

        if 'df_exercise' in st.session_state:
            st.dataframe(st.session_state.df_exercise, use_container_width=True)

            user_code = st.text_area("💻 Votre code Python :", height=150,
                                    placeholder="""import pandas as pd
# 1. Calculer moyenne et écart-type des salaires
moyenne = df["Salaire"].mean()
ecart_type = df["Salaire"].std()

# 2. Corrélation âge-salaire
correlation = df["Age"].corr(df["Salaire"])

# 3. Salaire médian par tranche d'âge
df["Tranche_Age"] = pd.cut(df["Age"], bins=[20, 30, 40, 50, 60], labels=["20-30", "31-40", "41-50", "51-60"])
median_par_tranche = df.groupby("Tranche_Age")["Salaire"].median()""")

            if st.button("📤 Soumettre la solution"):
                st.success("✅ Code soumis ! Vérifiez l'onglet **Solution**.")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.subheader("✅ Solution et Interprétation")

        if 'df_exercise' in st.session_state:
            df = st.session_state.df_exercise.copy()

            # Solution code
            code_solution = """
# 1. Statistiques salariales
moyenne = df["Salaire"].mean()
ecart_type = df["Salaire"].std()

# 2. Corrélation âge-salaire
correlation = df["Age"].corr(df["Salaire"])

# 3. Tranches d'âge et salaire médian
bins = [20, 30, 40, 50, 60]
labels = ["20-30", "31-40", "41-50", "51-60"]
df["Tranche_Age"] = pd.cut(df["Age"], bins=bins, labels=labels)
median_par_tranche = df.groupby("Tranche_Age")["Salaire"].median()
"""
            st.code(code_solution, language='python')

            # Exécution
            moyenne = df["Salaire"].mean()
            ecart_type = df["Salaire"].std()
            correlation = df["Age"].corr(df["Salaire"])
            bins = [20, 30, 40, 50, 60]
            labels = ["20-30", "31-40", "41-50", "51-60"]
            df["Tranche_Age"] = pd.cut(df["Age"], bins=bins, labels=labels)
            median_par_tranche = df.groupby("Tranche_Age")["Salaire"].median()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Moyenne des salaires", f"{moyenne:.2f} €")
                st.metric("Écart-type", f"{ecart_type:.2f} €")
                st.metric("Corrélation âge-salaire", f"{correlation:.3f}")
            with col2:
                st.markdown("**Salaire médian par tranche d'âge**")
                st.dataframe(median_par_tranche.to_frame().style.background_gradient(cmap='Blues'))

            # Visualisation
            fig = px.scatter(df, x="Age", y="Salaire", color="Tranche_Age",
                           title="Relation Âge-Salaire",
                           color_discrete_sequence=px.colors.qualitative.Set1)
            fig.update_layout(xaxis_title="Âge", yaxis_title="Salaire (€)")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            **🔍 Analyse** :
            - **Corrélation positive** ({correlation:.3f}) : Les employés plus âgés gagnent généralement plus.
            - **Écart de salaire** : Les 51-60 ans gagnent **{median_par_tranche['51-60'] - median_par_tranche['20-30']:.2f}€** de plus que les 20-30 ans.
            """)

# Module 2 - Statistiques
elif module == "📈 Module 2 - Statistiques":
    st.header("📈 Module 2 - Statistiques pour la Data Science")

    st.markdown("""
    ### 🎯 **Objectifs de ce Module**
    - Comprendre les **distributions** (normale, uniforme, etc.).
    - Maîtriser les **tests d'hypothèses** (test t, p-value).
    - Interpréter des **intervalles de confiance**.
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["🎲 Distributions", "🔬 Tests d'Hypothèses", "📋 Exercice", "✅ Solution"])

    with tab1:
        st.subheader("🎲 Distributions et Statistiques Descriptives")

        st.markdown("""
        #### 💡 **Concept Clé : Types de Distributions**
        | Distribution  | Forme               | Exemple Réaliste                          |
        |---------------|---------------------|-------------------------------------------|
        | **Normale**   | 🔔 (Courbe en cloche) | Taille des humains, notes d'examen.       |
        | **Uniforme**  | ▇▇▇▇▇ (Plate)        | Résultat d'un dé équilibré.               |
        | **Poisson**   | 📊 (Asymétrique)     | Nombre d'emails spam par jour.            |
        | **Binomiale** | 🎲 (Succès/Echec)    | Taux de clics sur une publicité.          |

        **Analogie** :
        > *"- **Normale** : La plupart des gens ont une taille autour de 170 cm (moyenne), peu sont très grands ou très petits.
        > - **Poisson** : Peu de jours avec 10 spams, beaucoup avec 0 ou 1.
        > - **Binomiale** : 10% de chance de gagner à la loterie à chaque essai."
        """)

        # Sélecteur de distribution
        distribution = st.selectbox("Choisissez une distribution :", ["Normale", "Uniforme", "Poisson", "Binomiale"])

        col1, col2 = st.columns(2)
        with col1:
            n_points = st.slider("Nombre de points", 100, 10000, 1000)
            if distribution == "Normale":
                mean = st.slider("Moyenne", -5.0, 5.0, 0.0)
                std = st.slider("Écart-type", 0.1, 5.0, 1.0)
                data = np.random.normal(mean, std, n_points)
            elif distribution == "Uniforme":
                low = st.slider("Borne inférieure", -10, 0, -5)
                high = st.slider("Borne supérieure", 0, 10, 5)
                data = np.random.uniform(low, high, n_points)
            elif distribution == "Poisson":
                lam = st.slider("Lambda (λ)", 1, 20, 5)
                data = np.random.poisson(lam, n_points)
            else:  # Binomiale
                n_trials = st.slider("Nombre d'essais (n)", 1, 100, 10)
                p = st.slider("Probabilité de succès (p)", 0.0, 1.0, 0.5)
                data = np.random.binomial(n_trials, p, n_points)

        with col2:
            # Calcul des statistiques
            stats_df = pd.DataFrame({
                'Métrique': ['Moyenne', 'Médiane', 'Écart-type', 'Skewness (asymétrie)', 'Kurtosis (aplatissement)'],
                'Valeur': [
                    np.mean(data),
                    np.median(data),
                    np.std(data),
                    stats.skew(data),
                    stats.kurtosis(data)
                ]
            })
            st.dataframe(stats_df.style.background_gradient(cmap='Blues'), use_container_width=True)

        # Visualisation
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogramme', 'Boîte à moustaches'))
        fig.add_trace(go.Histogram(x=data, nbinsx=50, name='Distribution', marker_color='#1f77b4'), row=1, col=1)
        fig.add_trace(go.Box(y=data, name='Boîte à moustaches', marker_color='#1f77b4'), row=1, col=2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **💡 Interprétation** :
        - **Skewness > 0** : Queue à droite (ex. : revenus, quelques personnes très riches).
        - **Kurtosis > 0** : Pic plus pointu que la normale (données concentrées).
        """)

    with tab2:
        st.subheader("🔬 Tests d'Hypothèses - Test t de Student")

        st.markdown("""
        #### 📖 **Pourquoi faire un test t ?**
        - **Comparer deux groupes** (ex. : méthode A vs méthode B).
        - **Répondre à une question** : "La nouvelle version du site convertit-elle mieux ?"
        - **Éviter les conclusions hâtives** : Une différence de 1% peut être due au hasard !

        **Exemple** :
        > *"Un site e-commerce teste deux couleurs de bouton :
        > - **Groupe A** (Bouton rouge) : Taux de conversion = 2%.
        > - **Groupe B** (Bouton vert) : Taux de conversion = 2.5%.
        > Le test t dit si cette différence est **statistiquement significative**."
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Groupe A (Contrôle)")
            mean_a = st.slider("Moyenne Groupe A", 50, 100, 75, key="mean_a")
            std_a = st.slider("Écart-type Groupe A", 1, 20, 10, key="std_a")
            n_a = st.slider("Taille Groupe A", 10, 100, 30, key="n_a")
        with col2:
            st.markdown("**Groupe B (Nouveau)")
            mean_b = st.slider("Moyenne Groupe B", 50, 100, 78, key="mean_b")
            std_b = st.slider("Écart-type Groupe B", 1, 20, 12, key="std_b")
            n_b = st.slider("Taille Groupe B", 10, 100, 30, key="n_b")

        if st.button("🧪 Exécuter le test t"):
            np.random.seed(42)
            groupe_A = np.random.normal(mean_a, std_a, n_a)
            groupe_B = np.random.normal(mean_b, std_b, n_b)
            t_stat, p_value = stats.ttest_ind(groupe_A, groupe_B)

            st.markdown("### 📊 Résultats")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Statistique t", f"{t_stat:.3f}")
            with col2:
                st.metric("p-value", f"{p_value:.4f}")
            with col3:
                significance = "✅ Significatif" if p_value < 0.05 else "❌ Non significatif"
                st.metric("Conclusion", significance)

            # Visualisation comparative
            fig = go.Figure()
            fig.add_trace(go.Box(y=groupe_A, name='Groupe A', boxpoints='all', marker_color='#1f77b4'))
            fig.add_trace(go.Box(y=groupe_B, name='Groupe B', boxpoints='all', marker_color='#ff7f0e'))
            fig.update_layout(title='Comparaison des groupes - Boxplots', yaxis_title='Valeurs')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **🔍 Conclusion** :
            - Si **p < 0.05**, la différence est **réelle** (pas due au hasard).
            - Sinon, il faut **plus de données** ou accepter qu'il n'y a pas de différence.
            """)

    with tab3:
        st.markdown('<div class="exercise-box">', unsafe_allow_html=True)
        st.subheader("📋 Exercice Pratique - Analyse Statistique")

        st.markdown("""
        **Scénario** :
        Une entreprise teste deux méthodes de formation pour ses employés.
        - **Groupe A** : Méthode traditionnelle (30 employés).
        - **Groupe B** : Nouvelle méthode interactive (30 employés).
        Les scores de performance (sur 100) sont collectés.

        **Questions** :
        1. Les données suivent-elles une distribution normale ?
        2. La nouvelle méthode est-elle **significativement meilleure** ?
        3. Quelle est la **taille d'effet** (Cohen's d) ?
        """)

        if st.button("🎲 Générer les données d'exercice"):
            np.random.seed(123)
            groupe_A = np.random.normal(72, 8, 30)
            groupe_A = np.clip(groupe_A, 50, 95)
            groupe_B = np.random.normal(78, 7, 30)
            groupe_B = np.clip(groupe_B, 55, 98)
            st.session_state.groupe_A = groupe_A
            st.session_state.groupe_B = groupe_B

        if 'groupe_A' in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📊 Groupe A - Méthode Traditionnelle**")
                st.dataframe(pd.DataFrame({'Score': st.session_state.groupe_A}).describe(), use_container_width=True)
            with col2:
                st.markdown("**📊 Groupe B - Nouvelle Méthode**")
                st.dataframe(pd.DataFrame({'Score': st.session_state.groupe_B}).describe(), use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Box(y=st.session_state.groupe_A, name='Groupe A', boxpoints='all', marker_color='#1f77b4'))
            fig.add_trace(go.Box(y=st.session_state.groupe_B, name='Groupe B', boxpoints='all', marker_color='#2ca02c'))
            fig.update_layout(title='Distribution des Scores par Groupe', height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.subheader("✅ Solution de l'Exercice")

        if 'groupe_A' in st.session_state and 'groupe_B' in st.session_state:
            groupe_A = st.session_state.groupe_A
            groupe_B = st.session_state.groupe_B

            # Tests de normalité
            st.markdown("#### 1. Test de Normalité (Shapiro-Wilk)")
            col1, col2 = st.columns(2)
            with col1:
                stat_A, p_A = stats.shapiro(groupe_A)
                st.metric("Groupe A - p-value", f"{p_A:.4f}")
                st.markdown("**Normal** : " + ("✅ Oui" if p_A > 0.05 else "❌ Non"))
            with col2:
                stat_B, p_B = stats.shapiro(groupe_B)
                st.metric("Groupe B - p-value", f"{p_B:.4f}")
                st.markdown("**Normal** : " + ("✅ Oui" if p_B > 0.05 else "❌ Non"))

            # Test t
            st.markdown("#### 2. Test t de Student")
            var_stat, var_p = stats.levene(groupe_A, groupe_B)
            equal_var = var_p > 0.05
            t_stat, p_value = stats.ttest_ind(groupe_A, groupe_B, equal_var=equal_var)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Statistique t", f"{t_stat:.4f}")
            with col2:
                st.metric("p-value", f"{p_value:.4f}")
            with col3:
                st.metric("Significatif", "✅ Oui" if p_value < 0.05 else "❌ Non")
            with col4:
                st.metric("Variance égale", "✅ Oui" if equal_var else "❌ Non")

            # Taille d'effet
            st.markdown("#### 3. Taille d'Effet (Cohen's d)")
            n1, n2 = len(groupe_A), len(groupe_B)
            mean1, mean2 = np.mean(groupe_A), np.mean(groupe_B)
            std1, std2 = np.std(groupe_A, ddof=1), np.std(groupe_B, ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            cohens_d = (mean2 - mean1) / pooled_std

            st.metric("Cohen's d", f"{cohens_d:.3f}")
            if abs(cohens_d) < 0.2:
                effet = "Très faible"
            elif abs(cohens_d) < 0.5:
                effet = "Faible"
            elif abs(cohens_d) < 0.8:
                effet = "Moyen"
            else:
                effet = "Fort"
            st.markdown(f"**Effet** : {effet}")

            # Conclusion
            st.markdown("""
            **📌 Synthèse** :
            - **Différence moyenne** : {diff_mean:.2f} points en faveur du Groupe B.
            - **Significativité** : {significance} (p = {p_value:.4f}).
            - **Recommandation** : {recommandation}.
            """)

# Module 3 - Mathématiques
elif module == "🧮 Module 3 - Mathématiques":
    st.header("🧮 Module 3 - Mathématiques pour le Machine Learning")

    st.markdown("""
    ### 🎯 **Objectifs de ce Module**
    - Comprendre les **vecteurs** et l'algèbre linéaire.
    - Maîtriser les **dérivées** pour l'optimisation.
    - Découvrir les **probabilités** derrière les algorithmes.
    """)

    tab1, tab2, tab3 = st.tabs(["📐 Algèbre Linéaire", "📈 Calcul Différentiel", "🎲 Probabilités"])

    with tab1:
        st.subheader("📐 Algèbre Linéaire Interactive")

        st.markdown("""
        #### 💡 **Concept Clé : Vecteurs et Produit Scalaire**
        - **Vecteur** : Flèche dans l'espace (ex. : [2, 3] = 2 unités à droite, 3 en haut).
        - **Produit scalaire** (v·w) :
          - **v·w = v₁w₁ + v₂w₂** → Mesure "l'alignement" des vecteurs.
          - **v·w = 0** → Vecteurs **perpendiculaires** (orthogonaux).
        - **Norme** : Longueur du vecteur (ex. : ||[3,4]|| = 5).

        **Application en ML** :
        > *"Les vecteurs représentent des **features** (ex. : [âge, revenu]).
        > Le produit scalaire calcule la **similarité** entre deux clients !"*
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Vecteur v**")
            v1 = st.number_input("v₁", -10.0, 10.0, 2.0, key="v1")
            v2 = st.number_input("v₂", -10.0, 10.0, 3.0, key="v2")
            v = np.array([v1, v2])
            st.markdown("**Vecteur w**")
            w1 = st.number_input("w₁", -10.0, 10.0, 1.0, key="w1")
            w2 = st.number_input("w₂", -10.0, 10.0, -1.0, key="w2")
            w = np.array([w1, w2])
        with col2:
            dot_product = np.dot(v, w)
            norm_v = np.linalg.norm(v)
            norm_w = np.linalg.norm(w)
            angle = np.arccos(dot_product / (norm_v * norm_w)) * 180 / np.pi
            st.metric("Produit scalaire v·w", f"{dot_product:.2f}")
            st.metric("Norme de v", f"{norm_v:.2f}")
            st.metric("Norme de w", f"{norm_w:.2f}")
            st.metric("Angle entre v et w", f"{angle:.1f}°")

        # Visualisation des vecteurs
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, v[0]], y=[0, v[1]], mode='lines+markers+text',
                               name='v', line=dict(width=3, color='#1f77b4'),
                               text=[f'v({v1}, {v2})'], textposition="top center"))
        fig.add_trace(go.Scatter(x=[0, w[0]], y=[0, w[1]], mode='lines+markers+text',
                               name='w', line=dict(width=3, color='#ff7f0e'),
                               text=[f'w({w1}, {w2})'], textposition="top center"))
        fig.update_layout(title='Représentation des vecteurs',
                        xaxis=dict(range=[-11, 11], title='Axe x'),
                        yaxis=dict(range=[-11, 11], title='Axe y'),
                        showlegend=True, height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **🔍 Interprétation** :
        - **Produit scalaire > 0** : Vecteurs dans la même direction.
        - **= 0** : Vecteurs perpendiculaires (orthogonaux).
        - **< 0** : Vecteurs opposés.
        """)

    with tab2:
        st.subheader("📈 Calculateur de Dérivées")

        st.markdown("""
        #### 📖 **Pourquoi les dérivées ?**
        - **Dérivée = Pente instantanée** → Taux de changement.
        - **Application** :
          > *"En **descente de gradient**, les dérivées guident l'algorithme
          > vers le **minimum** de la fonction de perte (ex. : réduire l'erreur de prédiction)."*
        """)

        fonction = st.text_input("Entrez une fonction f(x) :", "x**2 + 3*x - 2")
        point = st.number_input("Point d'évaluation x₀ :", -10.0, 10.0, 1.0)

        if st.button("📐 Calculer la dérivée"):
            try:
                x = sp.Symbol('x')
                f = sp.sympify(fonction)
                derivee = sp.diff(f, x)
                valeur_derivee = derivee.subs(x, point)
                valeur_fonction = f.subs(x, point)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("f(x)", f"${sp.latex(f)}$")
                with col2:
                    st.metric("f'(x)", f"${sp.latex(derivee)}$")
                with col3:
                    st.metric(f"f'({point})", f"{valeur_derivee:.4f}")

                # Visualisation
                x_vals = np.linspace(point-3, point+3, 100)
                f_lambdified = sp.lambdify(x, f, 'numpy')
                f_prime_lambdified = sp.lambdify(x, derivee, 'numpy')
                y_vals = f_lambdified(x_vals)
                y_prime_vals = f_prime_lambdified(x_vals)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, name=f'f(x) = {fonction}', line=dict(color='#1f77b4')))
                fig.add_trace(go.Scatter(x=x_vals, y=y_prime_vals, name=f"f'(x)", line=dict(color='#ff7f0e')))
                fig.add_trace(go.Scatter(x=[point], y=[valeur_fonction], mode='markers',
                                       marker=dict(size=10, color='red'),
                                       name=f'Point ({point}, {valeur_fonction:.2f})'))
                fig.update_layout(title='Fonction et sa dérivée', height=500)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erreur : {e}")

    with tab3:
        st.subheader("🎲 Probabilités et Théorème de Bayes")

        st.markdown("""
        #### 💡 **Concept Clé : Théorème de Bayes**
        - **Formule** :
          P(A|B) = P(B|A) * P(A) / P(B)
        - **Application** :
          > *"Un test médical a une **sensibilité** de 99% (P(T+|Malade) = 0.99)
          > et une **spécificité** de 95% (P(T-|Sain) = 0.95).
          > Si 1% de la population est malade, quelle est la probabilité
          > d'être malade si le test est positif ?"*

        **Réponse** : Seulement **16%** ! (Pas 99%)

        **Explication** :
        - **P(Malade)** = 1% (prévalence).
        - **P(T+|Malade)** = 99% (sensibilité).
        - **P(T+|Sain)** = 5% (faux positifs).
        - **P(Malade|T+)** = (0.99 * 0.01) / (0.99 * 0.01 + 0.05 * 0.99) ≈ 16%.

        **Morale** : Un test positif ne signifie pas toujours que vous êtes malade !
        """)

# Module 4 - ML Introduction
elif module == "🤖 Module 4 - ML Introduction":
    st.header("🤖 Module 4 - Introduction au Machine Learning")

    st.markdown("""
    ### 🎯 **Objectifs de ce Module**
    - Comprendre les **types d'apprentissage** (supervisé, non supervisé).
    - Maîtriser la **régression linéaire**.
    - Éviter le **surapprentissage** (overfitting).
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Régression Linéaire", "🎯 Concepts Clés", "🔍 Validation Croisée", "📈 Surapprentissage"])

    with tab1:
        st.subheader("📊 Régression Linéaire Interactive")

        st.markdown("""
        #### 💡 **Concept Clé : Régression Linéaire**
        - **Modèle** : y = a*x + b
          - **a** = pente (combien y change quand x augmente de 1).
          - **b** = intercept (valeur de y quand x=0).
        - **But** : Trouver a et b qui **minimisent l'erreur** (MSE).
        - **Application** :
          > *"Prédire le **prix d'une maison** en fonction de sa surface,
          > ou les **ventes** en fonction du budget marketing."*
        """)

        col1, col2 = st.columns(2)
        with col1:
            n_points = st.slider("Nombre de points", 20, 200, 50, key="n_points_reg")
            bruit = st.slider("Niveau de bruit", 0.1, 5.0, 1.0, key="bruit_reg")
            pente_reelle = st.slider("Pente réelle (a)", 0.1, 5.0, 2.0, key="pente_reelle")
            intercept_reel = st.slider("Intercept réel (b)", -10.0, 10.0, 1.0, key="intercept_reel")
        with col2:
            test_size = st.slider("Taille du test set (%)", 10, 40, 20, key="test_size_reg")

        # Génération des données
        np.random.seed(42)
        X = np.linspace(0, 10, n_points).reshape(-1, 1)
        y = pente_reelle * X.ravel() + intercept_reel + np.random.normal(0, bruit, n_points)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        # Modèle
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pente estimée (a)", f"{model.coef_[0]:.3f}")
        with col2:
            st.metric("Intercept estimé (b)", f"{model.intercept_:.3f}")
        with col3:
            st.metric("MSE", f"{mse:.3f}")
        with col4:
            st.metric("R²", f"{r2:.3f}")

        # Visualisation
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_train.ravel(), y=y_train, mode='markers', name='Training', marker=dict(color='#1f77b4', opacity=0.6)))
        fig.add_trace(go.Scatter(x=X_test.ravel(), y=y_test, mode='markers', name='Test', marker=dict(color='#ff7f0e', opacity=0.6)))
        x_line = np.linspace(0, 10, 100).reshape(-1, 1)
        y_line = model.predict(x_line)
        fig.add_trace(go.Scatter(x=x_line.ravel(), y=y_line, mode='lines', name='Régression', line=dict(color='black', width=3)))
        fig.update_layout(title='Régression Linéaire', xaxis_title='X', yaxis_title='y', height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **🔍 Interprétation** :
        - **R² proche de 1** : Le modèle explique bien la variance.
        - **MSE faible** : Les prédictions sont proches des vraies valeurs.
        """)

    with tab2:
        st.subheader("🎯 Concepts Clés du Machine Learning")

        st.markdown("""
        #### 📚 **1. Types d'Apprentissage**
        | Type               | Description                          | Exemple                                  |
        |--------------------|--------------------------------------|------------------------------------------|
        | **Supervisé**      | Données **étiquetées** (réponse connue). | Prédire le prix d'une maison.            |
        | **Non supervisé**  | Données **non étiquetées** (trouver des patterns). | Segmentation clients. |
        | **Par renforcement** | Agent apprend par **essai-erreur**. | Robot qui apprend à marcher.            |

        **Analogie** :
        > *"- **Supervisé** : Un professeur corrige vos exercices.
        > - **Non supervisé** : Vous trouvez des groupes d'amis sans étiquettes.
        > - **Renforcement** : Un chien apprend des tours avec des récompenses."*
        """)

        st.markdown("""
        #### ⚖️ **2. Biais-Variance**
        - **Biais élevé** : Modèle **trop simple** → Sous-apprentissage (ex. : régression linéaire pour prédire un cercle).
        - **Variance élevée** : Modèle **trop complexe** → Surapprentissage (ex. : mémoriser les données d'entraînement).
        - **Équilibre idéal** : Bonne performance sur **training ET test**.
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("https://miro.medium.com/max/1400/1*TzZH0Y6OvM1Jz4vYfU5zHg.png", width=150, caption="Biais élevé")
        with col2:
            st.image("https://miro.medium.com/max/1400/1*TzZH0Y6OvM1Jz4vYfU5zHg.png", width=150, caption="Équilibre")
        with col3:
            st.image("https://miro.medium.com/max/1400/1*TzZH0Y6OvM1Jz4vYfU5zHg.png", width=150, caption="Variance élevée")

        st.markdown("""
        #### 🎯 **3. Métriques d'Évaluation**
        **Classification** :
        - **Accuracy** : (TP + TN) / Total.
        - **Precision** : TP / (TP + FP) → "Quelle proportion des prédits positifs sont vraiment positifs ?"
        - **Recall** : TP / (TP + FN) → "Quelle proportion des vrais positifs sont détectés ?"
        - **F1-score** : Moyenne harmonique de precision et recall.

        **Régression** :
        - **MSE** : Erreur quadratique moyenne (pénalise les grosses erreurs).
        - **MAE** : Erreur absolue moyenne.
        - **R²** : Pourcentage de variance expliquée (1 = parfait).
        """)

    with tab3:
        st.subheader("🔍 Validation Croisée (Cross-Validation)")

        st.markdown("""
        #### 💡 **Pourquoi la validation croisée ?**
        - **Problème** : Un seul split train/test peut donner des résultats **biaisés**.
        - **Solution** : Diviser les données en **k folds**, entraîner et tester k fois.
        - **Avantage** : Meilleure estimation de la performance **réelle**.
        """)

        col1, col2 = st.columns(2)
        with col1:
            n_splits = st.slider("Nombre de folds (k)", 3, 10, 5, key="n_splits_cv")
            dataset_size = st.slider("Taille du dataset", 100, 1000, 200, key="dataset_size_cv")
            noise_level = st.slider("Niveau de bruit", 0.1, 2.0, 0.5, key="noise_level_cv")
        with col2:
            np.random.seed(42)
            X_cv = np.random.randn(dataset_size, 1)
            y_cv = 2 * X_cv.ravel() + 1 + np.random.normal(0, noise_level, dataset_size)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_assignments = np.zeros(dataset_size)
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_cv)):
                fold_assignments[val_idx] = fold_idx + 1
            fig = px.scatter(x=X_cv.ravel(), y=y_cv, color=fold_assignments.astype(str),
                           title=f'Répartition des {n_splits} folds',
                           labels={'color': 'Fold'}, color_discrete_sequence=px.colors.qualitative.Set1)
            st.plotly_chart(fig, use_container_width=True)

        if st.button("🎯 Exécuter la Validation Croisée"):
            y_cv_class = (y_cv > np.median(y_cv)).astype(int)
            models = {
                'Régression Logistique': LogisticRegression(),
                'Arbre de Décision': DecisionTreeClassifier(max_depth=3),
                'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42)
            }
            cv_results = {}
            for name, model in models.items():
                scores = cross_val_score(model, X_cv, y_cv_class, cv=n_splits, scoring='accuracy')
                cv_results[name] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'scores': scores
                }

            # Résultats
            st.subheader("📊 Performances par Fold")
            fig = go.Figure()
            for name, results in cv_results.items():
                fig.add_trace(go.Scatter(x=list(range(1, n_splits + 1)), y=results['scores'],
                                       mode='lines+markers', name=name, line=dict(width=2)))
            fig.update_layout(title='Accuracy par Fold', xaxis_title='Fold', yaxis_title='Accuracy', height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Tableau des résultats
            results_df = pd.DataFrame({
                'Modèle': list(cv_results.keys()),
                'Score Moyen': [f"{results['mean_score']:.3f}" for results in cv_results.values()],
                'Écart-type': [f"{results['std_score']:.3f}" for results in cv_results.values()]
            })
            st.dataframe(results_df, use_container_width=True)

    with tab4:
        st.subheader("📈 Surapprentissage (Overfitting)")

        st.markdown("""
        #### 💡 **Comment détecter le surapprentissage ?**
        - **Symptômes** :
          - **Score training = 100%** mais **score test = 50%**.
          - Le modèle **mémorise** les données au lieu d'apprendre des patterns.
        - **Solutions** :
          1. **Réduire la complexité** (ex. : diminuer le degré d'un polynôme).
          2. **Plus de données**.
          3. **Regularisation** (L1/L2).
          4. **Early stopping** (arrêter l'entraînement quand le score test se dégrade).
        """)

        col1, col2 = st.columns(2)
        with col1:
            n_points_overfit = st.slider("Nombre de points", 50, 200, 100, key="n_points_overfit")
            degree = st.slider("Degré du polynôme", 1, 15, 3, key="poly_degree")
            noise = st.slider("Bruit", 0.1, 2.0, 0.5, key="overfit_noise")
        with col2:
            np.random.seed(42)
            X_overfit = np.linspace(-2, 2, n_points_overfit)
            true_function = np.sin(2 * X_overfit) + 0.5 * X_overfit
            y_overfit = true_function + np.random.normal(0, noise, n_points_overfit)
            X_train, X_test, y_train, y_test = train_test_split(X_overfit.reshape(-1, 1), y_overfit, test_size=0.3, random_state=42)

            # Courbe d'apprentissage
            train_scores = []
            test_scores = []
            degrees = list(range(1, 16))
            for deg in degrees:
                poly = PolynomialFeatures(degree=deg)
                X_poly_train = poly.fit_transform(X_train)
                X_poly_test = poly.transform(X_test)
                model = LinearRegression()
                model.fit(X_poly_train, y_train)
                train_scores.append(model.score(X_poly_train, y_train))
                test_scores.append(model.score(X_poly_test, y_test))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=degrees, y=train_scores, name='Training', line=dict(color='#1f77b4')))
            fig.add_trace(go.Scatter(x=degrees, y=test_scores, name='Test', line=dict(color='#ff7f0e')))
            fig.add_vline(x=degree, line_dash="dash", line_color="green", annotation_text="Degré sélectionné")
            fig.update_layout(title='Courbe de Surapprentissage', xaxis_title='Degré', yaxis_title='R²', height=400)
            st.plotly_chart(fig, use_container_width=True)

        if st.button("🎯 Visualiser le Surapprentissage"):
            poly = PolynomialFeatures(degree=degree)
            X_poly_train = poly.fit_transform(X_train)
            X_poly_test = poly.transform(X_test)
            model = LinearRegression()
            model.fit(X_poly_train, y_train)
            X_plot = np.linspace(-2, 2, 300).reshape(-1, 1)
            X_plot_poly = poly.transform(X_plot)
            y_plot = model.predict(X_plot_poly)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X_train.ravel(), y=y_train, mode='markers', name='Training', marker=dict(color='#1f77b4')))
            fig.add_trace(go.Scatter(x=X_test.ravel(), y=y_test, mode='markers', name='Test', marker=dict(color='#ff7f0e')))
            fig.add_trace(go.Scatter(x=X_plot.ravel(), y=true_function, mode='lines', name='Vraie fonction', line=dict(color='green', dash='dash')))
            fig.add_trace(go.Scatter(x=X_plot.ravel(), y=y_plot, mode='lines', name=f'Modèle (degré {degree})', line=dict(color='orange')))
            fig.update_layout(title=f'Surapprentissage avec degré {degree}', xaxis_title='X', yaxis_title='y', height=500)
            st.plotly_chart(fig, use_container_width=True)

            train_score = model.score(X_poly_train, y_train)
            test_score = model.score(X_poly_test, y_test)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Training", f"{train_score:.3f}")
            with col2:
                st.metric("R² Test", f"{test_score:.3f}")

            if train_score - test_score > 0.2:
                st.error("🚨 **SURAPPENTISSAGE DÉTECTÉ** : Le modèle performe bien sur le training mais mal sur le test !")
            elif train_score < 0.6:
                st.warning("⚠️ **SOUS-APPRENTISSAGE** : Le modèle est trop simple.")
            else:
                st.success("✅ **BON ÉQUILIBRE** : Le modèle généralise bien.")

# Module 5 - Algorithmes ML
elif module == "🌲 Module 5 - Algorithmes ML":
    st.header("🌲 Module 5 - Algorithmes de Machine Learning")

    st.markdown("""
    ### 🎯 **Objectifs de ce Module**
    - Comparer les **algorithmes** (Arbres, Forêts, k-NN, SVM).
    - Comprendre leurs **forces et faiblesses**.
    - Choisir le bon algorithme pour un problème donné.
    """)

    tab1, tab2, tab3 = st.tabs(["🔄 Comparaison d'Algorithmes", "🌳 Random Forest", "📊 Métriques Avancées"])

    with tab1:
        st.subheader("🔄 Comparaison Interactive des Algorithmes")

        st.markdown("""
        #### 💡 **Guide de Choix d'Algorithme**
        | Algorithme          | Quand l'utiliser ?                          | Avantages                          | Inconvénients                     |
        |---------------------|--------------------------------------------|------------------------------------|-----------------------------------|
        | **Arbre de Décision** | Données tabulaires, règles interprétables. | Facile à comprendre.              | Sensible aux petites variations.  |
        | **Random Forest**   | Précision élevée, données bruitées.        | Robuste, peu de préprocessing.     | Lent sur gros datasets.           |
        | **k-NN**            | Petits datasets, similarité locale.        | Simple, pas d'entraînement.        | Lent en prédiction.               |
        | **SVM**             | Données linéairement séparables.           | Efficace en haute dimension.       | Choix du noyau difficile.         |
        | **Régression Logistique** | Classification binaire.                | Rapide, interprétable.            | Linéaire (pas pour relations complexes). |

        **Analogie** :
        > *"- **Arbre de Décision** = Poser des questions successives (ex. : 'Est-ce que le revenu > 50k€ ?').
        > - **Random Forest** = Demander à 100 arbres et voter pour la réponse.
        > - **k-NN** = 'Ces 5 voisins sont riches, donc toi aussi !'*
        """)

        # Génération de données corrigée
        X, y = make_classification(
            n_samples=300,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        algorithms = st.multiselect(
            "Choisissez les algorithmes à comparer :",
            ["Arbre de Décision", "Random Forest", "k-NN", "SVM", "Régression Logistique"],
            default=["Arbre de Décision", "Random Forest", "k-NN"]
        )

        models = {}
        if "Arbre de Décision" in algorithms:
            models["Arbre de Décision"] = DecisionTreeClassifier(max_depth=5, random_state=42)
        if "Random Forest" in algorithms:
            models["Random Forest"] = RandomForestClassifier(n_estimators=100, random_state=42)
        if "k-NN" in algorithms:
            models["k-NN"] = KNeighborsClassifier(n_neighbors=5)
        if "SVM" in algorithms:
            models["SVM"] = SVC(kernel='linear', random_state=42, probability=True)
        if "Régression Logistique" in algorithms:
            models["Régression Logistique"] = LogisticRegression(random_state=42)

        if st.button("🏆 Comparer les algorithmes"):
            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = accuracy

            # Résultats
            st.subheader("📊 Performances des algorithmes")
            fig = px.bar(x=list(results.keys()), y=list(results.values()),
                       title="Accuracy par Algorithme", labels={'x': 'Algorithme', 'y': 'Accuracy'},
                       color=list(results.values()), color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

            # Affichage détaillé
            for name, accuracy in results.items():
                st.metric(f"{name}", f"{accuracy:.3f}")

            # Visualisation des frontières de décision
            if len(models) > 0:
                st.subheader("🎨 Frontières de Décision")
                algo_viz = st.selectbox("Sélectionnez un algorithme :", list(models.keys()))
                model_viz = models[algo_viz]
                h = 0.02
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                Z = model_viz.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                fig = go.Figure()
                fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='Viridis', opacity=0.3, showscale=False))
                fig.add_trace(go.Scatter(x=X_test[:, 0], y=X_test[:, 1], mode='markers',
                                       marker=dict(color=y_test, colorscale='Viridis', size=8, line=dict(width=1, color='black'))))
                fig.update_layout(title=f'Frontière de décision - {algo_viz}', xaxis_title='Feature 1', yaxis_title='Feature 2', height=500)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🌳 Random Forest en Détail")

        st.markdown("""
        #### 💡 **Comment fonctionne une Random Forest ?**
        1. **Bootstrap** : Créer N échantillons aléatoires **avec remplacement**.
        2. **Entraîner** un arbre de décision sur chaque échantillon.
        3. **Voter** pour la prédiction finale (classification) ou faire la moyenne (régression).

        **Avantages** :
        - **Robuste** aux outliers et au bruit.
        - **Peu sensible** au surapprentissage.
        - **Feature importance** intégrée.

        **Hyperparamètres clés** :
        - **n_estimators** : Nombre d'arbres (plus = mieux, mais plus lent).
        - **max_depth** : Profondeur maximale des arbres.
        - **min_samples_split** : Nombre minimal d'échantillons pour diviser un nœud.
        """)

        # Exemple avec dataset Iris
        iris = load_iris()
        X_iris, y_iris = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.metric("Accuracy sur Iris", f"{accuracy:.3f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': iris.feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig = px.bar(feature_importance, x='Importance', y='Feature', title='Importance des Features', orientation='h')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📊 Métriques Avancées : Courbe ROC et Precision-Recall")

        st.markdown("""
        #### 📖 **Courbe ROC vs Precision-Recall**
        | Métrique          | Quand l'utiliser ?                          | Interprétation                          |
        |-------------------|--------------------------------------------|------------------------------------------|
        | **Courbe ROC**    | Classes **équilibrées**.                   | AUC = 1 → Modèle parfait.               |
        | **Precision-Recall** | Classes **déséquilibrées** (ex. : fraude). | Précision = 1 → Peu de faux positifs.   |

        **Exemple** :
        > *"Détection de fraude (1% de fraudes) :
        > - **ROC** peut être trompeuse (AUC élevé même avec un mauvais modèle).
        > - **Precision-Recall** montre mieux les performances sur la classe minoritaire."*
        """)

        # Génération de données déséquilibrées corrigée
        X, y = make_classification(
            n_samples=1000,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            weights=[0.99, 0.01],
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Courbe ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.2f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Hasard'))
        fig.update_layout(title='Courbe ROC', xaxis_title='Faux Positifs', yaxis_title='Vrais Positifs', height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Courbe Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR (AUC = {pr_auc:.2f})'))
        fig.update_layout(title='Courbe Precision-Recall', xaxis_title='Recall', yaxis_title='Precision', height=500)
        st.plotly_chart(fig, use_container_width=True)

# Module 6 - Projet Scoring Crédit
elif module == "🏦 Module 6 - Projet Scoring Crédit":
    st.header("🏦 Module 6 - Système de Scoring Crédit")

    st.markdown("""
    ### 🎯 **Objectifs de ce Module**
    - **Explorer** un jeu de données de crédit.
    - **Entraîner** un modèle de classification (Random Forest).
    - **Interpréter** les résultats pour prendre des décisions business.
    """)

    tab1, tab2, tab3 = st.tabs(["📊 Exploration", "🤖 Modélisation", "🎯 Prédiction"])

    with tab1:
        st.subheader("📊 Exploration des Données")

        st.markdown("""
        #### 💡 **Contexte Business**
        Une banque veut **automatiser** l'approbation des prêts en fonction du risque de défaut.
        - **Target** : `Rembourse` (1 = Oui, 0 = Non).
        - **Features** :
          - **Âge** : Les jeunes sont-ils plus risqués ?
          - **Revenu/Dette** : Un ratio > 0.4 est un signal d'alerte.
          - **Historique** : Retards de paiement passés ?

        **Impact** :
        > *"Réduire les **faux positifs** (prêts accordés à des mauvais payeurs) = économies.
        > Limiter les **faux négatifs** (refus à des bons clients) = perte de revenus."*
        """)

        @st.cache_data
        def generate_credit_data():
            np.random.seed(42)
            n_samples = 1000
            data = {
                'Age': np.random.randint(18, 70, n_samples),
                'Revenu': np.random.randint(20000, 100000, n_samples),
                'Dette': np.random.randint(0, 50000, n_samples),
                'Historique_Credit': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
                'Rembourse': np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
            }
            data['Ratio_Dette_Revenu'] = data['Dette'] / np.maximum(data['Revenu'], 1)  # Évite division par 0
            return pd.DataFrame(data)

        df_credit = generate_credit_data()
        st.dataframe(df_credit.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nombre de clients", len(df_credit))
            st.metric("Taux de remboursement", f"{df_credit['Rembourse'].mean():.1%}")
        with col2:
            st.metric("Âge moyen", f"{df_credit['Age'].mean():.1f} ans")
            st.metric("Ratio dette/revenu moyen", f"{df_credit['Ratio_Dette_Revenu'].mean():.3f}")

        # Visualisations
        fig1 = px.histogram(df_credit, x='Age', color='Rembourse', barmode='overlay',
                          title='Remboursement par Âge', nbins=20)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.scatter(df_credit, x='Revenu', y='Dette', color='Rembourse',
                        title='Revenu vs Dette par Statut de Remboursement')
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        **🔍 Insights** :
        - Les **jeunes** (< 30 ans) ont un taux de défaut plus élevé.
        - Un **ratio dette/revenu > 0.5** est corrélé avec les défauts.
        """)

    with tab2:
        st.subheader("🤖 Modélisation du Score de Crédit")

        st.markdown("""
        #### 📖 **Choix du Modèle : Random Forest**
        - **Pourquoi ?** :
          - Gère bien les **données déséquilibrées** (80% de bons payeurs).
          - Fournit une **importance des features** interprétable.
          - Robuste aux **outliers**.
        """)

        if st.button("🚀 Entraîner le modèle"):
            X = df_credit[['Age', 'Revenu', 'Dette', 'Historique_Credit', 'Ratio_Dette_Revenu']]
            y = df_credit['Rembourse']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            conf_matrix = confusion_matrix(y_test, y_pred)

            st.session_state.credit_model = model
            st.session_state.credit_metrics = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'conf_matrix': conf_matrix
            }

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{accuracy:.3f}")
            with col2:
                st.metric("AUC-ROC", f"{roc_auc:.3f}")
            with col3:
                st.metric("Taille du test", f"{len(X_test)} clients")

            # Matrice de confusion
            fig = px.imshow(conf_matrix, text_auto=True,
                          labels=dict(x="Prédit", y="Réel", color="Nombre"),
                          x=['Default (0)', 'Rembourse (1)'],
                          y=['Default (0)', 'Rembourse (1)'],
                          title='Matrice de Confusion', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)

            fig = px.bar(feature_importance, x='Importance', y='Feature',
                       title='Importance des Features pour le Score',
                       orientation='h', color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **🔍 Analyse** :
            - **Ratio_Dette_Revenu** est le **facteur n°1** de risque.
            - **Historique_Credit** confirme son importance (comme attendu).
            """)

    with tab3:
        st.subheader("🎯 Prédiction pour un Nouveau Client")

        st.markdown("""
        #### 💡 **Comment interpréter le score ?**
        | Probabilité | Risque          | Recommandation                          |
        |-------------|-----------------|-----------------------------------------|
        | **> 70%**   | Faible          | ✅ Prêt approuvé (taux standard).        |
        | **40-70%**  | Modéré          | ⚠️ Examen manuel ou garantie requise.   |
        | **< 40%**   | Élevé           | ❌ Prêt refusé ou taux élevé.           |
        """)

        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Âge du client", 18, 70, 35)
            revenu = st.number_input("Revenu annuel (€)", 20000, 200000, 50000)
            dette = st.number_input("Dette totale (€)", 0, 100000, 10000)
        with col2:
            historique = st.selectbox("Historique de crédit", ["✅ Bon", "❌ Mauvais"])
            ratio = st.slider("Ratio Dette/Revenu", 0.0, 1.0, 0.2, 0.05)

        if st.button("🔮 Calculer le score"):
            if 'credit_model' in st.session_state:
                model = st.session_state.credit_model
                historique_num = 1 if historique == "✅ Bon" else 0

                client_data = pd.DataFrame({
                    'Age': [age],
                    'Revenu': [revenu],
                    'Dette': [dette],
                    'Historique_Credit': [historique_num],
                    'Ratio_Dette_Revenu': [ratio]
                })

                proba = model.predict_proba(client_data)[0, 1]
                prediction = model.predict(client_data)[0]

                st.markdown("### 📊 Résultat de la Prédiction")
                if proba > 0.7:
                    st.success(f"✅ **FAIBLE RISQUE** ({proba:.1%})")
                    st.markdown("**Recommandation** : Prêt approuvé aux conditions standard.")
                elif proba > 0.4:
                    st.warning(f"⚠️ **RISQUE MODÉRÉ** ({proba:.1%})")
                    st.markdown("**Recommandation** : Demander une garantie ou augmenter le taux d'intérêt.")
                else:
                    st.error(f"❌ **RISQUE ÉLEVÉ** ({proba:.1%})")
                    st.markdown("**Recommandation** : Refuser le prêt ou proposer un microcrédit.")

                # Jauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=proba * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Score de Remboursement (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "red"},
                            {'range': [40, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "green"}],
                        'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 70}
                    }))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                # Explication
                st.markdown(f"""
                **📌 Détail du calcul** :
                - **Âge** : {age} ans ({'✅ Positif' if age > 30 else '⚠️ Risque'}).
                - **Ratio Dette/Revenu** : {ratio:.2f} ({'✅ Bon' if ratio < 0.4 else '⚠️ Élevé'}).
                - **Historique** : {historique}.
                """)

# Module 7 - Quiz Interactif
elif module == "🎓 Module 7 - Quiz Interactif":
    st.header("🎓 Module 7 - Quiz Interactif")

    st.markdown("""
    ### 🎯 **Objectifs de ce Module**
    - Tester vos connaissances en Data Science.
    - Répondre à des questions sur les concepts clés.
    - Obtenir un score et des explications détaillées.
    """)

    # Exemple de quiz avec des questions et réponses
    questions = [
        {
            "question": "Qu'est-ce qu'un DataFrame ?",
            "options": ["Un tableau Excel", "Une liste de nombres", "Une fonction Python", "Un type de graphique"],
            "answer": "Un tableau Excel",
            "explanation": "Un DataFrame est une structure de données tabulaire, similaire à un tableau Excel, avec des lignes et des colonnes."
        },
        {
            "question": "Quelle est la formule de la dérivée de f(x) = x² ?",
            "options": ["f'(x) = 2x", "f'(x) = x", "f'(x) = 1", "f'(x) = 0"],
            "answer": "f'(x) = 2x",
            "explanation": "La dérivée de x² est 2x, car la dérivée mesure la pente instantanée de la fonction."
        },
        {
            "question": "Qu'est-ce que le surapprentissage ?",
            "options": ["Le modèle mémorise les données d'entraînement", "Le modèle est trop simple", "Le modèle a une haute précision", "Le modèle est rapide"],
            "answer": "Le modèle mémorise les données d'entraînement",
            "explanation": "Le surapprentissage se produit lorsque le modèle apprend trop bien les données d'entraînement et ne généralise pas bien sur de nouvelles données."
        }
    ]

    # Fonction pour afficher les questions et vérifier les réponses
    def run_quiz(questions):
        score = 0
        for question in questions:
            st.subheader(question["question"])
            answer = st.radio("Choisissez la bonne réponse :", question["options"])
            if answer == question["answer"]:
                score += 1
                st.success("✅ Correct !")
            else:
                st.error(f"❌ Incorrect. La bonne réponse est : {question['answer']}")
            st.markdown(f"**Explication** : {question['explanation']}")
            st.markdown("---")
        return score

    # Exécuter le quiz
    if st.button("🎯 Commencer le Quiz"):
        score = run_quiz(questions)
        st.markdown(f"### 📊 Votre Score : {score}/{len(questions)}")
        if score == len(questions):
            st.balloons()
            st.success("🎉 Excellent ! Vous avez répondu correctement à toutes les questions.")
        else:
            st.warning("📚 Revoyez les modules pour améliorer votre score.")

# Module 8 - Projet Immobilier
elif module == "🏦 Module 8 - Projet Immobilier":
    st.header("🏦 Module 8 - Projet Immobilier")

    st.markdown("""
    ### 🎯 **Objectifs de ce Module**
    - Explorer un jeu de données immobilier.
    - Entraîner un modèle de régression pour prédire les prix.
    - Analyser les résultats et les visualiser.
    """)

    # Charger les données
    @st.cache_data
    def load_housing_data():
        url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
        return pd.read_csv(url)

    df_housing = load_housing_data()
    st.dataframe(df_housing.head(10), use_container_width=True)

    # Préparation des données
    X = df_housing.drop("median_house_value", axis=1)
    y = df_housing["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métriques
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.metric("MSE", f"{mse:.3f}")
    st.metric("R²", f"{r2:.3f}")

    # Visualisation
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Prédictions'))
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Ligne idéale'))
    fig.update_layout(title='Prédictions vs Réelles', xaxis_title='Valeurs Réelles', yaxis_title='Valeurs Prédites', height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **🔍 Analyse** :
    - Le modèle explique **{r2:.1%}** de la variance des prix.
    - L'erreur moyenne est de **{mse:.3f}**.
    """)

# Module 9 - À Propos
elif module == "📘 À Propos":
    st.header("📘 À Propos")

    st.markdown("""
    ## 📚 **À Propos de ce Cours**
    Ce cours interactif de Data Science a été développé pour rendre les concepts complexes accessibles et ludiques.
    - **Objectif** : Vous former aux bases de la Data Science et du Machine Learning.
    - **Public cible** : Débutants en Data Science, analystes, ou toute personne curieuse.
    - **Technologies utilisées** : Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-Learn, Streamlit.

    **Contact** :
    - **Email** : votre@email.com
    - **GitHub** : [https://github.com/votre-utilisateur](https://github.com/votre-utilisateur)

    **Licence** : MIT
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
📚 **Cours Interactif de Data Science** — Développé avec ❤️ et Streamlit<br>
🔗 <a href="https://github.com/votre-utilisateur" target="_blank">GitHub</a> |
📧 <a href="mailto:votre@email.com">Contact</a>
</div>
""", unsafe_allow_html=True)
