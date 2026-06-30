import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os

st.set_page_config(
    page_title="DiabètePredict CI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .header {
        background: linear-gradient(135deg, #C0392B, #7B241C);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .header h1 { margin: 0; font-size: 2.2rem; font-weight: 800; }
    .header p  { margin: 0.3rem 0 0; font-size: 0.95rem; opacity: 0.8; }

    .card-rouge {
        border-left: 5px solid #C0392B;
        background: #FDECEA;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .card-verte {
        border-left: 5px solid #1E8449;
        background: #EAFAF1;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .card-titre { font-size: 1.3rem; font-weight: 800; margin-bottom: 0.3rem; }
    .card-texte { font-size: 0.9rem; color: #444; }

    .stButton > button {
        background: #C0392B !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem !important;
    }
    .stButton > button:hover { background: #A93226 !important; }

    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def charger_modele():
    base = os.path.dirname(os.path.abspath(__file__))
    modele = joblib.load(os.path.join(base, '..', 'model', 'meilleur_modele.pkl'))
    scaler = joblib.load(os.path.join(base, '..', 'model', 'scaler.pkl'))
    return modele, scaler

modele, scaler = charger_modele()

# Moyennes du dataset Pima Indians Diabetes
MOYENNES = {
    "Grossesses": 3.8,
    "Glucose": 120.9,
    "Tension": 69.1,
    "Épaisseur peau": 20.5,
    "Insuline": 79.8,
    "IMC": 31.9,
    "Antécédents": 0.47,
    "Âge": 33.2,
}

COLONNES = ["Pregnancies", "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age"]

LABELS = ["Grossesses", "Glucose", "Tension", "Épaisseur peau",
          "Insuline", "IMC", "Antécédents", "Âge"]


with st.sidebar:
    st.markdown("### DiabètePredict CI")
    st.markdown("Projet ECP — Coding Academy C-DAT-900")
    st.divider()
    st.markdown("**Dataset**")
    st.markdown("- Source : Pima Indians Diabetes\n- 768 patientes\n- 8 variables cliniques\n- Classification binaire")
    st.divider()
    st.markdown("**Modèle**")
    st.markdown("- Random Forest\n- Précision : 77.9%\n- AUC : 0.818")
    st.divider()
    st.caption("Djamilat Diarrassouba — 2026")


st.markdown("""
<div class="header">
    <h1>DiabètePredict CI</h1>
    <p>Prédiction du risque de diabète à partir de données cliniques — Projet ECP · Coding Academy</p>
</div>
""", unsafe_allow_html=True)

st.caption("Renseignez les données du patient à gauche, puis cliquez sur Prédire. Cet outil est indicatif et ne remplace pas un avis médical.")
st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("#### Données cliniques")

    grossesses     = st.slider("Nombre de grossesses", 0, 20, 1)
    glucose        = st.slider("Glucose (mg/dL)", 50, 250, 120)
    tension        = st.slider("Tension artérielle diastolique (mmHg)", 20, 150, 70)
    epaisseur_peau = st.slider("Épaisseur du pli cutané (mm)", 0, 100, 25)
    insuline       = st.slider("Insuline sérique à 2h (mu U/ml)", 0, 900, 80)
    imc            = st.slider("IMC (kg/m²)", 10.0, 70.0, 32.0, step=0.1)
    antecedents    = st.slider("Score antécédents familiaux", 0.0, 2.5, 0.5, step=0.01)
    age            = st.slider("Âge (années)", 10, 100, 35)

    st.markdown("")
    bouton = st.button("Prédire le risque de diabète", use_container_width=True)


with col2:
    st.markdown("#### Résultat")

    if bouton:
        valeurs = [grossesses, glucose, tension, epaisseur_peau,
                   insuline, imc, antecedents, age]

        donnees = pd.DataFrame([valeurs], columns=COLONNES)
        donnees_norm = scaler.transform(donnees)
        prediction   = modele.predict(donnees_norm)[0]
        probabilite  = modele.predict_proba(donnees_norm)[0][1] * 100
        pct          = int(probabilite)

        # Carte résultat
        if prediction == 1:
            st.markdown(f"""
            <div class="card-rouge">
                <div class="card-titre" style="color:#C0392B;">Risque élevé de diabète</div>
                <div class="card-texte">Probabilité estimée : <b>{pct}%</b> — Une consultation médicale est recommandée.</div>
            </div>""", unsafe_allow_html=True)
            couleur = "#C0392B"
        else:
            st.markdown(f"""
            <div class="card-verte">
                <div class="card-titre" style="color:#1E8449;">Risque faible de diabète</div>
                <div class="card-texte">Probabilité estimée : <b>{pct}%</b> — Continuez un mode de vie sain.</div>
            </div>""", unsafe_allow_html=True)
            couleur = "#1E8449"

        # Jauge
        st.markdown(f"""
        <div style="margin:1rem 0 0.3rem; font-size:0.9rem; color:#555;">Probabilité de diabète</div>
        <div style="background:#e8e8e8; border-radius:20px; height:18px; overflow:hidden;">
            <div style="width:{pct}%; height:100%; background:{couleur}; border-radius:20px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#999; margin-top:4px;">
            <span>0%</span><span>Seuil : 50%</span><span>100%</span>
        </div>
        """, unsafe_allow_html=True)

        # Récapitulatif données saisies
        st.markdown("---")
        st.markdown("**Données saisies**")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Grossesses", grossesses)
        r2.metric("Glucose", f"{glucose} mg/dL")
        r3.metric("Tension", f"{tension} mmHg")
        r4.metric("Épaisseur peau", f"{epaisseur_peau} mm")
        r5, r6, r7, r8 = st.columns(4)
        r5.metric("Insuline", f"{insuline} mu U/ml")
        r6.metric("IMC", f"{imc:.1f}")
        r7.metric("Antécédents", f"{antecedents:.2f}")
        r8.metric("Âge", f"{age} ans")

        st.markdown("---")

        # Onglets pour les deux graphiques
        tab1, tab2 = st.tabs(["Importance des variables", "Comparaison aux moyennes"])

        with tab1:
            st.markdown("**Quelles variables ont le plus influencé la prédiction ?**")
            if hasattr(modele, 'feature_importances_'):
                importances = modele.feature_importances_
                indices = np.argsort(importances)
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('#0E1117')
                ax.set_facecolor('#0E1117')
                barres = ax.barh(
                    [LABELS[i] for i in indices],
                    importances[indices],
                    color=["#C0392B" if importances[i] == max(importances) else "#5D6D7E" for i in indices]
                )
                ax.set_xlabel("Importance", color="#aaa")
                ax.tick_params(colors="#ccc")
                ax.spines[:].set_color("#333")
                ax.set_title("Importance des variables (Random Forest)", color="white", fontsize=10)
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Graphique disponible uniquement avec Random Forest.")

        with tab2:
            st.markdown("**Comparaison de vos valeurs avec les moyennes du dataset (768 patientes)**")
            valeurs_patiente = [grossesses, glucose, tension, epaisseur_peau,
                                insuline, imc, antecedents, age]
            moyennes_liste   = list(MOYENNES.values())
            labels_court     = list(MOYENNES.keys())

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor('#0E1117')
            ax2.set_facecolor('#0E1117')

            x = np.arange(len(labels_court))
            width = 0.35

            # Normaliser pour comparaison visuelle
            max_vals = [max(v, m) for v, m in zip(valeurs_patiente, moyennes_liste)]
            norm_patient  = [v / m if m > 0 else 0 for v, m in zip(valeurs_patiente, max_vals)]
            norm_moyenne  = [m / m if m > 0 else 0 for m in max_vals]

            ax2.bar(x - width/2, valeurs_patiente, width, label="Patiente", color="#C0392B", alpha=0.85)
            ax2.bar(x + width/2, moyennes_liste,   width, label="Moyenne dataset", color="#5D6D7E", alpha=0.85)

            ax2.set_xticks(x)
            ax2.set_xticklabels(labels_court, rotation=30, ha='right', color="#ccc", fontsize=8)
            ax2.tick_params(axis='y', colors="#ccc")
            ax2.spines[:].set_color("#333")
            ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
            ax2.set_title("Valeurs patiente vs moyennes", color="white", fontsize=10)
            st.pyplot(fig2)
            plt.close()

            # Tableau différences
            diff_data = []
            for label, val, moy in zip(labels_court, valeurs_patiente, moyennes_liste):
                ecart = val - moy
                statut = "Au-dessus" if ecart > 0 else ("En-dessous" if ecart < 0 else "Dans la moyenne")
                diff_data.append({"Variable": label, "Valeur": val, "Moyenne": moy,
                                   "Écart": f"{ecart:+.1f}", "Statut": statut})
            st.dataframe(pd.DataFrame(diff_data), use_container_width=True, hide_index=True)

    else:
        st.info("Remplissez le formulaire à gauche puis cliquez sur **Prédire**.")
        st.markdown("---")
        st.markdown("**Comment ça fonctionne**")
        st.markdown("""
1. Entrez les données cliniques de la patiente
2. Le modèle Random Forest analyse les paramètres
3. Un score de probabilité de diabète est calculé
4. Le résultat s'affiche avec une recommandation

Modèle entraîné sur 768 patientes — Précision : 77.9% — AUC : 0.818
        """)

st.divider()
st.caption("DiabètePredict CI — Projet ECP | Djamilat Diarrassouba | Coding Academy C-DAT-900 | 2026")
