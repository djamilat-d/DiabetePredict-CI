import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="DiabètePredict CI", page_icon=" ", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def charger_modele():
    modele = joblib.load("../model/meilleur_modele.pkl")
    scaler = joblib.load("../model/scaler.pkl")
    return modele,scaler

modele, scaler = charger_modele()

st.title("DiabètePredict CI")
st.markdown("Système de Prediction de risque de diabète")
st.markdown("""
Cette application utilise un modèle de **Machine Learning (Random Forest)**
entraîné sur des données médicales réelles pour estimer le risque de diabète
d'une patiente à partir de ses informations médicales.

> *Cet outil est à titre indicatif uniquement.
> Il ne remplace pas un diagnostic médical professionnel.*
""")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Entrez vos informations médicales")
    st.markdown("*Déplacez les curseurs selon vos valeurs*")
    st.markdown("")
    
    grossesses = st.slider(
        "Nombre de grossesses",
        min_value=0, max_value=20, value=1,
        help="Nombre total de grossesses"
    )

    glucose = st.slider(
        "Taux de glucose (mg/dL)",
        min_value=50, max_value=250, value=120,
        help="Taux de glucose mesuré lors d'un test oral"
    )

    tension = st.slider(
        "Tension artérielle diastolique (mm Hg)",
        min_value=20, max_value=150, value=70,
        help="Pression artérielle diastolique"
    )

    epaisseur_peau = st.slider(
        "Épaisseur du pli cutané (mm)",
        min_value=0, max_value=100, value=25,
        help="Épaisseur mesurée au niveau du triceps"
    )

    insuline = st.slider(
        "Taux d'insuline (mu U/ml)",
        min_value=0, max_value=900, value=80,
        help="Taux d'insuline mesuré 2h après un test oral"
    )

    imc = st.slider(
        "IMC — Indice de Masse Corporelle",
        min_value=10.0, max_value=70.0, value=32.0, step=0.1,
        help="Poids (kg) / Taille² (m)"
    )

    antecedents = st.slider(
        "Score d'antécédents familiaux (DiabetesPedigreeFunction)",
        min_value=0.0, max_value=2.5, value=0.5, step=0.01,
        help="Score calculé selon les antécédents familiaux de diabète"
    )

    age = st.slider(
        "Âge (années)",
        min_value=10, max_value=100, value=35,
        help="Âge de la patiente"
    )
    
    st.markdown("")
    bouton_prediction = st.button( "Predire mon risque de diabète", use_container_width=True)
    

with col2:
    st.subheader("Résultat de la prédiction")

    if bouton_prediction:

        # On rassemble les données saisies dans un tableau
        donnees_patiente = pd.DataFrame([[
            grossesses, glucose, tension, epaisseur_peau,
            insuline, imc, antecedents, age
        ]], columns=[
            "Pregnancies", "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age"
        ])
        donnees_normalisees = scaler.transform(donnees_patiente)
        prediction = modele.predict(donnees_normalisees)[0]
        probabilite = modele.predict_proba(donnees_normalisees)[0][1] * 100
        st.markdown("")
        
        if prediction == 1:
            st.error("Risque élevé de diabète détecté")
            st.markdown(f"""
            ### Probabilité de diabète : `{probabilite:.1f}%`
            """)
            st.markdown("""
            **Recommandation :** Consultez un professionnel de santé
            dès que possible pour un diagnostic complet.
            """)
        else:
            st.success("Faible risque de diabète")
            st.markdown(f"""
            ### Probabilité de diabète : `{probabilite:.1f}%`
            """)
            st.markdown("""
            **Recommandation :** Continuez à maintenir
            un mode de vie sain et faites des bilans réguliers.
            """)
            
            st.markdown("#### Niveau de risque")
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.barh(
            ["Risque"],
            [probabilite],
            color="#E74C3C" if probabilite > 50 else "#2ECC71",
            height=0.4
        )
        ax.barh(
            ["Risque"],
            [100 - probabilite],
            left=[probabilite],
            color="#EAECEE",
            height=0.4
        )
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probabilité (%)")
        ax.axvline(x=50, color="orange", linestyle="--",
                   linewidth=1.5, label="Seuil 50%")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Probabilité de diabète : {probabilite:.1f}%",
                     fontsize=11, fontweight="bold")
        st.pyplot(fig)

        # --- Récapitulatif des données saisies ---
        st.markdown("#### Vos données saisies")
        recap = pd.DataFrame({
            "Variable"  : [
                "Grossesses", "Glucose", "Tension",
                "Épaisseur peau", "Insuline", "IMC",
                "Antécédents", "Âge"
            ],
            "Valeur"    : [
                grossesses, glucose, tension,
                epaisseur_peau, insuline, imc,
                antecedents, age
            ]
        })
        st.dataframe(recap, use_container_width=True, hide_index=True)

    else:
        
        st.info("""
         Remplissez le formulaire à gauche
        puis cliquez sur **Prédire mon risque**
        pour obtenir votre résultat.
        """)
        st.markdown("")
        st.markdown("#### Comment fonctionne cette application ?")
        st.markdown("""
        1. Entrez vos informations médicales à gauche
        2. Cliquez sur le bouton de prédiction
        3. Le modèle **Random Forest** analyse vos données
        4. Vous obtenez votre niveau de risque en temps réel

        Le modèle a été entraîné sur **768 patientes**
        avec une précision de **77.9%** et un score AUC de **0.818**.
        """)
        
st.divider()
st.markdown("""
<div style='text-align: center; color: grey; font-size: 13px;'>
DiabètePredict CI — End of Course Project<br>
Modèle : Random Forest | Dataset : Pima Indians Diabetes (UCI/Kaggle)
</div>
""", unsafe_allow_html=True)

