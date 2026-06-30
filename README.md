# DiabètePredict CI
### Système de prédiction du risque de diabète basé sur le Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Dataset](https://img.shields.io/badge/Dataset-Pima%20Indians-green)

---

## Contexte

Le diabète est l'une des maladies chroniques les plus répandues
en Côte d'Ivoire et en Afrique de l'Ouest. Souvent détectée trop
tard faute d'outils accessibles, cette maladie peut pourtant être
anticipée grâce à des données médicales simples.

**DiabètePredict CI** est une application web interactive qui permet
à n'importe qui d'estimer son risque de diabète en entrant ses
informations médicales de base.

---

## Problématique

Comment le Machine Learning peut-il aider à identifier les personnes
à risque de diabète à partir de données médicales simples,
avant même un diagnostic clinique officiel ?


## Structure du projet

diabete-predict-ci/
│

├── data/

│   ├── diabetes.csv          ← Dataset brut (Pima Indians)

│   └── diabetes_clean.csv    ← Dataset nettoyé

│

├── notebooks/

│   ├── EDA.ipynb          ← Analyse exploratoire des données

│   └──Modelisation.ipynb ← Entraînement et évaluation des modèles

│

├── app/

│   └── app.py                ← Dashboard Streamlit interactif

│

├── model/

│   ├── meilleur_modele.pkl   ← Modèle Random Forest sauvegardé

│   └── scaler.pkl            ← Normaliseur sauvegardé

│

├── docs/

│   └── Roadmap_ECP.docx      ← Roadmap du projet

│

└── README.md

---

##  Dataset

- **Nom** : Pima Indians Diabetes Dataset
- **Source** : [Kaggle / UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Taille** : 768 patientes × 9 variables médicales
- **Cible** : `Outcome` → 0 = Non diabétique / 1 = Diabétique

| Variable | Description |
|---|---|
| Pregnancies | Nombre de grossesses |
| Glucose | Taux de glucose (mg/dL) |
| BloodPressure | Tension artérielle (mm Hg) |
| SkinThickness | Épaisseur du pli cutané (mm) |
| Insulin | Taux d'insuline (mu U/ml) |
| BMI | Indice de masse corporelle |
| DiabetesPedigreeFunction | Score d'antécédents familiaux |
| Age | Âge de la patiente |

---

## Pipeline Machine Learning
Données brutes → Nettoyage → EDA → Modélisation → Dashboard

**1. Analyse exploratoire (EDA)**
- Détection des zéros biologiquement impossibles
- Visualisation des distributions et corrélations
- Insight clé : le glucose est la variable la plus corrélée (0.47)

**2. Nettoyage**
- Remplacement des zéros suspects par la médiane de chaque colonne
- Normalisation avec StandardScaler

**3. Modèles testés**

| Modèle | Accuracy | AUC |
|---|---|---|
| Logistic Regression | 70.8% | 0.813 |
| **Random Forest**  | **77.9%** | **0.818** |
| SVM | 74.0% | 0.796 |

→ **Random Forest** sélectionné comme meilleur modèle

---

## Dashboard Streamlit

L'application permet à l'utilisateur de :
- Entrer ses données médicales via des curseurs interactifs
- Obtenir une prédiction de risque en temps réel (0 ou 1)
- Visualiser sa probabilité de diabète avec une jauge
- Consulter un récapitulatif de ses données saisies

---

## Installation et lancement

**1. Cloner le repo**
```bash
git clone https://github.com/[organisation-epitech]/diabete-predict-ci.git
cd diabete-predict-ci
```

**2. Installer les dépendances**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

**3. Lancer le dashboard**
```bash
cd app
streamlit run app.py
```

---

## Technologies utilisées

- **Python 3.x**
- **pandas** — manipulation des données
- **numpy** — calculs numériques
- **matplotlib / seaborn** — visualisations
- **scikit-learn** — Machine Learning
- **Streamlit** — dashboard interactif
- **joblib** — sauvegarde du modèle

---

## Auteure

**Djamilat**
Formation Data / Intelligence Artificielle
Coding Academy by Epitech — Abidjan, Côte d'Ivoire

---

## Disclaimer

Cette application est développée à des fins éducatives dans le cadre
d'un projet de fin de formation. Elle ne constitue pas un diagnostic
médical. Consultez toujours un professionnel de santé.