# Projet7-Openclassrooms

"Implémentez un modèle de scoring"

## Objectif du projet
* Développer un modèle de classification sur un jeu de données déséquilibré
* Mise en place d'une API Flask pour déployer le modèle de prédiction 
* Réaliser un dashboard interactif pour présenter et détailler ces prédictions
* Utilisation de github pour versioning et stockage du code

Source des données : https://www.kaggle.com/c/home-credit-default-risk/data

## Découpage des dossiers
* .github/workflows/ contient un seul fichier, le fichier yml contenant le code pour la pipeline CI/CD
* flask_application contient tous les scripts et fichier nécessaire au déploiement du modèle sur
  PythonAnywhere via flask
* mlruns/0 contient les runs réalisés avec mlflow (il n'y en a qun' seul actuellement pour l'exemple)
* streamlit_dashboard contient tous les scripts et fichier nécessaire au déploiement du dashboard sur
  Streamlit
* le notebook "Implémentez_un_modèle_de_scoring_EDA" contient l'analyse exploratoire, la création
  des datasets ainsi que le feature engineering
* le notebook "Test_different_modeles" contient l'entrainement du modèle et l'enregistrement des runs
  via mlflow
* le notebook "Analyse_data_drift" et le fichier html "data_drift_report" contiennent l'analyse et le
  résultat du DataDrift
* "utils.py" regroupe un ensemble de fonctions pythons utilisés lors de ce projet
