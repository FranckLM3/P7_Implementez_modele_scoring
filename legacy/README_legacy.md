# Projet : Implémentez un modèle de scoring

**Autor :** Franck Le Mat

**Date :** 09/03/2022

**Durée totale :** 120 heures

Lien vers dashboard : https://francklm3-p7-dashboard-deploy-dashboard-67h1jz.streamlitapp.com/

## Background du projet :
Vous êtes Data Scientist au sein d'une société financière, nommée "Prêt à dépenser",  qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.
L’entreprise souhaite développer un modèle de scoring de la probabilité de défaut de paiement du client pour étayer la décision d'accorder ou non un prêt à un client potentiel en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).
De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.
Elle décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.
Données : https://www.kaggle.com/c/home-credit-default-risk/data
On accorde une importance particulière à la phase de cleaning des données du fait du déséquilibre des classes sur les targets.

## Key point du projet :
- Cleaning & Undersampling de la classe majoritaire
- Feature Engineering & OHE
- Imputation & Standardisation
- Classification avec Régression Logistique, RandomForestClassifier, LightGBM classifier, XGBoost
- Création d'une custom metric adaptée à la problématique métier (pénalisation forte des faux négatifs)
- Interprétabilité des modèles avec SHAP
- Création d'une API via fastAPI
- Création d'un dashboard intéractif pour le client avec Streamlit, Seaborn, & Pickle
- Déploiement du Dashboard dans le cloud d'Heroku


## Livrables :
- API :
- Le dashboard interactif :
- Un dossier sur un outil de versioning de code contenant :
    - Le code de la modélisation (du prétraitement à la prédiction)
    - Le code générant le dashboard
    - Le code permettant de déployer le modèle sous forme d'API
- Une note méthodologique décrivant :
    - La méthodologie d'entraînement du modèle
    - La fonction coût, l'algorithme d'optimisation et la métrique d'évaluation
    - L’interprétabilité du modèle
    - Les limites et les améliorations possibles
- Un support de présentation


## Compétences évaluées :
- Présenter son travail de modélisation à l'oral
- Déployer un modèle via une API dans le Web
- Utiliser un logiciel de version de code pour assurer l’intégration du modèle
- Rédiger une note méthodologique afin de communiquer sa démarche de modélisation
- Réaliser un dashboard pour présenter son travail de modélisation
