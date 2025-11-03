# Legacy - Anciens fichiers du projet

Ce dossier contient les anciens fichiers du projet original (2022) avant la refactorisation de 2025.

## Notebooks originaux

- **P7_EDA.ipynb** : Exploration des données (Exploratory Data Analysis)
- **P7_modelisation.ipynb** : Notebook d'entraînement original (42 cellules)
- **P7_functions.py** : Fonctions utilitaires originales

## Anciens modèles (old_models/)

Modèles entraînés avec le notebook original, sauvegardés en pickle :

- **classifier** : Modèle LightGBM original (pickle)
- **feats** : Features sélectionnées (pickle)
- **pipeline** : Pipeline sklearn original (pickle)
- **lime_explainer** : Explainer LIME (pickle, 119 KB)
- **shap_explainer** : Explainer SHAP (pickle, 4.9 MB)

## Documents et livrables

- **P7_methodologie.docx** / **.pdf** : Note méthodologique du projet
- **P7_support.pdf** : Support de présentation
- **P7_franck_lemat.zip** : Archive complète du projet original
- **Procfile** : Configuration Heroku (obsolète)

## MLflow

- **MLflow.py** : Script d'optimisation des hyperparamètres
  - Contient les hyperparamètres optimaux trouvés via RandomizedSearchCV
  - Ces hyperparamètres sont maintenant intégrés dans `train_lightgbm_modern.ipynb`

## Ancienne documentation

- **README_old.md** : Ancien README avant simplification

## Différences avec l'architecture refactorée

| Aspect | Ancien (legacy/) | Nouveau (racine) |
|--------|------------------|------------------|
| **Notebooks** | P7_modelisation.ipynb (42 cellules) | train_lightgbm_modern.ipynb (22 cellules) |
| **Code** | Inline dans notebooks | Modules `src/credit_scoring/` |
| **Modèles** | Pickles séparés (classifier, feats, etc.) | Pipeline sklearn complet (joblib) |
| **Tests** | Aucun | `tests/` avec pytest |
| **CI/CD** | Aucune | `.github/workflows/ci.yml` |
| **Deployment** | Heroku | Cloud Run |

## Pourquoi ces fichiers sont obsolètes ?

1. **Notebooks monolithiques** : Tout le code dans une seule cellule, difficile à maintenir
2. **Pas de tests** : Aucun test automatique, risque de régression
3. **Code non réutilisable** : Copier-coller du code preprocessing partout
4. **Pickles multiples** : Difficile de gérer plusieurs fichiers (classifier, feats, pipeline, etc.)
5. **Pas de CI/CD** : Tests et déploiement manuels

## Avantages de la nouvelle architecture

1. **Code modulaire** : `src/credit_scoring/` réutilisable
2. **Tests automatiques** : `pytest` + CI sur GitHub Actions
3. **Pipeline unique** : Un seul fichier `pipeline.joblib` (preprocessing + modèle)
4. **Documentation** : ARCHITECTURE.md + notebook avec explications
5. **Déploiement moderne** : Cloud Run avec CI/CD

---

**Note :** Ces fichiers sont conservés pour référence historique. Pour entraîner un nouveau modèle, utilisez `train_lightgbm_modern.ipynb` à la racine du projet.
