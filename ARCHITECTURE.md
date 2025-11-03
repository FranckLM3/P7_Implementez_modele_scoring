# Architecture - Credit Scoring Model

**Projet :** Implémentez un modèle de scoring (OpenClassrooms P7)  
**Date refactor :** Novembre 2025  
**Auteur :** Franck Le Mat

---

## Vue d'ensemble

Ce projet entraîne un modèle LightGBM pour prédire le risque de défaut de paiement de crédit. L'architecture a été refactorisée pour être **simple**, **maintenable** et **compatible avec l'API Cloud Run existante**.

### Principes
- **Minimaliste** : Uniquement le code nécessaire
- **Modulaire** : Package Python avec 4 modules
- **Reproductible** : Pipeline sklearn + hyperparamètres figés
- **Testé** : Tests unitaires basiques
- **Déployable** : Export joblib direct vers Cloud Run

---

## Structure du projet

```
P7_Implementez_modele_scoring/
├── src/credit_scoring/          # Package Python
│   ├── data.py                  # Chargement CSV
│   ├── features.py              # Preprocessing (imputation + scaling)
│   ├── model.py                 # Pipeline sklearn + LightGBM
│   └── metrics.py               # Métriques avec seuil G-mean
├── scripts/train.py             # CLI d'entraînement
├── train_lightgbm_modern.ipynb  # Notebook d'entraînement
├── tests/                       # Tests unitaires
├── artifacts/                   # Outputs (pipeline.joblib, metrics.json)
├── legacy/                      # Anciens fichiers (2022)
├── Makefile                     # Commandes rapides
├── pyproject.toml               # Configuration package
└── requirements.txt             # Dépendances

```

---

## Modules Python (`src/credit_scoring/`)

### 1. `data.py` - Chargement des données
```python
def load_dataset(path: str | Path, low_memory: bool = False) -> pd.DataFrame
```
- Charge `data/model_dataset.csv` avec encoding ISO-8859-1
- Gère les chemins relatifs/absolus

```python
def split_features_target(df: pd.DataFrame, target_col: str = "TARGET")
```
- Sépare features (X) et target (y)
- Remplace les valeurs infinies par NaN
- Retourne (X: DataFrame, y: Series)

### 2. `features.py` - Preprocessing

```python
def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer
```
Crée un ColumnTransformer sklearn qui:
1. **Détecte automatiquement** les colonnes numériques vs catégorielles
2. **Pipeline numérique:**
   - `SimpleImputer(strategy="median")` → remplit les NaN
   - `StandardScaler()` → normalise
3. **Pipeline catégoriel:**
   - `SimpleImputer(strategy="constant", fill_value="missing")` → remplit NaN
   - `OneHotEncoder(handle_unknown="ignore")` → encode

**Exemple:**
```python
from credit_scoring.features import build_preprocessor
preprocessor = build_preprocessor(X_train)
X_transformed = preprocessor.fit_transform(X_train)
```

### 3. `model.py` - Pipeline d'entraînement

**Note:** Ce module n'est **pas utilisé** dans l'architecture finale. Le script `train.py` construit directement le pipeline.

### 4. `metrics.py` - Évaluation

```python
@dataclass
class EvalReport:
    threshold: float
    accuracy: float
    recall: float
    precision: float
    f1: float
    gmean: float
    roc_auc: float
    tn: int
    fp: int
    fn: int
    tp: int
```

```python
def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> EvalReport
```
- Calcule la courbe ROC
- Trouve le seuil optimal avec **G-mean** = √(TPR × (1-FPR))
- Calcule toutes les métriques au seuil optimal
- Retourne `EvalReport` avec `.to_json()` pour export

**Pourquoi G-mean ?**  
Pour les classes déséquilibrées (8% de défauts), G-mean équilibre:
- **TPR** (True Positive Rate / Recall) → détecter les défauts
- **TNR** (True Negative Rate / Specificity) → ne pas rejeter les bons clients

---

## Script d'entraînement (`scripts/train.py`)

CLI pour entraîner le modèle LightGBM avec **hyperparamètres optimisés** (trouvés par optimisation bayésienne dans `legacy/MLflow.py`).

### Arguments

```bash
python scripts/train.py --help
```

- `--data` : Chemin CSV (défaut: `data/model_dataset.csv`)
- `--out-dir` : Dossier sortie (défaut: `artifacts/`)
- `--test-size` : Proportion test (défaut: 0.2)

### Workflow

1. **Chargement** : `load_dataset()` → DataFrame
2. **Nettoyage** : Remplace inf par NaN, supprime SK_ID_CURR
3. **Séparation** : X (features) et y (TARGET)
4. **Détection** : Colonnes numériques vs catégorielles
5. **Preprocessing** : ColumnTransformer (imputation + scaling + OHE)
6. **Feature selection** : SelectFromModel avec LightGBM
7. **Modèle** : LightGBM avec hyperparamètres optimaux
8. **Pipeline** : Assemblage preprocessing → selection → classifier
9. **Split** : Train/test stratifié (80/20)
10. **Entraînement** : `pipeline.fit(X_train, y_train)`
11. **Évaluation** : Métriques avec seuil G-mean optimal
12. **Export** : Artefacts pour déploiement

### Hyperparamètres LightGBM (optimisés)

```python
LGBMClassifier(
    n_estimators=1000,           # Nombre d'arbres
    learning_rate=0.02,          # Taux d'apprentissage
    num_leaves=44,               # Feuilles par arbre
    colsample_bytree=0.235,      # Échantillonnage features
    subsample=0.150,             # Échantillonnage observations
    max_depth=9,                 # Profondeur max
    reg_alpha=0.647,             # Régularisation L1
    reg_lambda=0.884,            # Régularisation L2
    min_split_gain=0.773,        # Gain minimal pour split
    min_child_weight=34.7,       # Poids minimal feuille
    scale_pos_weight=11.39,      # Gère le déséquilibre des classes
    random_state=123,
    verbose=-1
)
```

**Note:** `scale_pos_weight=11.39` est crucial pour gérer le déséquilibre (92% classe 0, 8% classe 1).

### Exports (artifacts/)

- **`pipeline.joblib`** (5.2M) : Pipeline complet sklearn prêt pour l'API
- **`metrics.json`** : Métriques d'évaluation (AUC, recall, precision, F1, seuil, confusion matrix)
- **`feature_importance.csv`** : Importances des features sélectionnées

### Utilisation

```bash
# Avec Makefile (recommandé)
make train

# Ou directement
.venv/bin/python scripts/train.py \
  --data data/model_dataset.csv \
  --out-dir artifacts
```

---

## Notebook (`train_lightgbm_modern.ipynb`)

Alternative interactive au script CLI. Permet de visualiser les résultats pas à pas.

**Contenu :**
1. Configuration et imports
2. Chargement des données
3. Preprocessing (ColumnTransformer)
4. Feature selection + LightGBM optimisé
5. Entraînement
6. Évaluation (métriques + visualisations)
7. Feature importance
8. Export du pipeline
9. Test du pipeline rechargé

**Avantages vs legacy :**
- Structure simple
- Utilise les hyperparamètres optimisés
- Visualisations modernes (ROC, G-mean, distributions, confusion matrix)
- Export compatible API Cloud Run

**Utilisation :**
1. Ouvrir dans VS Code
2. Sélectionner kernel `.venv`
3. Run All

---

## Tests (`tests/`)

Tests unitaires basiques pour vérifier le bon fonctionnement.

### `test_data.py`
- `test_split_features_target_minimal()` : Vérifie séparation X/y

### `test_model.py`
- `test_training_pipeline_runs_on_synthetic_data()` : Test E2E avec données synthétiques

**Lancer les tests :**
```bash
pytest tests/
# ou
make test
```

---

## Configuration

### `requirements.txt`

Dépendances minimales :

```
pandas>=2.2.0
numpy>=2.0.0
scikit-learn>=1.5.0
lightgbm>=4.5.0
joblib>=1.3.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

### `Makefile`

Commandes rapides :

```bash
make install    # pip install -r requirements.txt
make train      # Lance scripts/train.py
make test       # pytest tests/
make help       # Affiche l'aide
```

### `pyproject.toml`

Configuration du package Python :
- **name:** credit-scoring
- **version:** 0.1.0
- **Python:** ≥3.9
- **pytest:** pythonpath=src

---

## Workflow complet

### 1. Installation

```bash
cd P7_Implementez_modele_scoring
pip install -r requirements.txt
# ou
make install
```

### 2. Entraînement

**Option A : Script CLI (recommandé)**
```bash
make train
# ou
.venv/bin/python scripts/train.py
```

**Option B : Notebook interactif**
1. Ouvrir `train_lightgbm_modern.ipynb` dans VS Code
2. Sélectionner kernel `.venv`
3. Run All

**Résultat :** Génère dans `artifacts/` :
- `pipeline.joblib` (5.2M) → **à copier vers l'API**
- `metrics.json` → métriques d'évaluation
- `feature_importance.csv` → features sélectionnées

### 3. Déploiement sur Cloud Run

```bash
# Copier le pipeline vers l'API
cp artifacts/pipeline.joblib ../P7_api_deploy/ressource/

# Redéployer l'API
cd ../P7_api_deploy
gcloud run deploy credit-score-api --source . --region europe-west1

# Tester
curl -X POST "https://credit-score-api-572900860091.europe-west1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"id": 162473}'
```

### 4. Tests

```bash
pytest tests/
# ou
make test
```

---

## Flux de données

```
data/model_dataset.csv
    ↓ load_dataset()
DataFrame (307k lignes, 141 colonnes)
    ↓ Remplace inf → NaN, supprime SK_ID_CURR
    ↓ split X, y
X (features), y (TARGET)
    ↓ train_test_split(stratify=y)
X_train (80%), X_test (20%)
    ↓ 
ColumnTransformer
   ├─ Numeric: SimpleImputer(median) → StandardScaler
   └─ Categorical: SimpleImputer(constant) → OneHotEncoder
    ↓
SelectFromModel(LGBMClassifier)
   └─ Sélectionne ~200 features importantes
    ↓
LGBMClassifier (hyperparams optimisés)
   └─ 1000 arbres, scale_pos_weight=11.39
    ↓ pipeline.fit()
Pipeline entraîné
    ↓ pipeline.predict_proba()
y_proba (probabilités de défaut)
    ↓ compute_metrics()
EvalReport (AUC, seuil G-mean, métriques)
    ↓ joblib.dump()
artifacts/pipeline.joblib (5.2M)
    ↓
Cloud Run API (predict endpoint)
```

---

## Migration depuis legacy

### Avant (`legacy/P7_modelisation.ipynb` - 2022)

**Problèmes :**
- 42 cellules monolithiques
- Code dupliqué partout
- Preprocessing inline non réutilisable
- Pickle brut (classifier, feats, pipeline séparés)
- Difficile à maintenir/tester
- Hyperparamètres dispersés

### Après (architecture refactorisée - 2025)

**Améliorations :**
- **Modules Python** (`src/credit_scoring/`) : code réutilisable
- **Pipeline sklearn complet** : preprocessing + selection + model en 1 objet
- **Script CLI** (`scripts/train.py`) : entraînement en 1 commande
- **Notebook moderne** : 16 cellules claires avec visualisations
- **Hyperparamètres centralisés** : dans `train.py` (issus de `MLflow.py`)
- **Export propre** : 1 seul `pipeline.joblib` pour l'API
- **Tests unitaires** : garantit le fonctionnement
- **Legacy séparé** : anciens fichiers dans `legacy/`

---

## Avantages de l'architecture

| Critère | Avant | Après |
|---------|-------|-------|
| **Simplicité** | 42 cellules mélangées | 4 modules + 1 script |
| **Réutilisabilité** | Code dupliqué | Fonctions importables |
| **Reproductibilité** | Pickle brut | Pipeline sklearn + seed |
| **Maintenabilité** | Difficile à lire | Code clair et documenté |
| **Testabilité** | Aucun test | Tests unitaires |
| **Déploiement** | 3 fichiers séparés | 1 seul `pipeline.joblib` |
| **Documentation** | 9 guides complexes | 1 README + 1 ARCHITECTURE |

**En résumé :** Architecture minimaliste, fonctionnelle et maintenable.

---

## Support

**Documentation :**
- `README.md` : Guide utilisateur
- `ARCHITECTURE.md` : Ce document (détails techniques)
- `legacy/README.md` : Anciens fichiers (2022)

**Fichiers obsolètes :**
Tout ce qui est dans `legacy/` est conservé pour référence historique mais n'est **plus utilisé**.

**Contact :**
Franck Le Mat - franck.lemat.pro@proton.me
