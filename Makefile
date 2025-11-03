# Makefile Simple - Credit Scoring

PYTHON := .venv/bin/python
VENV := .venv/bin/python

.PHONY: install train test help

# Afficher l'aide
help:
	@echo "Commandes disponibles:"
	@echo "  make install    - Installer le package credit_scoring"
	@echo "  make train      - Entraîner le modèle LightGBM"
	@echo "  make test       - Lancer les tests (optionnel)"
	@echo ""
	@echo "Vous pouvez aussi utiliser directement:"
	@echo "  python scripts/train.py --data data/model_dataset.csv --out-dir artifacts"

# Installer le package (équivalent à: pip install -e .)
install:
	@echo "Installation du package credit_scoring..."
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .
	@echo "Installation terminée!"

# Entraîner le modèle
train:
	@echo "Entraînement du modèle LightGBM..."
	$(PYTHON) scripts/train.py --data data/model_dataset.csv --out-dir artifacts
	@echo "Entraînement terminé! Modèle dans: artifacts/pipeline.joblib"

# Lancer les tests (optionnel)
test:
	@echo "Lancement des tests..."
	$(PYTHON) -m pytest -q tests/
	@echo "Tests terminés!"
