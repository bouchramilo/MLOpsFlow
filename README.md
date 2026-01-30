# MLOps Pipeline – Déploiement & Monitoring d’un Service de Prédiction ML

## Description du projet

Ce projet consiste à concevoir et mettre en œuvre une **chaîne MLOps complète de bout en bout**, couvrant tout le cycle de vie d’un modèle de **Machine Learning**, depuis l’entraînement jusqu’au **déploiement en production** et au **monitoring en temps réel**.

Le modèle de prédiction est exposé via une **API REST développée avec FastAPI**, intégrée à un écosystème MLOps robuste incluant :
- **MLflow** pour le suivi des expérimentations et le versioning des modèles,
- **GitHub Actions** pour l’automatisation CI/CD,
- **Docker** pour un déploiement reproductible,
- **Prometheus & Grafana** pour la supervision et le monitoring du service.

Le projet a été réalisé **en binôme**, en respectant les bonnes pratiques de développement collaboratif et de production ML.

---

## Objectifs

- Mettre en place un pipeline MLOps fiable et maintenable
- Assurer la traçabilité complète des modèles (paramètres, métriques, versions)
- Déployer uniquement les modèles validés en **Production**
- Surveiller en temps réel les performances et la disponibilité de l’API
- Automatiser les tests, validations et le déploiement via CI/CD

---

## Fonctionnalités principales

### Machine Learning & MLflow
- Entraînement du modèle avec tracking MLflow
- Journalisation des paramètres, métriques et artefacts
- Comparaison des runs via l’interface MLflow
- Versioning des modèles dans le **Model Registry**
- Promotion automatique du modèle **Staging → Production**
- Chargement dynamique du modèle Production dans l’API

### API de prédiction
- API REST développée avec **FastAPI**
- Endpoint `/predict` pour la prédiction
- Validation des données d’entrée avec **Pydantic**
- Documentation interactive via **Swagger UI**
- Endpoint `/metrics` pour l’exposition des métriques Prometheus

### CI/CD avec GitHub Actions
- Lancement automatique des workflows à chaque `push`
- Tests unitaires ML et API
- Validation de la qualité des données
- Validation des performances du modèle
- Contrôle de la qualité du code
- Build et versioning de l’image Docker
- Déploiement continu du service

### Monitoring & Observabilité
- Collecte des métriques applicatives avec **Prometheus**
- Visualisation en temps réel avec **Grafana**
- Suivi des métriques clés :
  - Nombre de requêtes
  - Latence (p95)
  - Taux d’erreurs
  - Temps d’inférence
  - Consommation CPU / RAM du conteneur
- Dashboard Grafana prêt à l’emploi
- Alertes sur métriques critiques (optionnel)

---

## Structure du projet

```bash
.
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # Pipeline CI/CD GitHub Actions
│
├── api/
│   ├── main.py                # API FastAPI (endpoints /predict, /metrics)
│   ├── model_loader.py        # Chargement du modèle depuis MLflow
│   └── schemas.py             # Schémas Pydantic pour validation des données
│
├── ml/
│   ├── data/
│   │   ├── dataset-diabete.csv
│   │   └── dataset-diabete-processed.csv
│   ├── models/
│   │   └── model.pkl          # Modèle entraîné (local)
│   ├── preprossessing.py      # Prétraitement des données
│   ├── train.py               # Entraînement + logging MLflow
│   └── promote_model.py       # Promotion Staging → Production
│
├── monitoring/
│   ├── prometheus.yml         # Configuration Prometheus
│   ├── alert.rules.yml        # Règles d’alertes
│   └── grafana/
│       ├── dashboards/
│       │   └── mlops_dashboard.json
│       └── provisioning/
│           ├── datasources/
│           │   └── datasource.yml
│           └── dashboards/
│               └── dashboard.yml
│
├── tests/
│   ├── ml/
│   │   └── test_train.py
│   ├── test_api.py
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── Dockerfile                 # Image Docker du service ML
├── docker-compose.yml         # Orchestration API, MLflow, Prometheus, Grafana
├── requirements.txt           # Dépendances Python
└── README.md                  # Documentation du projet
```

---

## Technologies utilisées

- Python
- FastAPI
- MLflow
- Scikit-learn
- Pandas / NumPy
- Pydantic
- Docker & Docker Compose
- GitHub Actions
- Prometheus
- Grafana
- Pytest

---

## Installation et exécution du projet

1️⃣ Cloner le projet

```bash
git clone https://github.com/bouchramilo/MLOpsFlow.git
cd MLOpsFlow
```

2️⃣ Lancer l’infrastructure complète

```bash
docker-compose up --build
```

--- 

Merci.
