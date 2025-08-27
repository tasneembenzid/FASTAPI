# Guide d'Intégration ML - Gradient Boosting pour Credit Scoring

## 🎯 Vue d'ensemble

Votre API FastAPI est maintenant intégrée avec un modèle de Machine Learning Gradient Boosting pour le scoring de crédit. Cette intégration comprend :

- **Modèle ML** : Gradient Boosting Classifier avec hyperparameter tuning
- **Preprocessing** : Validation des données et feature engineering
- **API Endpoints** : Endpoints enrichis pour prédiction et gestion du modèle
- **Fallback** : Système de règles de base si le modèle ML n'est pas disponible

## 📁 Structure des Fichiers

```
fastapi/
├── main.py                    # API FastAPI avec intégration ML
├── ml_model.py               # Gestionnaire du modèle ML
├── data_preprocessing.py     # Preprocessing et feature engineering
├── train_model.py           # Script d'entraînement du modèle
├── test_api.py              # Tests de l'API
├── models/                  # Répertoire pour les modèles sauvegardés
│   └── credit_scoring_model.joblib
├── requirements.txt         # Dépendances Python
└── *.postman_collection.json # Collections Postman
```

## 🚀 Démarrage Rapide

### 1. Installer les Dépendances ML
```bash
pip install scikit-learn pandas numpy joblib
```

### 2. Entraîner le Modèle (Optionnel)
```bash
python train_model.py
```

### 3. Démarrer l'API
```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Tester l'API
```bash
python test_api.py
```

## 🔧 Nouveaux Endpoints API

### 1. Prédiction ML Enrichie
**POST** `/predict`

Maintenant utilise le modèle Gradient Boosting avec :
- Validation automatique des données
- Feature engineering avancé
- Prédiction ML ou règles de fallback
- Analyse détaillée des facteurs de risque

**Réponse enrichie :**
```json
{
  "client_id": "CLIENT_001",
  "risk_score": 85,
  "risk_class": "Faible",
  "decision": "Approuvé",
  "confidence": 0.92,
  "model_info": {
    "model_version": "gradient_boost_v1.0",
    "prediction_method": "gradient_boosting",
    "weighted_score": 87.5
  },
  "factors_analyzed": {
    "revenu_mensuel": 4500.0,
    "ratio_endettement": 0.25,
    "employment_stability": 0.8,
    "credit_history_score": 95.0,
    "debt_to_income": 0.29,
    "risk_flags_count": 0
  },
  "engineered_features": {
    "debt_to_income": 0.29,
    "savings_to_income": 0.33,
    "employment_stability": 0.8,
    "credit_history_score": 95.0,
    "payment_burden": 0.089,
    "age_group": "middle_age"
  }
}
```

### 2. Informations sur le Modèle
**GET** `/model/info`

Retourne des informations sur le modèle actuel :
```json
{
  "model_info": {
    "model_type": "GradientBoostingClassifier",
    "is_trained": true,
    "feature_count": 40,
    "categorical_features": 11
  },
  "preprocessor_available": true,
  "api_version": "2.0.0",
  "status": "operational"
}
```

### 3. Validation des Données
**POST** `/model/validate`

Valide les données d'entrée sans faire de prédiction :
```json
{
  "validation_status": "success",
  "validated_data": { ... },
  "engineered_features": { ... },
  "feature_count": 45
}
```

### 4. Importance des Features
**GET** `/features/importance`

Retourne l'importance des features utilisées :
```json
{
  "feature_weights": {
    "revenu_mensuel": 0.15,
    "ratio_endettement": 0.12,
    "nombre_incidents_12m": 0.10
  },
  "sorted_features": [
    ["revenu_mensuel", 0.15],
    ["ratio_endettement", 0.12]
  ]
}
```

## 🧠 Features du Modèle ML

### Features Principales
- **Financières** : revenus, dettes, ratios financiers
- **Emploi** : ancienneté, stabilité, type de contrat
- **Historique** : incidents, retards, comportement de paiement
- **Démographiques** : âge, situation familiale, région
- **Crédit** : montant demandé, durée, garanties

### Features Engineered
- `debt_to_income` : Ratio dette/revenu annuel
- `savings_to_income` : Ratio épargne/revenu annuel
- `employment_stability` : Score de stabilité d'emploi (0-1)
- `credit_history_score` : Score d'historique de crédit (0-100)
- `payment_burden` : Charge de remboursement mensuelle
- `age_group` : Catégorie d'âge
- `risk_flags_count` : Nombre de drapeaux de risque

## 🔄 Intégration de Votre Modèle

### Option 1 : Remplacer le Modèle par Défaut

1. **Entraînez votre modèle** avec vos données réelles
2. **Sauvegardez le modèle** :
```python
from ml_model import CreditScoringModel

# Votre modèle entraîné
your_model = GradientBoostingClassifier()
your_model.fit(X_train, y_train)

# Sauvegarder
model_manager = CreditScoringModel()
model_manager.model = your_model
model_manager.scaler = your_scaler
model_manager.is_trained = True
model_manager.save_model('models/credit_scoring_model.joblib')
```

### Option 2 : Utiliser le Script d'Entraînement

1. **Modifiez `train_model.py`** pour charger vos données :
```python
def load_your_data():
    # Remplacez generate_synthetic_data() par votre chargement de données
    df = pd.read_csv('your_data.csv')
    return df
```

2. **Exécutez l'entraînement** :
```bash
python train_model.py
```

### Option 3 : Intégration Personnalisée

Modifiez `ml_model.py` pour adapter :
- Les features attendues
- Le preprocessing
- La logique de prédiction
- Les métriques de sortie

## 📊 Configuration Postman

### Nouvelle Collection
Importez `Credit_Scoring_API.postman_collection.json` qui inclut maintenant :
- Tests des nouveaux endpoints ML
- Exemples de données enrichies
- Variables d'environnement mises à jour

### Tests Recommandés
1. **Health checks** - Vérifier l'état de l'API
2. **Model info** - Vérifier le modèle chargé
3. **Data validation** - Tester la validation des données
4. **Predictions** - Tester avec différents profils clients
5. **Feature importance** - Comprendre les facteurs importants

## 🔍 Monitoring et Debugging

### Logs
L'API génère des logs détaillés pour :
- Chargement du modèle
- Validation des données
- Prédictions
- Erreurs et exceptions

### Métriques
Surveillez :
- Temps de réponse des prédictions
- Taux d'erreur de validation
- Distribution des scores de risque
- Utilisation des features

## 🚨 Gestion d'Erreurs

### Fallback Automatique
Si le modèle ML n'est pas disponible, l'API utilise automatiquement un système de règles de base.

### Validation Robuste
- Validation des types de données
- Gestion des valeurs manquantes
- Normalisation des valeurs aberrantes
- Messages d'erreur détaillés

## 🔧 Personnalisation Avancée

### Ajout de Nouvelles Features
1. Modifiez `expected_features` dans `ml_model.py`
2. Ajoutez la logique dans `engineer_features()` de `data_preprocessing.py`
3. Mettez à jour les validateurs si nécessaire

### Modification des Seuils de Risque
Modifiez `_get_risk_assessment()` dans `ml_model.py` :
```python
def _get_risk_assessment(self, risk_score: int) -> Tuple[str, str]:
    if risk_score >= 85:  # Seuil personnalisé
        return "Très faible", "Approuvé automatiquement"
    # ... autres seuils
```

### Intégration Base de Données
Pour sauvegarder les prédictions :
```python
@app.post("/predict")
def predict(data: ScoringRequest):
    # ... logique de prédiction
    
    # Sauvegarder en base
    save_prediction_to_db(data.client_id, response)
    
    return response
```

## 📈 Prochaines Étapes

1. **Entraîner avec vos données réelles**
2. **Optimiser les hyperparamètres**
3. **Ajouter des métriques de monitoring**
4. **Implémenter la validation croisée**
5. **Ajouter des tests automatisés**
6. **Configurer le déploiement en production**

## 🆘 Support

Pour toute question ou problème :
1. Vérifiez les logs de l'application
2. Testez avec `test_api.py`
3. Consultez la documentation FastAPI : `http://localhost:8000/docs`
4. Vérifiez les issues GitHub du projet
