# Guide de Configuration Postman pour l'API Credit Scoring

## 🚀 Configuration Initiale

### 1. Démarrer le Serveur FastAPI
```bash
cd c:\Users\USER\Desktop\fastapi
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Le serveur sera accessible sur : `http://localhost:8000`

### 2. Vérifier que l'API fonctionne
Ouvrez votre navigateur et allez sur : `http://localhost:8000/docs`
Vous devriez voir la documentation automatique de FastAPI (Swagger UI).

## 📦 Importer la Collection Postman

### Option 1: Importer les fichiers JSON
1. Ouvrez Postman
2. Cliquez sur "Import" en haut à gauche
3. Sélectionnez les fichiers :
   - `Credit_Scoring_API.postman_collection.json`
   - `Credit_Scoring_Environment.postman_environment.json`
4. Cliquez sur "Import"

### Option 2: Configuration Manuelle

#### Créer un nouvel Environment
1. Cliquez sur l'icône "Environment" (engrenage) en haut à droite
2. Cliquez sur "Add"
3. Nommez l'environment : "Credit Scoring"
4. Ajoutez la variable :
   - **Variable** : `base_url`
   - **Initial Value** : `http://localhost:8000`
   - **Current Value** : `http://localhost:8000`
5. Sauvegardez

## 🧪 Tests Disponibles

### 1. Health Check (GET /)
- **URL** : `{{base_url}}/`
- **Méthode** : GET
- **Description** : Vérifier que l'API fonctionne

### 2. Health Check Détaillé (GET /health)
- **URL** : `{{base_url}}/health`
- **Méthode** : GET
- **Description** : Informations détaillées sur l'état de l'API

### 3. Prédiction de Score de Crédit (POST /predict)
- **URL** : `{{base_url}}/predict`
- **Méthode** : POST
- **Headers** : 
  - `Content-Type: application/json`

## 📋 Exemples de Données de Test

### Bon Client (Faible Risque)
```json
{
  "client_id": "CLIENT_001",
  "cin": 12345678,
  "age": 35,
  "sexe": "M",
  "situation_familiale": "Marié",
  "nombre_enfants": 2,
  "niveau_education": "Universitaire",
  "region": "Tunis",
  "profession": "Ingénieur",
  "secteur_activite": "Technologie",
  "anciennete_emploi": 36,
  "type_contrat": "CDI",
  "type_logement": "Propriétaire",
  "anciennete_logement": 60,
  "taux_chomage_sectoriel": 0.05,
  "revenu_mensuel": 4500.0,
  "autres_revenus": 500.0,
  "revenu_total": 5000.0,
  "valeur_immobilier": 150000.0,
  "valeur_vehicule": 25000.0,
  "epargne": 15000.0,
  "patrimoine_total": 190000.0,
  "dette_immobiliere": 80000.0,
  "dette_auto": 5000.0,
  "dette_personnelle": 2000.0,
  "dette_totale": 87000.0,
  "ratio_endettement": 0.25,
  "reste_a_vivre": 2750.0,
  "capacite_remboursement": 1250.0,
  "garanties_disponibles": 50000.0,
  "nombre_credits_anterieurs": 2,
  "anciennete_relation_bancaire": 72,
  "banque_principale": "Banque Centrale",
  "retard_maximum_jours": 0,
  "nombre_incidents_12m": 0,
  "nombre_demandes_6m": 1,
  "taux_utilisation_credit": 0.3,
  "regularite_paiements": 0.98,
  "nombre_rejets_12m": 0,
  "score_comportement": 85.0,
  "montant_demande": 20000.0,
  "duree_demande": 60,
  "type_credit": "Personnel",
  "mensualite_demandee": 400.0,
  "taux_propose": 0.08,
  "ratio_mensualite_revenu": 0.08,
  "apport_personnel": 5000.0,
  "valeur_garanties": 25000.0,
  "classe_risque": "A",
  "score_pd": 0.02,
  "score_lgd": 0.3,
  "score_ead": 0.8,
  "perte_attendue": 0.0048
}
```

### Client Risqué (Risque Élevé)
```json
{
  "client_id": "CLIENT_002",
  "cin": 87654321,
  "age": 28,
  "sexe": "F",
  "situation_familiale": "Célibataire",
  "nombre_enfants": 0,
  "niveau_education": "Secondaire",
  "region": "Sfax",
  "profession": "Employé",
  "secteur_activite": "Commerce",
  "anciennete_emploi": 12,
  "type_contrat": "CDD",
  "type_logement": "Locataire",
  "anciennete_logement": 24,
  "taux_chomage_sectoriel": 0.12,
  "revenu_mensuel": 1200.0,
  "autres_revenus": 0.0,
  "revenu_total": 1200.0,
  "valeur_immobilier": 0.0,
  "valeur_vehicule": 8000.0,
  "epargne": 500.0,
  "patrimoine_total": 8500.0,
  "dette_immobiliere": 0.0,
  "dette_auto": 6000.0,
  "dette_personnelle": 3000.0,
  "dette_totale": 9000.0,
  "ratio_endettement": 0.75,
  "reste_a_vivre": 300.0,
  "capacite_remboursement": 200.0,
  "garanties_disponibles": 2000.0,
  "nombre_credits_anterieurs": 3,
  "anciennete_relation_bancaire": 18,
  "banque_principale": "Banque Populaire",
  "retard_maximum_jours": 45,
  "nombre_incidents_12m": 3,
  "nombre_demandes_6m": 4,
  "taux_utilisation_credit": 0.9,
  "regularite_paiements": 0.65,
  "nombre_rejets_12m": 2,
  "score_comportement": 35.0,
  "montant_demande": 5000.0,
  "duree_demande": 36,
  "type_credit": "Personnel",
  "mensualite_demandee": 180.0,
  "taux_propose": 0.15,
  "ratio_mensualite_revenu": 0.15,
  "apport_personnel": 0.0,
  "valeur_garanties": 1000.0,
  "classe_risque": "D",
  "score_pd": 0.25,
  "score_lgd": 0.7,
  "score_ead": 0.95,
  "perte_attendue": 0.166
}
```

## 📊 Réponses Attendues

### Réponse pour Bon Client
```json
{
  "client_id": "CLIENT_001",
  "risk_score": 100,
  "risk_class": "Faible",
  "decision": "Approuvé",
  "factors_analyzed": {
    "revenu_mensuel": 4500.0,
    "ratio_endettement": 0.25,
    "anciennete_emploi": 36,
    "nombre_incidents_12m": 0,
    "epargne": 15000.0
  },
  "timestamp": "2025-08-21T00:00:00Z"
}
```

### Réponse pour Client Risqué
```json
{
  "client_id": "CLIENT_002",
  "risk_score": 0,
  "risk_class": "Très élevé",
  "decision": "Refusé",
  "factors_analyzed": {
    "revenu_mensuel": 1200.0,
    "ratio_endettement": 0.75,
    "anciennete_emploi": 12,
    "nombre_incidents_12m": 3,
    "epargne": 500.0
  },
  "timestamp": "2025-08-21T00:00:00Z"
}
```

## 🔧 Dépannage

### Problème : "Could not get response"
- Vérifiez que le serveur FastAPI est en cours d'exécution
- Vérifiez l'URL : `http://localhost:8000`
- Assurez-vous que le port 8000 n'est pas bloqué

### Problème : Erreur 422 (Validation Error)
- Vérifiez que tous les champs requis sont présents dans votre JSON
- Vérifiez les types de données (int, float, string)
- Utilisez les exemples fournis comme référence

### Problème : Erreur 500 (Internal Server Error)
- Vérifiez les logs du serveur dans le terminal
- Assurez-vous que toutes les dépendances sont installées

## 📚 Ressources Supplémentaires

- **Documentation FastAPI** : `http://localhost:8000/docs`
- **Documentation Alternative** : `http://localhost:8000/redoc`
- **Test Script Python** : Exécutez `python test_api.py` pour tester l'API

## 🎯 Prochaines Étapes

1. Testez tous les endpoints avec Postman
2. Modifiez les données de test pour explorer différents scénarios
3. Intégrez votre modèle ML réel dans l'endpoint `/predict`
4. Ajoutez des tests automatisés avec Postman
5. Configurez l'authentification si nécessaire
