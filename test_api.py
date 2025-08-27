import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_detailed_health():
    """Test the detailed health endpoint"""
    print("🔍 Testing detailed health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_credit_scoring_good_client():
    """Test credit scoring with a good client profile"""
    print("🔍 Testing credit scoring - Good Client...")
    
    good_client_data = {
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
    
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        json=good_client_data
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_credit_scoring_risky_client():
    """Test credit scoring with a risky client profile"""
    print("🔍 Testing credit scoring - Risky Client...")
    
    risky_client_data = {
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
    
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        json=risky_client_data
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_model_info():
    """Test the model info endpoint"""
    print("🔍 Testing model info...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_data_validation():
    """Test the data validation endpoint"""
    print("🔍 Testing data validation...")

    good_client_data = {
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

    response = requests.post(
        f"{BASE_URL}/model/validate",
        headers={"Content-Type": "application/json"},
        json=good_client_data
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_feature_importance():
    """Test the feature importance endpoint"""
    print("🔍 Testing feature importance...")
    response = requests.get(f"{BASE_URL}/features/importance")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("🚀 Testing Credit Scoring API with ML")
    print("=" * 50)

    try:
        test_health_check()
        test_detailed_health()
        test_model_info()
        test_feature_importance()
        test_data_validation()
        test_credit_scoring_good_client()
        test_credit_scoring_risky_client()
        print("All tests completed!")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Error: {e}")
