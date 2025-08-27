from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from datetime import datetime
from typing import Optional

# Import our ML modules
from ml_model import CreditScoringModel
from data_preprocessing import DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Scoring API with ML",
    description="API pour l'évaluation du risque de crédit avec modèle Gradient Boosting",
    version="2.0.0"
)

# Add CORS middleware to allow requests from Postman and other clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML components
try:
    # Try to load existing model, otherwise create default
    model_manager = CreditScoringModel(model_path="models/credit_scoring_model.joblib")
    data_preprocessor = DataPreprocessor()
    logger.info("ML components initialized successfully")
except Exception as e:
    logger.error(f"Error initializing ML components: {str(e)}")
    model_manager = None
    data_preprocessor = None

class ScoringRequest(BaseModel):
    client_id: object
    cin: int
    age: int
    sexe: object
    situation_familiale: object
    nombre_enfants: int
    niveau_education: object
    region: object
    profession: object
    secteur_activite: object
    anciennete_emploi: int
    type_contrat: object
    type_logement: object
    anciennete_logement: int
    taux_chomage_sectoriel: float
    revenu_mensuel: float
    autres_revenus: float
    revenu_total: float
    valeur_immobilier: float
    valeur_vehicule: float
    epargne: float
    patrimoine_total: float
    dette_immobiliere: float
    dette_auto: float
    dette_personnelle: float
    dette_totale: float
    ratio_endettement: float
    reste_a_vivre: float
    capacite_remboursement: float
    garanties_disponibles: float
    nombre_credits_anterieurs: int
    anciennete_relation_bancaire: int
    banque_principale: object
    retard_maximum_jours: int
    nombre_incidents_12m: int
    nombre_demandes_6m: int
    taux_utilisation_credit: float
    regularite_paiements: float
    nombre_rejets_12m: int
    score_comportement: float
    montant_demande: float
    duree_demande: int
    type_credit: object
    mensualite_demandee: float
    taux_propose: float
    ratio_mensualite_revenu: float
    apport_personnel: float
    valeur_garanties: float
    classe_risque: object
    score_pd: float
    score_lgd: float
    score_ead: float
    perte_attendue: float


@app.get("/")
def read_root():
    """Health check endpoint"""
    return {"message": "Credit Scoring API is running", "status": "healthy"}

@app.get("/health")
def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "service": "Credit Scoring API",
        "version": "1.0.0"
    }

@app.post("/predict")
def predict(data: ScoringRequest):
    """
    Endpoint pour prédire le score de crédit avec modèle ML

    Reçoit les données d'un client et retourne une évaluation de risque
    utilisant un modèle Gradient Boosting
    """
    try:
        logger.info(f"Received prediction request for client: {data.client_id}")

        # Convert Pydantic model to dictionary
        input_data = data.model_dump()

        # Check if ML components are available
        if model_manager is None or data_preprocessor is None:
            raise HTTPException(
                status_code=500,
                detail="ML components not available. Please check server configuration."
            )

        # Validate and preprocess data
        validated_data = data_preprocessor.validate_data(input_data)
        engineered_data = data_preprocessor.engineer_features(validated_data)

        # Get ML model prediction
        ml_prediction = model_manager.predict(engineered_data)

        # Calculate additional weighted score for comparison
        weighted_score = data_preprocessor.calculate_weighted_score(engineered_data)

        # Prepare response with comprehensive analysis
        response = {
            "client_id": data.client_id,
            "risk_score": ml_prediction["risk_score"],
            "risk_class": ml_prediction["risk_class"],
            "decision": ml_prediction["decision"],
            "confidence": ml_prediction.get("confidence", 0.0),
            "model_info": {
                "model_version": ml_prediction.get("model_version", "unknown"),
                "prediction_method": "gradient_boosting" if model_manager.is_trained else "rule_based",
                "weighted_score": weighted_score
            },
            "factors_analyzed": {
                "revenu_mensuel": validated_data.get("revenu_mensuel", 0),
                "ratio_endettement": validated_data.get("ratio_endettement", 0),
                "anciennete_emploi": validated_data.get("anciennete_emploi", 0),
                "nombre_incidents_12m": validated_data.get("nombre_incidents_12m", 0),
                "epargne": validated_data.get("epargne", 0),
                "employment_stability": engineered_data.get("employment_stability", 0),
                "credit_history_score": engineered_data.get("credit_history_score", 0),
                "debt_to_income": engineered_data.get("debt_to_income", 0),
                "risk_flags_count": engineered_data.get("risk_flags_count", 0)
            },
            "engineered_features": {
                "debt_to_income": engineered_data.get("debt_to_income", 0),
                "savings_to_income": engineered_data.get("savings_to_income", 0),
                "employment_stability": engineered_data.get("employment_stability", 0),
                "credit_history_score": engineered_data.get("credit_history_score", 0),
                "payment_burden": engineered_data.get("payment_burden", 0),
                "age_group": engineered_data.get("age_group", "unknown")
            },
            "timestamp": ml_prediction.get("prediction_timestamp", datetime.now().isoformat())
        }

        logger.info(f"Prediction completed for client {data.client_id}: {ml_prediction['risk_class']}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during prediction: {str(e)}"
        )

@app.get("/model/info")
def get_model_info():
    """
    Get information about the current ML model
    """
    try:
        if model_manager is None:
            raise HTTPException(status_code=500, detail="Model manager not available")

        model_info = model_manager.get_model_info()
        return {
            "model_info": model_info,
            "preprocessor_available": data_preprocessor is not None,
            "api_version": "2.0.0",
            "status": "operational"
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/validate")
def validate_input_data(data: ScoringRequest):
    """
    Validate input data without making a prediction
    """
    try:
        if data_preprocessor is None:
            raise HTTPException(status_code=500, detail="Data preprocessor not available")

        input_data = data.model_dump()
        validated_data = data_preprocessor.validate_data(input_data)
        engineered_data = data_preprocessor.engineer_features(validated_data)

        return {
            "validation_status": "success",
            "validated_data": validated_data,
            "engineered_features": {
                "debt_to_income": engineered_data.get("debt_to_income", 0),
                "savings_to_income": engineered_data.get("savings_to_income", 0),
                "employment_stability": engineered_data.get("employment_stability", 0),
                "credit_history_score": engineered_data.get("credit_history_score", 0),
                "payment_burden": engineered_data.get("payment_burden", 0),
                "age_group": engineered_data.get("age_group", "unknown"),
                "risk_flags_count": engineered_data.get("risk_flags_count", 0)
            },
            "feature_count": len(engineered_data),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")

@app.get("/features/importance")
def get_feature_importance():
    """
    Get feature importance weights used in the model
    """
    try:
        if data_preprocessor is None:
            raise HTTPException(status_code=500, detail="Data preprocessor not available")

        weights = data_preprocessor.get_feature_importance_weights()

        # Sort by importance
        sorted_features = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        return {
            "feature_weights": weights,
            "sorted_features": sorted_features,
            "total_features": len(weights),
            "description": "Feature importance weights used in credit scoring model"
        }
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
