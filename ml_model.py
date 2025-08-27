"""
ML Model Management Module for Credit Scoring
Handles model loading, prediction, and model lifecycle management
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditScoringModel:
    """
    Credit Scoring Model Manager
    Handles model loading, preprocessing, and predictions
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Credit Scoring Model
        
        Args:
            model_path: Path to the saved model file
        """
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.model_path = model_path
        self.is_trained = False
        
        # Define expected features for the model
        self.expected_features = [
            'age', 'nombre_enfants', 'anciennete_emploi', 'anciennete_logement',
            'taux_chomage_sectoriel', 'revenu_mensuel', 'autres_revenus', 'revenu_total',
            'valeur_immobilier', 'valeur_vehicule', 'epargne', 'patrimoine_total',
            'dette_immobiliere', 'dette_auto', 'dette_personnelle', 'dette_totale',
            'ratio_endettement', 'reste_a_vivre', 'capacite_remboursement',
            'garanties_disponibles', 'nombre_credits_anterieurs', 'anciennete_relation_bancaire',
            'retard_maximum_jours', 'nombre_incidents_12m', 'nombre_demandes_6m',
            'taux_utilisation_credit', 'regularite_paiements', 'nombre_rejets_12m',
            'score_comportement', 'montant_demande', 'duree_demande',
            'mensualite_demandee', 'taux_propose', 'ratio_mensualite_revenu',
            'apport_personnel', 'valeur_garanties', 'score_pd', 'score_lgd',
            'score_ead', 'perte_attendue'
        ]
        
        # Categorical features that need encoding
        self.categorical_features = [
            'sexe', 'situation_familiale', 'niveau_education', 'region',
            'profession', 'secteur_activite', 'type_contrat', 'type_logement',
            'banque_principale', 'type_credit', 'classe_risque'
        ]
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.info("No model path provided or file doesn't exist. Creating new model.")
            self._create_default_model()
    
    def _create_default_model(self):
        """Create a default Gradient Boosting model for demonstration"""
        logger.info("Creating default Gradient Boosting model...")
        
        # Create a default model with good parameters for credit scoring
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
        
        # Create default scaler and encoders
        self.scaler = StandardScaler()
        self.label_encoders = {feature: LabelEncoder() for feature in self.categorical_features}
        
        # For demo purposes, fit encoders with common values
        self._fit_default_encoders()
        
        logger.info("Default model created successfully")
    
    def _fit_default_encoders(self):
        """Fit label encoders with common categorical values"""
        default_categories = {
            'sexe': ['M', 'F'],
            'situation_familiale': ['Célibataire', 'Marié', 'Divorcé', 'Veuf'],
            'niveau_education': ['Primaire', 'Secondaire', 'Universitaire', 'Post-universitaire'],
            'region': ['Tunis', 'Sfax', 'Sousse', 'Kairouan', 'Bizerte', 'Gabès', 'Autre'],
            'profession': ['Ingénieur', 'Médecin', 'Enseignant', 'Employé', 'Ouvrier', 'Commerçant', 'Autre'],
            'secteur_activite': ['Technologie', 'Santé', 'Education', 'Commerce', 'Industrie', 'Services', 'Autre'],
            'type_contrat': ['CDI', 'CDD', 'Freelance', 'Fonctionnaire'],
            'type_logement': ['Propriétaire', 'Locataire', 'Famille'],
            'banque_principale': ['Banque Centrale', 'Banque Populaire', 'BIAT', 'STB', 'Autre'],
            'type_credit': ['Personnel', 'Immobilier', 'Auto', 'Consommation'],
            'classe_risque': ['A', 'B', 'C', 'D']
        }
        
        for feature, categories in default_categories.items():
            if feature in self.label_encoders:
                self.label_encoders[feature].fit(categories)
    
    def preprocess_data(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess input data for model prediction
        
        Args:
            data: Dictionary containing client data
            
        Returns:
            Preprocessed feature array
        """
        try:
            # Create DataFrame from input data
            df = pd.DataFrame([data])
            
            # Handle categorical features
            for feature in self.categorical_features:
                if feature in df.columns:
                    # Handle unknown categories
                    try:
                        df[feature] = self.label_encoders[feature].transform(df[feature])
                    except ValueError:
                        # If category is unknown, assign the most common category (0)
                        logger.warning(f"Unknown category for {feature}: {df[feature].iloc[0]}")
                        df[feature] = 0
            
            # Select only numerical features for the model
            numerical_features = [col for col in self.expected_features if col in df.columns]
            X = df[numerical_features].values
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0)
            
            # Scale features if scaler is fitted
            if hasattr(self.scaler, 'mean_'):
                X = self.scaler.transform(X)
            
            return X
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for credit scoring
        
        Args:
            data: Dictionary containing client data
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess the data
            X = self.preprocess_data(data)
            
            # For demonstration, if model is not trained, use rule-based scoring
            if not self.is_trained:
                return self._rule_based_prediction(data)
            
            # Make prediction with the trained model
            prediction_proba = self.model.predict_proba(X)[0]

            # Convert to risk score (0-100)
            risk_score = int((1 - prediction_proba[1]) * 100)  # Assuming class 1 is default
            
            # Determine risk class and decision
            risk_class, decision = self._get_risk_assessment(risk_score)
            
            return {
                "risk_score": risk_score,
                "risk_class": risk_class,
                "decision": decision,
                "confidence": float(max(prediction_proba)),
                "model_version": "gradient_boost_v1.0",
                "prediction_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            # Fallback to rule-based prediction
            return self._rule_based_prediction(data)
    
    def _rule_based_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback rule-based prediction when ML model is not available
        
        Args:
            data: Dictionary containing client data
            
        Returns:
            Dictionary containing prediction results
        """
        risk_score = 50  # Start with neutral score
        
        # Income factors
        if data.get('revenu_mensuel', 0) > 3000:
            risk_score += 15
        elif data.get('revenu_mensuel', 0) < 1000:
            risk_score -= 20
        
        # Employment stability
        if data.get('anciennete_emploi', 0) > 24:  # More than 2 years
            risk_score += 10
        elif data.get('anciennete_emploi', 0) < 6:  # Less than 6 months
            risk_score -= 15
        
        # Debt ratio
        debt_ratio = data.get('ratio_endettement', 0)
        if debt_ratio < 0.3:
            risk_score += 20
        elif debt_ratio > 0.5:
            risk_score -= 25
        
        # Savings
        if data.get('epargne', 0) > 10000:
            risk_score += 15
        elif data.get('epargne', 0) < 1000:
            risk_score -= 10
        
        # Payment history
        incidents = data.get('nombre_incidents_12m', 0)
        if incidents == 0:
            risk_score += 20
        elif incidents > 2:
            risk_score -= 30
        
        # Late payments
        if data.get('retard_maximum_jours', 0) > 30:
            risk_score -= 20
        
        # Age factor
        age = data.get('age', 0)
        if 25 <= age <= 55:
            risk_score += 5
        
        # Normalize score to 0-100 range
        risk_score = max(0, min(100, risk_score))
        
        # Determine risk class and decision
        risk_class, decision = self._get_risk_assessment(risk_score)
        
        return {
            "risk_score": risk_score,
            "risk_class": risk_class,
            "decision": decision,
            "confidence": 0.75,  # Rule-based confidence
            "model_version": "rule_based_v1.0",
            "prediction_timestamp": datetime.now().isoformat()
        }
    
    def _get_risk_assessment(self, risk_score: int) -> Tuple[str, str]:
        """
        Convert risk score to risk class and decision
        
        Args:
            risk_score: Risk score (0-100)
            
        Returns:
            Tuple of (risk_class, decision)
        """
        if risk_score >= 80:
            return "Faible", "Approuvé"
        elif risk_score >= 60:
            return "Moyen", "Approuvé avec conditions"
        elif risk_score >= 40:
            return "Élevé", "Examen manuel requis"
        else:
            return "Très élevé", "Refusé"
    
    def save_model(self, path: str):
        """Save the trained model and preprocessors"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str):
        """Load a trained model and preprocessors"""
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data.get('feature_names', [])
            self.is_trained = model_data.get('is_trained', True)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        model_type = "None"
        model_params = {}

        if self.model is not None:
            model_type = type(self.model).__name__
            try:
                # Get basic model parameters (not fitted attributes)
                basic_params = {
                    'n_estimators': getattr(self.model, 'n_estimators', None),
                    'learning_rate': getattr(self.model, 'learning_rate', None),
                    'max_depth': getattr(self.model, 'max_depth', None),
                    'random_state': getattr(self.model, 'random_state', None)
                }
                model_params = {k: v for k, v in basic_params.items() if v is not None}
            except Exception as e:
                model_params = {"error": f"Could not retrieve model parameters: {str(e)}"}

        return {
            "model_type": model_type,
            "is_trained": self.is_trained,
            "feature_count": len(self.expected_features),
            "categorical_features": len(self.categorical_features),
            "model_path": self.model_path,
            "model_parameters": model_params,
            "last_updated": datetime.now().isoformat()
        }
