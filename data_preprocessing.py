"""
Data Preprocessing Module for Credit Scoring
Handles feature engineering, data validation, and transformation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessing and feature engineering for credit scoring
    """
    
    def __init__(self):
        """Initialize the data preprocessor"""
        self.feature_validators = self._setup_validators()
        self.feature_ranges = self._setup_feature_ranges()
    
    def _setup_validators(self) -> Dict[str, Dict[str, Any]]:
        """Setup validation rules for each feature"""
        return {
            'age': {'min': 18, 'max': 100, 'type': int},
            'cin': {'min': 10000000, 'max': 99999999, 'type': int},
            'nombre_enfants': {'min': 0, 'max': 20, 'type': int},
            'anciennete_emploi': {'min': 0, 'max': 600, 'type': int},  # months
            'anciennete_logement': {'min': 0, 'max': 600, 'type': int},  # months
            'taux_chomage_sectoriel': {'min': 0.0, 'max': 1.0, 'type': float},
            'revenu_mensuel': {'min': 0.0, 'max': 50000.0, 'type': float},
            'autres_revenus': {'min': 0.0, 'max': 20000.0, 'type': float},
            'revenu_total': {'min': 0.0, 'max': 70000.0, 'type': float},
            'valeur_immobilier': {'min': 0.0, 'max': 2000000.0, 'type': float},
            'valeur_vehicule': {'min': 0.0, 'max': 200000.0, 'type': float},
            'epargne': {'min': 0.0, 'max': 500000.0, 'type': float},
            'patrimoine_total': {'min': 0.0, 'max': 3000000.0, 'type': float},
            'dette_immobiliere': {'min': 0.0, 'max': 1500000.0, 'type': float},
            'dette_auto': {'min': 0.0, 'max': 150000.0, 'type': float},
            'dette_personnelle': {'min': 0.0, 'max': 100000.0, 'type': float},
            'dette_totale': {'min': 0.0, 'max': 1800000.0, 'type': float},
            'ratio_endettement': {'min': 0.0, 'max': 2.0, 'type': float},
            'reste_a_vivre': {'min': 0.0, 'max': 20000.0, 'type': float},
            'capacite_remboursement': {'min': 0.0, 'max': 15000.0, 'type': float},
            'garanties_disponibles': {'min': 0.0, 'max': 500000.0, 'type': float},
            'nombre_credits_anterieurs': {'min': 0, 'max': 50, 'type': int},
            'anciennete_relation_bancaire': {'min': 0, 'max': 600, 'type': int},  # months
            'retard_maximum_jours': {'min': 0, 'max': 365, 'type': int},
            'nombre_incidents_12m': {'min': 0, 'max': 50, 'type': int},
            'nombre_demandes_6m': {'min': 0, 'max': 20, 'type': int},
            'taux_utilisation_credit': {'min': 0.0, 'max': 2.0, 'type': float},
            'regularite_paiements': {'min': 0.0, 'max': 1.0, 'type': float},
            'nombre_rejets_12m': {'min': 0, 'max': 20, 'type': int},
            'score_comportement': {'min': 0.0, 'max': 100.0, 'type': float},
            'montant_demande': {'min': 0.0, 'max': 500000.0, 'type': float},
            'duree_demande': {'min': 1, 'max': 360, 'type': int},  # months
            'mensualite_demandee': {'min': 0.0, 'max': 10000.0, 'type': float},
            'taux_propose': {'min': 0.0, 'max': 0.5, 'type': float},
            'ratio_mensualite_revenu': {'min': 0.0, 'max': 1.0, 'type': float},
            'apport_personnel': {'min': 0.0, 'max': 200000.0, 'type': float},
            'valeur_garanties': {'min': 0.0, 'max': 1000000.0, 'type': float},
            'score_pd': {'min': 0.0, 'max': 1.0, 'type': float},
            'score_lgd': {'min': 0.0, 'max': 1.0, 'type': float},
            'score_ead': {'min': 0.0, 'max': 1.0, 'type': float},
            'perte_attendue': {'min': 0.0, 'max': 1.0, 'type': float}
        }
    
    def _setup_feature_ranges(self) -> Dict[str, Dict[str, float]]:
        """Setup feature ranges for normalization"""
        return {
            'age': {'low': 25, 'high': 55},
            'revenu_mensuel': {'low': 1000, 'high': 5000},
            'ratio_endettement': {'low': 0.2, 'high': 0.4},
            'epargne': {'low': 2000, 'high': 20000},
            'anciennete_emploi': {'low': 12, 'high': 60},
            'score_comportement': {'low': 60, 'high': 90}
        }
    
    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data and return cleaned data
        
        Args:
            data: Raw input data
            
        Returns:
            Validated and cleaned data
        """
        validated_data = {}
        validation_errors = []
        
        for field, value in data.items():
            if field in self.feature_validators:
                validator = self.feature_validators[field]
                
                try:
                    # Type conversion
                    if validator['type'] == int:
                        clean_value = int(float(value)) if value is not None else 0
                    elif validator['type'] == float:
                        clean_value = float(value) if value is not None else 0.0
                    else:
                        clean_value = value
                    
                    # Range validation for numerical fields
                    if validator['type'] in [int, float]:
                        if clean_value < validator['min']:
                            validation_errors.append(f"{field}: value {clean_value} below minimum {validator['min']}")
                            clean_value = validator['min']
                        elif clean_value > validator['max']:
                            validation_errors.append(f"{field}: value {clean_value} above maximum {validator['max']}")
                            clean_value = validator['max']
                    
                    validated_data[field] = clean_value
                    
                except (ValueError, TypeError) as e:
                    validation_errors.append(f"{field}: invalid value {value} - {str(e)}")
                    # Set default value
                    if validator['type'] == int:
                        validated_data[field] = 0
                    elif validator['type'] == float:
                        validated_data[field] = 0.0
                    else:
                        validated_data[field] = None
            else:
                # Keep non-validated fields as is
                validated_data[field] = value
        
        if validation_errors:
            logger.warning(f"Validation errors: {validation_errors}")
        
        return validated_data
    
    def engineer_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create additional features from existing data
        
        Args:
            data: Input data
            
        Returns:
            Data with engineered features
        """
        engineered_data = data.copy()
        
        try:
            # Financial ratios
            if data.get('revenu_mensuel', 0) > 0:
                engineered_data['debt_to_income'] = data.get('dette_totale', 0) / (data.get('revenu_mensuel', 1) * 12)
                engineered_data['savings_to_income'] = data.get('epargne', 0) / (data.get('revenu_mensuel', 1) * 12)
                engineered_data['loan_to_income'] = data.get('montant_demande', 0) / (data.get('revenu_mensuel', 1) * 12)
            
            # Asset ratios
            if data.get('patrimoine_total', 0) > 0:
                engineered_data['debt_to_assets'] = data.get('dette_totale', 0) / data.get('patrimoine_total', 1)
                engineered_data['liquid_assets_ratio'] = data.get('epargne', 0) / data.get('patrimoine_total', 1)
            
            # Employment stability score
            employment_months = data.get('anciennete_emploi', 0)
            if employment_months >= 60:  # 5+ years
                engineered_data['employment_stability'] = 1.0
            elif employment_months >= 24:  # 2+ years
                engineered_data['employment_stability'] = 0.8
            elif employment_months >= 12:  # 1+ year
                engineered_data['employment_stability'] = 0.6
            elif employment_months >= 6:  # 6+ months
                engineered_data['employment_stability'] = 0.4
            else:
                engineered_data['employment_stability'] = 0.2
            
            # Credit history score
            incidents = data.get('nombre_incidents_12m', 0)
            rejections = data.get('nombre_rejets_12m', 0)
            max_delay = data.get('retard_maximum_jours', 0)
            
            credit_history_score = 100
            credit_history_score -= incidents * 10
            credit_history_score -= rejections * 5
            credit_history_score -= min(max_delay / 30, 10) * 5  # Penalty for delays
            
            engineered_data['credit_history_score'] = max(0, credit_history_score)
            
            # Age group categorization
            age = data.get('age', 0)
            if age < 25:
                engineered_data['age_group'] = 'young'
            elif age < 35:
                engineered_data['age_group'] = 'young_adult'
            elif age < 50:
                engineered_data['age_group'] = 'middle_age'
            elif age < 65:
                engineered_data['age_group'] = 'mature'
            else:
                engineered_data['age_group'] = 'senior'
            
            # Loan characteristics
            if data.get('duree_demande', 0) > 0 and data.get('montant_demande', 0) > 0:
                # Monthly payment calculation (simple interest)
                principal = data.get('montant_demande', 0)
                duration = data.get('duree_demande', 1)
                rate = data.get('taux_propose', 0.1) / 12  # Monthly rate
                
                if rate > 0:
                    monthly_payment = principal * (rate * (1 + rate)**duration) / ((1 + rate)**duration - 1)
                else:
                    monthly_payment = principal / duration
                
                engineered_data['calculated_monthly_payment'] = monthly_payment
                
                # Payment burden
                if data.get('revenu_mensuel', 0) > 0:
                    engineered_data['payment_burden'] = monthly_payment / data.get('revenu_mensuel', 1)
            
            # Risk indicators
            risk_flags = 0
            
            # High debt ratio
            if data.get('ratio_endettement', 0) > 0.5:
                risk_flags += 1
            
            # Recent incidents
            if data.get('nombre_incidents_12m', 0) > 0:
                risk_flags += 1
            
            # Low savings
            if data.get('epargne', 0) < data.get('revenu_mensuel', 0):
                risk_flags += 1
            
            # Unstable employment
            if data.get('anciennete_emploi', 0) < 12:
                risk_flags += 1
            
            # Multiple recent applications
            if data.get('nombre_demandes_6m', 0) > 2:
                risk_flags += 1
            
            engineered_data['risk_flags_count'] = risk_flags
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
        
        return engineered_data
    
    def get_feature_importance_weights(self) -> Dict[str, float]:
        """
        Return feature importance weights for scoring
        
        Returns:
            Dictionary of feature weights
        """
        return {
            'revenu_mensuel': 0.15,
            'ratio_endettement': 0.12,
            'nombre_incidents_12m': 0.10,
            'anciennete_emploi': 0.08,
            'epargne': 0.08,
            'score_comportement': 0.07,
            'retard_maximum_jours': 0.06,
            'age': 0.05,
            'regularite_paiements': 0.05,
            'debt_to_income': 0.04,
            'employment_stability': 0.04,
            'credit_history_score': 0.04,
            'payment_burden': 0.03,
            'risk_flags_count': 0.03,
            'autres_revenus': 0.02,
            'valeur_garanties': 0.02,
            'anciennete_relation_bancaire': 0.02
        }
    
    def calculate_weighted_score(self, data: Dict[str, Any]) -> float:
        """
        Calculate a weighted risk score based on feature importance
        
        Args:
            data: Processed data with engineered features
            
        Returns:
            Weighted risk score (0-100)
        """
        weights = self.get_feature_importance_weights()
        total_score = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in data:
                value = data[feature]
                
                # Normalize value to 0-1 scale based on feature type
                if feature in self.feature_ranges:
                    range_info = self.feature_ranges[feature]
                    normalized = min(1.0, max(0.0, (value - range_info['low']) / (range_info['high'] - range_info['low'])))
                else:
                    # Default normalization for other features
                    if feature in ['ratio_endettement', 'debt_to_income', 'payment_burden']:
                        # Lower is better for these features
                        normalized = max(0.0, 1.0 - min(1.0, value))
                    elif feature in ['nombre_incidents_12m', 'retard_maximum_jours', 'risk_flags_count']:
                        # Zero is best for these features
                        normalized = max(0.0, 1.0 - min(1.0, value / 10))
                    else:
                        # Higher is better for most other features
                        normalized = min(1.0, max(0.0, value / 100))
                
                total_score += normalized * weight
                total_weight += weight
        
        # Convert to 0-100 scale
        if total_weight > 0:
            final_score = (total_score / total_weight) * 100
        else:
            final_score = 50  # Default neutral score
        
        return round(final_score, 2)
