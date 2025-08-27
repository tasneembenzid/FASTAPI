"""
Training script for Credit Scoring Gradient Boosting Model
This script demonstrates how to train and save your ML model
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
from datetime import datetime
import logging

# Import our modules
from ml_model import CreditScoringModel
from data_preprocessing import DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic credit scoring data for demonstration
    In production, replace this with your actual data loading
    """
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        # Generate correlated features that make sense for credit scoring
        age = np.random.normal(40, 12)
        age = max(18, min(80, age))
        
        # Income correlated with age and education
        base_income = 1000 + age * 50 + np.random.normal(0, 500)
        revenu_mensuel = max(500, base_income)
        
        # Employment tenure correlated with age
        anciennete_emploi = min(age - 18, np.random.exponential(36))
        
        # Debt ratio - key risk factor
        ratio_endettement = np.random.beta(2, 5)  # Skewed towards lower values
        
        # Savings correlated with income
        epargne = max(0, revenu_mensuel * np.random.uniform(0, 2) + np.random.normal(0, 5000))
        
        # Payment history
        nombre_incidents_12m = np.random.poisson(0.5)
        retard_maximum_jours = np.random.exponential(10) if nombre_incidents_12m > 0 else 0
        
        # Other features
        nombre_enfants = np.random.poisson(1.5)
        anciennete_logement = np.random.exponential(24)
        
        # Calculate derived features
        dette_totale = revenu_mensuel * 12 * ratio_endettement
        reste_a_vivre = revenu_mensuel - (dette_totale / 12)
        
        # Target variable (0 = good, 1 = default)
        # Higher risk if: high debt ratio, low income, many incidents, young age
        risk_score = 0
        risk_score += ratio_endettement * 100
        risk_score += max(0, 3 - revenu_mensuel/1000) * 20
        risk_score += nombre_incidents_12m * 25
        risk_score += max(0, 25 - age) * 2
        risk_score += max(0, retard_maximum_jours/30) * 10
        
        # Add some randomness
        risk_score += np.random.normal(0, 15)
        
        # Convert to binary target (default probability)
        default_prob = 1 / (1 + np.exp(-(risk_score - 50) / 10))
        target = 1 if np.random.random() < default_prob else 0
        
        sample = {
            'client_id': f'CLIENT_{i+1:04d}',
            'cin': np.random.randint(10000000, 99999999),
            'age': int(age),
            'sexe': np.random.choice(['M', 'F']),
            'situation_familiale': np.random.choice(['Célibataire', 'Marié', 'Divorcé']),
            'nombre_enfants': int(nombre_enfants),
            'niveau_education': np.random.choice(['Secondaire', 'Universitaire', 'Post-universitaire']),
            'region': np.random.choice(['Tunis', 'Sfax', 'Sousse', 'Autre']),
            'profession': np.random.choice(['Ingénieur', 'Employé', 'Commerçant', 'Autre']),
            'secteur_activite': np.random.choice(['Technologie', 'Commerce', 'Services', 'Autre']),
            'anciennete_emploi': int(anciennete_emploi),
            'type_contrat': np.random.choice(['CDI', 'CDD', 'Freelance']),
            'type_logement': np.random.choice(['Propriétaire', 'Locataire']),
            'anciennete_logement': int(anciennete_logement),
            'taux_chomage_sectoriel': np.random.uniform(0.05, 0.15),
            'revenu_mensuel': revenu_mensuel,
            'autres_revenus': max(0, np.random.normal(200, 300)),
            'revenu_total': revenu_mensuel + max(0, np.random.normal(200, 300)),
            'valeur_immobilier': max(0, np.random.exponential(100000)),
            'valeur_vehicule': max(0, np.random.exponential(15000)),
            'epargne': epargne,
            'patrimoine_total': epargne + max(0, np.random.exponential(100000)) + max(0, np.random.exponential(15000)),
            'dette_immobiliere': max(0, np.random.exponential(50000)),
            'dette_auto': max(0, np.random.exponential(8000)),
            'dette_personnelle': max(0, np.random.exponential(5000)),
            'dette_totale': dette_totale,
            'ratio_endettement': ratio_endettement,
            'reste_a_vivre': reste_a_vivre,
            'capacite_remboursement': max(0, reste_a_vivre * 0.3),
            'garanties_disponibles': max(0, np.random.exponential(20000)),
            'nombre_credits_anterieurs': np.random.poisson(2),
            'anciennete_relation_bancaire': int(min(anciennete_emploi, np.random.exponential(48))),
            'banque_principale': np.random.choice(['Banque Centrale', 'Banque Populaire', 'BIAT']),
            'retard_maximum_jours': int(retard_maximum_jours),
            'nombre_incidents_12m': nombre_incidents_12m,
            'nombre_demandes_6m': np.random.poisson(1),
            'taux_utilisation_credit': np.random.uniform(0, 1),
            'regularite_paiements': max(0.5, 1 - nombre_incidents_12m * 0.1 + np.random.normal(0, 0.1)),
            'nombre_rejets_12m': np.random.poisson(0.3),
            'score_comportement': max(0, 100 - nombre_incidents_12m * 15 + np.random.normal(0, 10)),
            'montant_demande': max(1000, np.random.exponential(20000)),
            'duree_demande': np.random.choice([12, 24, 36, 48, 60]),
            'type_credit': np.random.choice(['Personnel', 'Auto', 'Immobilier']),
            'mensualite_demandee': max(100, np.random.exponential(500)),
            'taux_propose': np.random.uniform(0.05, 0.20),
            'ratio_mensualite_revenu': np.random.uniform(0.1, 0.4),
            'apport_personnel': max(0, np.random.exponential(10000)),
            'valeur_garanties': max(0, np.random.exponential(30000)),
            'classe_risque': np.random.choice(['A', 'B', 'C', 'D']),
            'score_pd': np.random.uniform(0, 0.5),
            'score_lgd': np.random.uniform(0.2, 0.8),
            'score_ead': np.random.uniform(0.5, 1.0),
            'perte_attendue': np.random.uniform(0, 0.2),
            'target': target  # This is our target variable
        }
        
        data.append(sample)
    
    return pd.DataFrame(data)

def train_gradient_boosting_model():
    """
    Train a Gradient Boosting model for credit scoring
    """
    logger.info("Starting model training...")
    
    # Generate or load your data
    logger.info("Generating synthetic training data...")
    df = generate_synthetic_data(n_samples=2000)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Prepare features and target
    target = df['target']
    features_df = df.drop(['target', 'client_id'], axis=1)
    
    # Process each row through the preprocessor
    processed_data = []
    for _, row in features_df.iterrows():
        row_dict = row.to_dict()
        validated_data = preprocessor.validate_data(row_dict)
        engineered_data = preprocessor.engineer_features(validated_data)
        processed_data.append(engineered_data)
    
    processed_df = pd.DataFrame(processed_data)
    
    # Select numerical features for the model
    numerical_features = [col for col in processed_df.columns 
                         if processed_df[col].dtype in ['int64', 'float64']]
    
    X = processed_df[numerical_features].fillna(0)
    y = target
    
    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Features used: {list(X.columns)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Gradient Boosting model with hyperparameter tuning
    logger.info("Training Gradient Boosting model...")
    
    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [4, 6, 8],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10]
    }
    
    # Create base model
    gb_model = GradientBoostingClassifier(random_state=42)
    
    # Perform grid search
    logger.info("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        gb_model, param_grid, cv=5, scoring='roc_auc', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    test_auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"Test AUC: {test_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save the model
    model_manager = CreditScoringModel()
    model_manager.model = best_model
    model_manager.scaler = scaler
    model_manager.feature_names = list(X.columns)
    model_manager.is_trained = True
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_path = 'models/credit_scoring_model.joblib'
    model_manager.save_model(model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    return model_manager, test_auc

if __name__ == "__main__":
    try:
        model_manager, test_auc = train_gradient_boosting_model()
        print(f"\n✅ Model training completed successfully!")
        print(f"📊 Test AUC Score: {test_auc:.4f}")
        print(f"💾 Model saved to: models/credit_scoring_model.joblib")
        print(f"🚀 You can now restart your FastAPI server to use the trained model")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        print(f"❌ Training failed: {str(e)}")
