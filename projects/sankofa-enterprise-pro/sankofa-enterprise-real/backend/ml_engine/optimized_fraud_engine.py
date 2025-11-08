#!/usr/bin/env python3
"""
Motor de Fraude Otimizado - Versão Corrigida
Sankofa Enterprise Pro - Optimized Fraud Detection Engine
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, IsolationForest
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, GridSearchCV, 
    train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import joblib

logger = logging.getLogger(__name__)

@dataclass
class FraudPrediction:
    """Resultado de predição de fraude"""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    confidence: float
    explanation: Dict[str, Any]
    processing_time_ms: float
    model_version: str
    timestamp: str

@dataclass
class ModelPerformance:
    """Métricas de performance do modelo"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: List[List[int]]
    feature_importance: Dict[str, float]
    cross_val_scores: List[float]
    training_time_ms: float
    prediction_time_ms: float

class OptimizedFraudEngine:
    """Motor de Detecção de Fraude Otimizado"""
    
    def __init__(self, model_dir: str = "models", config_file: str = None):
        self.model_dir = model_dir
        self.config_file = config_file
        
        # Configurações otimizadas
        self.config = {
            'ensemble_models': {
                'random_forest': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'class_weight': 'balanced',
                    'random_state': 42
                },
                'gradient_boosting': {
                    'n_estimators': 150,
                    'learning_rate': 0.1,
                    'max_depth': 8,
                    'subsample': 0.8,
                    'random_state': 42
                },
                'extra_trees': {
                    'n_estimators': 100,
                    'max_depth': 12,
                    'min_samples_split': 8,
                    'class_weight': 'balanced',
                    'random_state': 42
                },
                'logistic_regression': {
                    'C': 1.0,
                    'class_weight': 'balanced',
                    'max_iter': 1000,
                    'random_state': 42
                },
                'neural_network': {
                    'hidden_layer_sizes': (100, 50),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.001,
                    'max_iter': 500,
                    'random_state': 42
                }
            },
            'feature_selection': {
                'method': 'rfe',  # 'kbest', 'rfe', 'importance'
                'k_features': 20,
                'selection_threshold': 0.01
            },
            'preprocessing': {
                'scaler': 'robust',  # 'standard', 'robust'
                'handle_outliers': True,
                'outlier_threshold': 3.0
            },
            'ensemble_voting': {
                'method': 'soft',  # 'hard', 'soft'
                'weights': [0.25, 0.25, 0.2, 0.15, 0.15]  # Para cada modelo
            },
            'risk_thresholds': {
                'low': 0.2,
                'medium': 0.5,
                'high': 0.8,
                'critical': 0.95
            },
            'calibration': {
                'enabled': True,
                'method': 'isotonic',  # 'sigmoid', 'isotonic'
                'cv_folds': 3
            },
            'cross_validation': {
                'folds': 5,
                'stratified': True,
                'shuffle': True,
                'random_state': 42
            }
        }
        
        # Carregar configuração personalizada se fornecida
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
                self.config.update(custom_config)
        
        # Inicializar componentes
        self.models = {}
        self.ensemble_model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_names = []
        self.is_trained = False
        self.model_version = f"optimized_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Métricas de performance
        self.performance_metrics = {}
        self.training_history = []
        
        # Criar diretório de modelos
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info("🚀 Motor de Fraude Otimizado inicializado")
        logger.info(f"📁 Diretório de modelos: {self.model_dir}")
        logger.info(f"🔧 Modelos no ensemble: {len(self.config['ensemble_models'])}")
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Inicializa modelos do ensemble com configurações otimizadas"""
        models = {}
        
        # Random Forest
        models['random_forest'] = RandomForestClassifier(
            **self.config['ensemble_models']['random_forest']
        )
        
        # Gradient Boosting
        models['gradient_boosting'] = GradientBoostingClassifier(
            **self.config['ensemble_models']['gradient_boosting']
        )
        
        # Extra Trees
        models['extra_trees'] = ExtraTreesClassifier(
            **self.config['ensemble_models']['extra_trees']
        )
        
        # Logistic Regression
        models['logistic_regression'] = LogisticRegression(
            **self.config['ensemble_models']['logistic_regression']
        )
        
        # Neural Network
        models['neural_network'] = MLPClassifier(
            **self.config['ensemble_models']['neural_network']
        )
        
        logger.info(f"🤖 {len(models)} modelos inicializados no ensemble")
        return models
    
    def _preprocess_data(self, X: pd.DataFrame, fit_transform: bool = False) -> np.ndarray:
        """Pré-processa os dados com técnicas otimizadas"""
        X_processed = X.copy()
        
        # Tratar valores ausentes
        X_processed = X_processed.fillna(X_processed.median())
        
        # Tratar outliers se configurado
        if self.config['preprocessing']['handle_outliers']:
            threshold = self.config['preprocessing']['outlier_threshold']
            
            for column in X_processed.select_dtypes(include=[np.number]).columns:
                Q1 = X_processed[column].quantile(0.25)
                Q3 = X_processed[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                X_processed[column] = X_processed[column].clip(lower_bound, upper_bound)
        
        # Aplicar scaling
        if fit_transform:
            if self.config['preprocessing']['scaler'] == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
            
            X_scaled = self.scaler.fit_transform(X_processed)
        else:
            if self.scaler is None:
                raise ValueError("Scaler não foi treinado. Execute fit primeiro.")
            X_scaled = self.scaler.transform(X_processed)
        
        return X_scaled
    
    def _select_features(self, X: np.ndarray, y: np.ndarray, fit_transform: bool = False) -> np.ndarray:
        """Seleciona features mais importantes"""
        if fit_transform:
            method = self.config['feature_selection']['method']
            k_features = self.config['feature_selection']['k_features']
            
            if method == 'kbest':
                self.feature_selector = SelectKBest(
                    score_func=f_classif,
                    k=min(k_features, X.shape[1])
                )
            elif method == 'rfe':
                # Usar Random Forest para RFE por ser mais rápido
                base_estimator = RandomForestClassifier(
                    n_estimators=50, random_state=42, class_weight='balanced'
                )
                self.feature_selector = RFE(
                    estimator=base_estimator,
                    n_features_to_select=min(k_features, X.shape[1])
                )
            
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Salvar nomes das features selecionadas
            if hasattr(self.feature_selector, 'get_support'):
                selected_indices = self.feature_selector.get_support(indices=True)
                if len(self.feature_names) > 0:
                    self.selected_features = [self.feature_names[i] for i in selected_indices]
                else:
                    self.selected_features = [f'feature_{i}' for i in selected_indices]
            
            logger.info(f"🎯 {X_selected.shape[1]} features selecionadas de {X.shape[1]}")
            
        else:
            if self.feature_selector is None:
                raise ValueError("Feature selector não foi treinado. Execute fit primeiro.")
            X_selected = self.feature_selector.transform(X)
        
        return X_selected
    
    def _train_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Treina ensemble de modelos com validação cruzada"""
        training_start = time.time()
        
        # Inicializar modelos
        self.models = self._initialize_models()
        
        # Configurar cross-validation
        cv_config = self.config['cross_validation']
        if cv_config['stratified']:
            cv = StratifiedKFold(
                n_splits=cv_config['folds'],
                shuffle=cv_config['shuffle'],
                random_state=cv_config['random_state']
            )
        else:
            cv = cv_config['folds']
        
        # Treinar cada modelo e avaliar
        model_performances = {}
        trained_models = {}
        
        for name, model in self.models.items():
            logger.info(f"🔄 Treinando modelo: {name}")
            
            model_start = time.time()
            
            # Treinar modelo
            model.fit(X, y)
            
            # Validação cruzada
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
            
            # Predições para métricas detalhadas
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Calcular métricas
            performance = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1_score': f1_score(y, y_pred, zero_division=0),
                'auc_roc': roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.5,
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time_ms': (time.time() - model_start) * 1000
            }
            
            model_performances[name] = performance
            trained_models[name] = model
            
            logger.info(f"✅ {name}: F1={performance['f1_score']:.3f}, "
                       f"CV={performance['cv_mean']:.3f}±{performance['cv_std']:.3f}")
        
        # Aplicar calibração se configurado
        if self.config['calibration']['enabled']:
            logger.info("🎯 Aplicando calibração de probabilidades")
            calibrated_models = {}
            
            for name, model in trained_models.items():
                if hasattr(model, 'predict_proba'):
                    calibrated_model = CalibratedClassifierCV(
                        model,
                        method=self.config['calibration']['method'],
                        cv=self.config['calibration']['cv_folds']
                    )
                    calibrated_model.fit(X, y)
                    calibrated_models[name] = calibrated_model
                else:
                    calibrated_models[name] = model
            
            trained_models = calibrated_models
        
        self.models = trained_models
        self.performance_metrics = model_performances
        
        training_time = (time.time() - training_start) * 1000
        logger.info(f"🎉 Ensemble treinado em {training_time:.1f}ms")
        
        return model_performances
    
    def _create_ensemble_prediction(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cria predição do ensemble usando voting"""
        if not self.models:
            raise ValueError("Modelos não foram treinados. Execute fit primeiro.")
        
        # Coletar predições de todos os modelos
        predictions = []
        probabilities = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[:, 1]
            else:
                proba = pred.astype(float)
            probabilities.append(proba)
        
        predictions = np.array(predictions).T
        probabilities = np.array(probabilities).T
        
        # Aplicar pesos se configurado
        weights = self.config['ensemble_voting']['weights']
        if len(weights) == len(self.models):
            weights = np.array(weights)
        else:
            weights = np.ones(len(self.models)) / len(self.models)
        
        # Voting
        if self.config['ensemble_voting']['method'] == 'hard':
            # Hard voting - maioria simples
            ensemble_pred = np.apply_along_axis(
                lambda x: np.bincount(x, weights=weights).argmax(),
                axis=1, arr=predictions
            )
            ensemble_proba = np.average(probabilities, axis=1, weights=weights)
        else:
            # Soft voting - média ponderada das probabilidades
            ensemble_proba = np.average(probabilities, axis=1, weights=weights)
            ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
        return ensemble_pred, ensemble_proba
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'OptimizedFraudEngine':
        """Treina o motor de fraude otimizado"""
        logger.info("🚀 Iniciando treinamento do motor de fraude otimizado")
        
        # Salvar nomes das features
        self.feature_names = list(X.columns)
        
        # Pré-processamento
        logger.info("🔄 Pré-processando dados")
        X_processed = self._preprocess_data(X, fit_transform=True)
        
        # Seleção de features
        logger.info("🎯 Selecionando features")
        X_selected = self._select_features(X_processed, y, fit_transform=True)
        
        # Treinar ensemble
        logger.info("🤖 Treinando ensemble de modelos")
        performances = self._train_ensemble(X_selected, y)
        
        # Avaliar ensemble completo
        logger.info("📊 Avaliando ensemble completo")
        ensemble_pred, ensemble_proba = self._create_ensemble_prediction(X_selected)
        
        # Métricas do ensemble
        ensemble_performance = {
            'accuracy': accuracy_score(y, ensemble_pred),
            'precision': precision_score(y, ensemble_pred, zero_division=0),
            'recall': recall_score(y, ensemble_pred, zero_division=0),
            'f1_score': f1_score(y, ensemble_pred, zero_division=0),
            'auc_roc': roc_auc_score(y, ensemble_proba) if len(np.unique(y)) > 1 else 0.5
        }
        
        self.performance_metrics['ensemble'] = ensemble_performance
        self.is_trained = True
        
        # Salvar histórico de treinamento
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'model_version': self.model_version,
            'n_samples': len(X),
            'n_features_original': X.shape[1],
            'n_features_selected': X_selected.shape[1],
            'performance': ensemble_performance,
            'individual_performances': performances
        }
        self.training_history.append(training_record)
        
        logger.info("✅ Treinamento concluído!")
        logger.info(f"📊 Ensemble F1-Score: {ensemble_performance['f1_score']:.3f}")
        logger.info(f"📊 Ensemble Accuracy: {ensemble_performance['accuracy']:.3f}")
        logger.info(f"📊 Ensemble Recall: {ensemble_performance['recall']:.3f}")
        logger.info(f"📊 Ensemble Precision: {ensemble_performance['precision']:.3f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> List[FraudPrediction]:
        """Faz predições de fraude otimizadas"""
        if not self.is_trained:
            raise ValueError("Motor não foi treinado. Execute fit primeiro.")
        
        predictions = []
        
        for idx, row in X.iterrows():
            pred_start = time.time()
            
            # Preparar dados
            X_single = pd.DataFrame([row])
            
            # Pré-processar
            X_processed = self._preprocess_data(X_single, fit_transform=False)
            X_selected = self._select_features(X_processed, fit_transform=False)
            
            # Predição do ensemble
            ensemble_pred, ensemble_proba = self._create_ensemble_prediction(X_selected)
            
            is_fraud = bool(ensemble_pred[0])
            fraud_probability = float(ensemble_proba[0])
            
            # Calcular risk score e level
            risk_score = fraud_probability
            
            if risk_score >= self.config['risk_thresholds']['critical']:
                risk_level = 'CRITICAL'
            elif risk_score >= self.config['risk_thresholds']['high']:
                risk_level = 'HIGH'
            elif risk_score >= self.config['risk_thresholds']['medium']:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            # Calcular confiança (baseada na consistência dos modelos)
            individual_probs = []
            for model in self.models.values():
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_selected)[0, 1]
                else:
                    prob = float(model.predict(X_selected)[0])
                individual_probs.append(prob)
            
            confidence = 1.0 - np.std(individual_probs)  # Menor desvio = maior confiança
            
            # Explicação simplificada (top 3 features mais importantes)
            explanation = {
                'model_count': len(self.models),
                'ensemble_method': self.config['ensemble_voting']['method'],
                'risk_factors': self._get_risk_factors(row),
                'model_agreement': confidence
            }
            
            processing_time = (time.time() - pred_start) * 1000
            
            prediction = FraudPrediction(
                transaction_id=str(idx),
                is_fraud=is_fraud,
                fraud_probability=fraud_probability,
                risk_score=risk_score,
                risk_level=risk_level,
                confidence=confidence,
                explanation=explanation,
                processing_time_ms=processing_time,
                model_version=self.model_version,
                timestamp=datetime.now().isoformat()
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def _get_risk_factors(self, transaction: pd.Series) -> List[str]:
        """Identifica fatores de risco na transação"""
        risk_factors = []
        
        # Verificar valor alto
        if 'amount' in transaction and transaction['amount'] > 5000:
            risk_factors.append('high_amount')
        
        # Verificar horário incomum
        if 'hour' in transaction and transaction['hour'] in [2, 3, 4, 23]:
            risk_factors.append('unusual_hour')
        
        # Verificar scores de risco
        if 'location_risk_score' in transaction and transaction['location_risk_score'] > 0.7:
            risk_factors.append('high_location_risk')
        
        if 'device_risk_score' in transaction and transaction['device_risk_score'] > 0.7:
            risk_factors.append('high_device_risk')
        
        return risk_factors
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de performance detalhadas"""
        if not self.is_trained:
            return {}
        
        return {
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'n_models': len(self.models),
            'performance_metrics': self.performance_metrics,
            'training_history': self.training_history,
            'config': self.config,
            'selected_features': getattr(self, 'selected_features', []),
            'last_updated': datetime.now().isoformat()
        }
    
    def save_model(self, filepath: str = None) -> str:
        """Salva o modelo treinado"""
        if not self.is_trained:
            raise ValueError("Motor não foi treinado. Execute fit primeiro.")
        
        if not filepath:
            filepath = os.path.join(
                self.model_dir, 
                f"optimized_fraud_engine_{self.model_version}.joblib"
            )
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'selected_features': getattr(self, 'selected_features', []),
            'config': self.config,
            'performance_metrics': self.performance_metrics,
            'training_history': self.training_history,
            'model_version': self.model_version,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"💾 Modelo salvo: {filepath}")
        
        return filepath
    
    def load_model(self, filepath: str) -> 'OptimizedFraudEngine':
        """Carrega modelo treinado"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.feature_names = model_data['feature_names']
        self.selected_features = model_data.get('selected_features', [])
        self.config = model_data['config']
        self.performance_metrics = model_data['performance_metrics']
        self.training_history = model_data['training_history']
        self.model_version = model_data['model_version']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"📂 Modelo carregado: {filepath}")
        logger.info(f"🔧 Versão: {self.model_version}")
        
        return self

# Instância global do motor otimizado
optimized_fraud_engine = OptimizedFraudEngine()

if __name__ == "__main__":
    # Teste do motor otimizado
    print("🚀 Testando Motor de Fraude Otimizado")
    print("=" * 50)
    
    # Gerar dados de teste
    np.random.seed(42)
    n_samples = 10000
    
    # Features mais realistas
    data = {
        'amount': np.random.lognormal(3, 1.5, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'merchant_category': np.random.randint(1, 20, n_samples),
        'payment_method': np.random.randint(1, 5, n_samples),
        'location_risk_score': np.random.beta(2, 5, n_samples),
        'device_risk_score': np.random.beta(2, 8, n_samples),
        'account_age_days': np.random.exponential(365, n_samples),
        'transaction_frequency': np.random.poisson(5, n_samples)
    }
    
    # Adicionar features PCA simuladas
    for i in range(1, 21):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    X = pd.DataFrame(data)
    
    # Gerar labels com padrões mais realistas
    fraud_rate = 0.02  # 2% de fraude
    n_frauds = int(n_samples * fraud_rate)
    
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, n_frauds, replace=False)
    y[fraud_indices] = 1
    
    # Modificar features para fraudes (padrões mais pronunciados)
    for idx in fraud_indices:
        # Fraudes em horários incomuns
        if np.random.random() < 0.4:
            X.loc[idx, 'hour'] = np.random.choice([2, 3, 4, 23])
        
        # Valores extremos
        if np.random.random() < 0.5:
            X.loc[idx, 'amount'] = np.random.choice([
                np.random.uniform(8000, 15000),  # Valores muito altos
                np.random.uniform(1, 5)          # Valores muito baixos
            ])
        
        # Scores de risco altos
        X.loc[idx, 'location_risk_score'] = np.random.beta(8, 2)
        X.loc[idx, 'device_risk_score'] = np.random.beta(7, 2)
        
        # Padrões suspeitos em features PCA
        for i in range(1, 11):  # Primeiras 10 features PCA
            X.loc[idx, f'V{i}'] = np.random.normal(2, 0.5)  # Valores anômalos
    
    print(f"📊 Dataset: {len(X)} transações, {y.sum()} fraudes ({y.mean()*100:.1f}%)")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Treinar motor
    print("🔄 Treinando motor otimizado...")
    start_time = time.time()
    
    engine = OptimizedFraudEngine()
    engine.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"✅ Treinamento concluído em {training_time:.1f}s")
    
    # Testar predições
    print("🔍 Testando predições...")
    test_sample = X_test.head(100)
    predictions = engine.predict(test_sample)
    
    # Avaliar performance no conjunto de teste
    y_pred = [p.is_fraud for p in predictions]
    y_true = y_test[:len(predictions)]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"📊 Performance no teste:")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1-Score: {f1:.3f}")
    
    # Mostrar algumas predições
    print(f"\n🔍 Exemplos de predições:")
    for i, pred in enumerate(predictions[:5]):
        print(f"   Transação {i+1}: {'FRAUDE' if pred.is_fraud else 'LEGÍTIMA'} "
              f"(Prob: {pred.fraud_probability:.3f}, Risco: {pred.risk_level})")
    
    # Salvar modelo
    model_path = engine.save_model()
    print(f"💾 Modelo salvo: {model_path}")
    
    print("🎉 Motor de Fraude Otimizado testado com sucesso!")
