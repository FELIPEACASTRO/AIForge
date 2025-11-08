#!/usr/bin/env python3
"""
Motor de Fraude Otimizado - Versão Rápida para Testes
Sankofa Enterprise Pro - Fast Optimized Fraud Detection Engine
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

@dataclass
class FraudPrediction:
    """Resultado de predição de fraude"""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    risk_level: str
    confidence: float
    processing_time_ms: float
    timestamp: str

class FastOptimizedFraudEngine:
    """Motor de Detecção de Fraude Otimizado - Versão Rápida"""
    
    def __init__(self):
        # Configurações otimizadas para velocidade
        self.config = {
            'models': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': -1
                },
                'gradient_boosting': {
                    'n_estimators': 50,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42
                },
                'logistic_regression': {
                    'C': 1.0,
                    'class_weight': 'balanced',
                    'max_iter': 500,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'feature_selection': {
                'k_features': 15
            },
            'risk_thresholds': {
                'low': 0.2,
                'medium': 0.5,
                'high': 0.8,
                'critical': 0.95
            }
        }
        
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_classif, k=15)
        self.is_trained = False
        self.performance_metrics = {}
        
        logger.info("🚀 Motor de Fraude Rápido inicializado")
    
    def _preprocess_data(self, X: pd.DataFrame, fit_transform: bool = False) -> np.ndarray:
        """Pré-processa os dados rapidamente"""
        X_processed = X.copy()
        
        # Tratar valores ausentes
        X_processed = X_processed.fillna(X_processed.median())
        
        # Aplicar scaling
        if fit_transform:
            X_scaled = self.scaler.fit_transform(X_processed)
        else:
            X_scaled = self.scaler.transform(X_processed)
        
        return X_scaled
    
    def _select_features(self, X: np.ndarray, y: np.ndarray = None, fit_transform: bool = False) -> np.ndarray:
        """Seleciona features importantes rapidamente"""
        if fit_transform:
            if y is None:
                raise ValueError("y é necessário para fit_transform=True")
            X_selected = self.feature_selector.fit_transform(X, y)
            logger.info(f"🎯 {X_selected.shape[1]} features selecionadas de {X.shape[1]}")
        else:
            X_selected = self.feature_selector.transform(X)
        
        return X_selected
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'FastOptimizedFraudEngine':
        """Treina o motor rapidamente"""
        logger.info("🚀 Iniciando treinamento rápido")
        
        # Pré-processamento
        X_processed = self._preprocess_data(X, fit_transform=True)
        X_selected = self._select_features(X_processed, y, fit_transform=True)
        
        # Dividir para validação
        X_train, X_val, y_train, y_val = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Treinar modelos
        self.models = {}
        
        # Random Forest
        rf = RandomForestClassifier(**self.config['models']['random_forest'])
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(**self.config['models']['gradient_boosting'])
        gb.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb
        
        # Logistic Regression
        lr = LogisticRegression(**self.config['models']['logistic_regression'])
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr
        
        # Avaliar ensemble
        ensemble_pred, ensemble_proba = self._create_ensemble_prediction(X_val)
        
        # Métricas
        self.performance_metrics = {
            'accuracy': accuracy_score(y_val, ensemble_pred),
            'precision': precision_score(y_val, ensemble_pred, zero_division=0),
            'recall': recall_score(y_val, ensemble_pred, zero_division=0),
            'f1_score': f1_score(y_val, ensemble_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_val, ensemble_proba) if len(np.unique(y_val)) > 1 else 0.5
        }
        
        self.is_trained = True
        
        logger.info("✅ Treinamento rápido concluído!")
        logger.info(f"📊 F1-Score: {self.performance_metrics['f1_score']:.3f}")
        logger.info(f"📊 Accuracy: {self.performance_metrics['accuracy']:.3f}")
        logger.info(f"📊 Recall: {self.performance_metrics['recall']:.3f}")
        logger.info(f"📊 Precision: {self.performance_metrics['precision']:.3f}")
        
        return self
    
    def _create_ensemble_prediction(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cria predição do ensemble"""
        probabilities = []
        
        for model in self.models.values():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[:, 1]
            else:
                proba = model.predict(X).astype(float)
            probabilities.append(proba)
        
        # Média das probabilidades
        ensemble_proba = np.mean(probabilities, axis=0)
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
        return ensemble_pred, ensemble_proba
    
    def predict(self, X: pd.DataFrame) -> List[FraudPrediction]:
        """Faz predições rapidamente"""
        if not self.is_trained:
            raise ValueError("Motor não foi treinado")
        
        predictions = []
        
        for idx, row in X.iterrows():
            pred_start = time.time()
            
            # Preparar dados
            X_single = pd.DataFrame([row])
            X_processed = self._preprocess_data(X_single, fit_transform=False)
            X_selected = self._select_features(X_processed, fit_transform=False)
            
            # Predição
            _, ensemble_proba = self._create_ensemble_prediction(X_selected)
            
            fraud_probability = float(ensemble_proba[0])
            is_fraud = fraud_probability >= 0.5
            
            # Risk level
            if fraud_probability >= self.config['risk_thresholds']['critical']:
                risk_level = 'CRITICAL'
            elif fraud_probability >= self.config['risk_thresholds']['high']:
                risk_level = 'HIGH'
            elif fraud_probability >= self.config['risk_thresholds']['medium']:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            processing_time = (time.time() - pred_start) * 1000
            
            prediction = FraudPrediction(
                transaction_id=str(idx),
                is_fraud=is_fraud,
                fraud_probability=fraud_probability,
                risk_score=fraud_probability,
                risk_level=risk_level,
                confidence=0.85,  # Confiança fixa para simplicidade
                processing_time_ms=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de performance"""
        return {
            'is_trained': self.is_trained,
            'performance_metrics': self.performance_metrics,
            'n_models': len(self.models),
            'last_updated': datetime.now().isoformat()
        }

# Instância global
fast_fraud_engine = FastOptimizedFraudEngine()

if __name__ == "__main__":
    # Teste rápido
    print("🚀 Testando Motor de Fraude Rápido")
    print("=" * 40)
    
    # Gerar dados de teste
    np.random.seed(42)
    n_samples = 5000
    
    data = {
        'amount': np.random.lognormal(3, 1.5, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'location_risk_score': np.random.beta(2, 5, n_samples),
        'device_risk_score': np.random.beta(2, 8, n_samples),
        'merchant_category': np.random.randint(1, 20, n_samples),
        'payment_method': np.random.randint(1, 5, n_samples)
    }
    
    # Adicionar features PCA
    for i in range(1, 16):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    X = pd.DataFrame(data)
    
    # Gerar labels com padrões realistas
    fraud_rate = 0.02
    n_frauds = int(n_samples * fraud_rate)
    
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, n_frauds, replace=False)
    y[fraud_indices] = 1
    
    # Criar padrões de fraude mais pronunciados
    for idx in fraud_indices:
        # Horários incomuns
        if np.random.random() < 0.5:
            X.loc[idx, 'hour'] = np.random.choice([2, 3, 4, 23])
        
        # Valores extremos
        if np.random.random() < 0.6:
            X.loc[idx, 'amount'] = np.random.uniform(10000, 20000)
        
        # Scores de risco altos
        X.loc[idx, 'location_risk_score'] = np.random.beta(8, 2)
        X.loc[idx, 'device_risk_score'] = np.random.beta(8, 2)
        
        # Anomalias em features PCA
        for i in range(1, 8):
            X.loc[idx, f'V{i}'] = np.random.normal(3, 0.5)
    
    print(f"📊 Dataset: {len(X)} transações, {y.sum()} fraudes ({y.mean()*100:.1f}%)")
    
    # Treinar
    start_time = time.time()
    engine = FastOptimizedFraudEngine()
    engine.fit(X, y)
    training_time = time.time() - start_time
    
    print(f"⏱️ Treinamento: {training_time:.1f}s")
    
    # Testar predições
    test_sample = X.head(100)
    predictions = engine.predict(test_sample)
    
    # Estatísticas
    fraud_predictions = sum(1 for p in predictions if p.is_fraud)
    avg_processing_time = np.mean([p.processing_time_ms for p in predictions])
    
    print(f"🔍 Predições: {fraud_predictions}/100 fraudes detectadas")
    print(f"⚡ Tempo médio por predição: {avg_processing_time:.2f}ms")
    
    metrics = engine.get_performance_metrics()
    print(f"📊 Métricas finais:")
    for metric, value in metrics['performance_metrics'].items():
        print(f"   {metric}: {value:.3f}")
    
    print("🎉 Teste concluído!")
