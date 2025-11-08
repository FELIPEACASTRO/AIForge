#!/usr/bin/env python3
"""
Sistema de Detecção de Drift para MLOps
Sankofa Enterprise Pro - Drift Detection System
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.metrics import jensen_shannon_distance
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class DriftReport:
    """Relatório de drift detectado"""
    feature_name: str
    drift_type: str  # 'data_drift', 'concept_drift'
    drift_score: float
    threshold: float
    is_drift_detected: bool
    detection_method: str
    timestamp: str
    reference_period: str
    current_period: str
    severity: str  # 'low', 'medium', 'high', 'critical'

class DriftDetector:
    """Sistema de detecção de drift de dados e conceitos"""
    
    def __init__(self, reference_window_hours: int = 168, detection_window_hours: int = 24):
        self.reference_window_hours = reference_window_hours  # 1 semana
        self.detection_window_hours = detection_window_hours  # 1 dia
        
        # Thresholds para diferentes tipos de drift
        self.drift_thresholds = {
            'data_drift': {
                'low': 0.1,
                'medium': 0.2,
                'high': 0.3,
                'critical': 0.5
            },
            'concept_drift': {
                'low': 0.15,
                'medium': 0.25,
                'high': 0.35,
                'critical': 0.5
            }
        }
        
        self.reference_data: Dict[str, Any] = {}
        self.drift_history: List[DriftReport] = []
        
        logger.info("🔍 Drift Detector inicializado")
        logger.info(f"📊 Janela de referência: {reference_window_hours}h")
        logger.info(f"🎯 Janela de detecção: {detection_window_hours}h")
    
    def set_reference_data(self, data: pd.DataFrame, target_column: str = None):
        """Define os dados de referência para comparação"""
        self.reference_data = {
            'data': data.copy(),
            'target_column': target_column,
            'timestamp': datetime.now().isoformat(),
            'statistics': self._calculate_statistics(data, target_column)
        }
        
        logger.info(f"📋 Dados de referência definidos: {len(data)} amostras")
        logger.info(f"🎯 Colunas: {list(data.columns)}")
    
    def detect_data_drift(self, current_data: pd.DataFrame) -> List[DriftReport]:
        """Detecta drift nos dados de entrada"""
        if not self.reference_data:
            logger.warning("⚠️ Dados de referência não definidos")
            return []
        
        drift_reports = []
        reference_df = self.reference_data['data']
        
        # Verificar drift para cada feature numérica
        numeric_columns = current_data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in reference_df.columns:
                drift_score = self._calculate_data_drift_score(
                    reference_df[column].values,
                    current_data[column].values
                )
                
                severity = self._determine_severity(drift_score, 'data_drift')
                is_drift = drift_score > self.drift_thresholds['data_drift']['low']
                
                report = DriftReport(
                    feature_name=column,
                    drift_type='data_drift',
                    drift_score=drift_score,
                    threshold=self.drift_thresholds['data_drift']['low'],
                    is_drift_detected=is_drift,
                    detection_method='jensen_shannon_divergence',
                    timestamp=datetime.now().isoformat(),
                    reference_period=f"{self.reference_window_hours}h_ago",
                    current_period=f"last_{self.detection_window_hours}h",
                    severity=severity
                )
                
                drift_reports.append(report)
                
                if is_drift:
                    logger.warning(f"📊 Data drift detectado em {column}: "
                                 f"{drift_score:.3f} ({severity})")
        
        # Verificar drift para features categóricas
        categorical_columns = current_data.select_dtypes(include=['object', 'category']).columns
        
        for column in categorical_columns:
            if column in reference_df.columns:
                drift_score = self._calculate_categorical_drift_score(
                    reference_df[column].values,
                    current_data[column].values
                )
                
                severity = self._determine_severity(drift_score, 'data_drift')
                is_drift = drift_score > self.drift_thresholds['data_drift']['low']
                
                report = DriftReport(
                    feature_name=column,
                    drift_type='data_drift',
                    drift_score=drift_score,
                    threshold=self.drift_thresholds['data_drift']['low'],
                    is_drift_detected=is_drift,
                    detection_method='chi_square_test',
                    timestamp=datetime.now().isoformat(),
                    reference_period=f"{self.reference_window_hours}h_ago",
                    current_period=f"last_{self.detection_window_hours}h",
                    severity=severity
                )
                
                drift_reports.append(report)
                
                if is_drift:
                    logger.warning(f"📊 Data drift detectado em {column}: "
                                 f"{drift_score:.3f} ({severity})")
        
        self.drift_history.extend(drift_reports)
        return drift_reports
    
    def detect_concept_drift(self, current_data: pd.DataFrame, 
                           current_predictions: np.ndarray,
                           current_actuals: np.ndarray = None) -> List[DriftReport]:
        """Detecta drift no conceito (relação entre features e target)"""
        if not self.reference_data or self.reference_data['target_column'] is None:
            logger.warning("⚠️ Dados de referência com target não definidos")
            return []
        
        drift_reports = []
        
        # Drift baseado na distribuição das predições
        reference_stats = self.reference_data['statistics']
        
        if 'prediction_distribution' in reference_stats:
            current_pred_dist = self._calculate_prediction_distribution(current_predictions)
            reference_pred_dist = reference_stats['prediction_distribution']
            
            drift_score = self._calculate_distribution_drift(
                reference_pred_dist, current_pred_dist
            )
            
            severity = self._determine_severity(drift_score, 'concept_drift')
            is_drift = drift_score > self.drift_thresholds['concept_drift']['low']
            
            report = DriftReport(
                feature_name='prediction_distribution',
                drift_type='concept_drift',
                drift_score=drift_score,
                threshold=self.drift_thresholds['concept_drift']['low'],
                is_drift_detected=is_drift,
                detection_method='prediction_distribution_shift',
                timestamp=datetime.now().isoformat(),
                reference_period=f"{self.reference_window_hours}h_ago",
                current_period=f"last_{self.detection_window_hours}h",
                severity=severity
            )
            
            drift_reports.append(report)
            
            if is_drift:
                logger.warning(f"🎯 Concept drift detectado na distribuição de predições: "
                             f"{drift_score:.3f} ({severity})")
        
        # Drift baseado na performance (se temos actuals)
        if current_actuals is not None and 'performance_metrics' in reference_stats:
            current_accuracy = np.mean(current_predictions == current_actuals)
            reference_accuracy = reference_stats['performance_metrics'].get('accuracy', 0.0)
            
            # Drift de performance (degradação significativa)
            performance_drift = abs(current_accuracy - reference_accuracy)
            
            severity = self._determine_severity(performance_drift, 'concept_drift')
            is_drift = performance_drift > self.drift_thresholds['concept_drift']['low']
            
            report = DriftReport(
                feature_name='model_performance',
                drift_type='concept_drift',
                drift_score=performance_drift,
                threshold=self.drift_thresholds['concept_drift']['low'],
                is_drift_detected=is_drift,
                detection_method='performance_degradation',
                timestamp=datetime.now().isoformat(),
                reference_period=f"{self.reference_window_hours}h_ago",
                current_period=f"last_{self.detection_window_hours}h",
                severity=severity
            )
            
            drift_reports.append(report)
            
            if is_drift:
                logger.warning(f"📉 Concept drift detectado na performance: "
                             f"{performance_drift:.3f} ({severity})")
        
        self.drift_history.extend(drift_reports)
        return drift_reports
    
    def _calculate_data_drift_score(self, reference_data: np.ndarray, 
                                  current_data: np.ndarray) -> float:
        """Calcula score de drift para dados numéricos usando Jensen-Shannon divergence"""
        try:
            # Criar histogramas normalizados
            bins = np.linspace(
                min(np.min(reference_data), np.min(current_data)),
                max(np.max(reference_data), np.max(current_data)),
                50
            )
            
            ref_hist, _ = np.histogram(reference_data, bins=bins, density=True)
            cur_hist, _ = np.histogram(current_data, bins=bins, density=True)
            
            # Normalizar para que sejam distribuições de probabilidade
            ref_hist = ref_hist / np.sum(ref_hist)
            cur_hist = cur_hist / np.sum(cur_hist)
            
            # Adicionar pequena constante para evitar divisão por zero
            epsilon = 1e-10
            ref_hist = ref_hist + epsilon
            cur_hist = cur_hist + epsilon
            
            # Calcular Jensen-Shannon divergence
            m = 0.5 * (ref_hist + cur_hist)
            js_div = 0.5 * stats.entropy(ref_hist, m) + 0.5 * stats.entropy(cur_hist, m)
            
            return np.sqrt(js_div)  # Jensen-Shannon distance
            
        except Exception as e:
            logger.error(f"❌ Erro ao calcular drift score: {e}")
            return 0.0
    
    def _calculate_categorical_drift_score(self, reference_data: np.ndarray, 
                                         current_data: np.ndarray) -> float:
        """Calcula score de drift para dados categóricos usando chi-square test"""
        try:
            # Obter categorias únicas
            all_categories = np.unique(np.concatenate([reference_data, current_data]))
            
            # Contar frequências
            ref_counts = pd.Series(reference_data).value_counts().reindex(all_categories, fill_value=0)
            cur_counts = pd.Series(current_data).value_counts().reindex(all_categories, fill_value=0)
            
            # Normalizar para proporções
            ref_props = ref_counts / ref_counts.sum()
            cur_props = cur_counts / cur_counts.sum()
            
            # Calcular chi-square estatística
            expected = ref_props * len(current_data)
            observed = cur_counts
            
            # Evitar divisão por zero
            expected = np.where(expected == 0, 1e-10, expected)
            
            chi_square = np.sum((observed - expected) ** 2 / expected)
            
            # Normalizar pelo número de categorias
            normalized_chi_square = chi_square / len(all_categories)
            
            return min(normalized_chi_square / 100, 1.0)  # Limitar a 1.0
            
        except Exception as e:
            logger.error(f"❌ Erro ao calcular drift categórico: {e}")
            return 0.0
    
    def _calculate_prediction_distribution(self, predictions: np.ndarray) -> Dict[str, float]:
        """Calcula distribuição das predições"""
        unique_values, counts = np.unique(predictions, return_counts=True)
        total = len(predictions)
        
        distribution = {}
        for value, count in zip(unique_values, counts):
            distribution[str(value)] = count / total
        
        return distribution
    
    def _calculate_distribution_drift(self, ref_dist: Dict[str, float], 
                                    cur_dist: Dict[str, float]) -> float:
        """Calcula drift entre duas distribuições"""
        all_keys = set(ref_dist.keys()) | set(cur_dist.keys())
        
        ref_values = np.array([ref_dist.get(key, 0.0) for key in all_keys])
        cur_values = np.array([cur_dist.get(key, 0.0) for key in all_keys])
        
        # Adicionar epsilon para evitar problemas numéricos
        epsilon = 1e-10
        ref_values = ref_values + epsilon
        cur_values = cur_values + epsilon
        
        # Normalizar
        ref_values = ref_values / np.sum(ref_values)
        cur_values = cur_values / np.sum(cur_values)
        
        # Jensen-Shannon divergence
        m = 0.5 * (ref_values + cur_values)
        js_div = 0.5 * stats.entropy(ref_values, m) + 0.5 * stats.entropy(cur_values, m)
        
        return np.sqrt(js_div)
    
    def _determine_severity(self, drift_score: float, drift_type: str) -> str:
        """Determina a severidade do drift"""
        thresholds = self.drift_thresholds[drift_type]
        
        if drift_score >= thresholds['critical']:
            return 'critical'
        elif drift_score >= thresholds['high']:
            return 'high'
        elif drift_score >= thresholds['medium']:
            return 'medium'
        elif drift_score >= thresholds['low']:
            return 'low'
        else:
            return 'none'
    
    def _calculate_statistics(self, data: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Calcula estatísticas dos dados de referência"""
        stats = {
            'shape': data.shape,
            'numeric_features': {},
            'categorical_features': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Estatísticas para features numéricas
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            stats['numeric_features'][column] = {
                'mean': float(data[column].mean()),
                'std': float(data[column].std()),
                'min': float(data[column].min()),
                'max': float(data[column].max()),
                'median': float(data[column].median()),
                'q25': float(data[column].quantile(0.25)),
                'q75': float(data[column].quantile(0.75))
            }
        
        # Estatísticas para features categóricas
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        for column in categorical_columns:
            value_counts = data[column].value_counts()
            stats['categorical_features'][column] = {
                'unique_values': len(value_counts),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'distribution': value_counts.to_dict()
            }
        
        # Estatísticas do target se fornecido
        if target_column and target_column in data.columns:
            if data[target_column].dtype in ['object', 'category']:
                target_counts = data[target_column].value_counts()
                stats['target_distribution'] = target_counts.to_dict()
            else:
                stats['target_statistics'] = {
                    'mean': float(data[target_column].mean()),
                    'std': float(data[target_column].std())
                }
        
        return stats
    
    def get_drift_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Retorna resumo dos drifts detectados"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_drifts = [
            drift for drift in self.drift_history
            if datetime.fromisoformat(drift.timestamp) > cutoff_time
        ]
        
        summary = {
            'total_drifts_detected': len([d for d in recent_drifts if d.is_drift_detected]),
            'data_drifts': len([d for d in recent_drifts if d.drift_type == 'data_drift' and d.is_drift_detected]),
            'concept_drifts': len([d for d in recent_drifts if d.drift_type == 'concept_drift' and d.is_drift_detected]),
            'severity_breakdown': {
                'critical': len([d for d in recent_drifts if d.severity == 'critical']),
                'high': len([d for d in recent_drifts if d.severity == 'high']),
                'medium': len([d for d in recent_drifts if d.severity == 'medium']),
                'low': len([d for d in recent_drifts if d.severity == 'low'])
            },
            'most_affected_features': self._get_most_affected_features(recent_drifts),
            'last_updated': datetime.now().isoformat()
        }
        
        return summary
    
    def _get_most_affected_features(self, drifts: List[DriftReport]) -> List[Dict[str, Any]]:
        """Identifica as features mais afetadas por drift"""
        feature_drifts = {}
        
        for drift in drifts:
            if drift.is_drift_detected:
                if drift.feature_name not in feature_drifts:
                    feature_drifts[drift.feature_name] = {
                        'count': 0,
                        'max_score': 0.0,
                        'avg_score': 0.0,
                        'scores': []
                    }
                
                feature_drifts[drift.feature_name]['count'] += 1
                feature_drifts[drift.feature_name]['scores'].append(drift.drift_score)
                feature_drifts[drift.feature_name]['max_score'] = max(
                    feature_drifts[drift.feature_name]['max_score'],
                    drift.drift_score
                )
        
        # Calcular médias e ordenar
        for feature_name in feature_drifts:
            scores = feature_drifts[feature_name]['scores']
            feature_drifts[feature_name]['avg_score'] = np.mean(scores)
        
        # Ordenar por score médio
        sorted_features = sorted(
            feature_drifts.items(),
            key=lambda x: x[1]['avg_score'],
            reverse=True
        )
        
        return [
            {
                'feature_name': name,
                'drift_count': data['count'],
                'max_drift_score': data['max_score'],
                'avg_drift_score': data['avg_score']
            }
            for name, data in sorted_features[:10]  # Top 10
        ]

# Instância global do detector de drift
drift_detector = DriftDetector()

if __name__ == "__main__":
    # Teste do detector de drift
    detector = DriftDetector()
    
    # Gerar dados de referência
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.exponential(2, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.binomial(1, 0.3, 1000)
    })
    
    detector.set_reference_data(reference_data, 'target')
    
    # Gerar dados com drift
    current_data = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.2, 500),  # Drift na média e variância
        'feature2': np.random.exponential(3, 500),    # Drift no parâmetro
        'category': np.random.choice(['A', 'B', 'C', 'D'], 500),  # Nova categoria
        'target': np.random.binomial(1, 0.5, 500)     # Drift no target
    })
    
    # Detectar drifts
    data_drifts = detector.detect_data_drift(current_data)
    
    print("📊 Drifts de Dados Detectados:")
    for drift in data_drifts:
        if drift.is_drift_detected:
            print(f"  {drift.feature_name}: {drift.drift_score:.3f} ({drift.severity})")
    
    # Simular predições
    current_predictions = np.random.binomial(1, 0.6, 500)  # Drift nas predições
    current_actuals = current_data['target'].values
    
    concept_drifts = detector.detect_concept_drift(
        current_data, current_predictions, current_actuals
    )
    
    print("\n🎯 Drifts de Conceito Detectados:")
    for drift in concept_drifts:
        if drift.is_drift_detected:
            print(f"  {drift.feature_name}: {drift.drift_score:.3f} ({drift.severity})")
    
    # Resumo
    summary = detector.get_drift_summary()
    print(f"\n📋 Resumo: {summary['total_drifts_detected']} drifts detectados")
    print("🔍 Drift Detector testado com sucesso!")
