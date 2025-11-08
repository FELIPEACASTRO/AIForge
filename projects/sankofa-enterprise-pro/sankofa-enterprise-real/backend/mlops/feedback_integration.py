#!/usr/bin/env python3
"""
Integração do Módulo de Feedback Humano com Sistema de Retreinamento Automático
Sankofa Enterprise Pro - Feedback Integration
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import threading
import time

from backend.mlops.human_feedback_module import HumanFeedbackModule
from backend.mlops.model_lifecycle_manager import ModelLifecycleManager, ModelMetrics, RetrainingTrigger

logger = logging.getLogger(__name__)

@dataclass
class FeedbackMetrics:
    """Métricas derivadas do feedback humano"""
    total_feedbacks: int
    model_accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    feedback_trend: str  # 'improving', 'degrading', 'stable'
    last_updated: str

class FeedbackIntegration:
    """Integração entre feedback humano e sistema de retreinamento"""
    
    def __init__(self, feedback_module: HumanFeedbackModule, 
                 lifecycle_manager: ModelLifecycleManager):
        self.feedback_module = feedback_module
        self.lifecycle_manager = lifecycle_manager
        
        # Configurações de thresholds para retreinamento baseado em feedback
        self.feedback_thresholds = {
            'min_feedbacks_for_analysis': 50,  # Mínimo de feedbacks para análise
            'accuracy_degradation_threshold': 0.05,  # 5% de degradação
            'false_positive_rate_threshold': 0.15,  # 15% de falsos positivos
            'false_negative_rate_threshold': 0.10,  # 10% de falsos negativos
            'feedback_window_hours': 168,  # 1 semana
            'min_feedback_trend_samples': 20  # Mínimo para análise de tendência
        }
        
        self.monitoring_active = False
        self.monitoring_thread = None
        self.last_feedback_analysis = None
        
        logger.info("🔗 Feedback Integration inicializada")
    
    def start_feedback_monitoring(self, interval_minutes: int = 60):
        """Inicia o monitoramento contínuo do feedback"""
        if self.monitoring_active:
            logger.warning("⚠️ Monitoramento de feedback já está ativo")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._feedback_monitoring_loop,
            args=(interval_minutes,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"👁️ Monitoramento de feedback iniciado (intervalo: {interval_minutes} minutos)")
    
    def stop_feedback_monitoring(self):
        """Para o monitoramento de feedback"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=30)
        
        logger.info("🛑 Monitoramento de feedback parado")
    
    def _feedback_monitoring_loop(self, interval_minutes: int):
        """Loop principal de monitoramento de feedback"""
        while self.monitoring_active:
            try:
                # Analisar feedback e verificar necessidade de retreinamento
                feedback_metrics = self.analyze_recent_feedback()
                
                if feedback_metrics:
                    self._check_feedback_triggers(feedback_metrics)
                
                # Aguardar próximo ciclo
                time.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"❌ Erro no loop de monitoramento de feedback: {e}")
                time.sleep(300)  # Aguardar 5 minutos antes de tentar novamente
    
    def analyze_recent_feedback(self, hours_back: int = None) -> Optional[FeedbackMetrics]:
        """Analisa o feedback recente e calcula métricas"""
        try:
            if hours_back is None:
                hours_back = self.feedback_thresholds['feedback_window_hours']
            
            # Obter feedbacks
            feedback_df = self.feedback_module.get_feedback()
            
            if feedback_df.empty:
                logger.info("📊 Nenhum feedback disponível para análise")
                return None
            
            # Filtrar feedbacks recentes
            feedback_df['feedback_timestamp'] = pd.to_datetime(feedback_df['feedback_timestamp'])
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_feedback = feedback_df[feedback_df['feedback_timestamp'] >= cutoff_time]
            
            if len(recent_feedback) < self.feedback_thresholds['min_feedbacks_for_analysis']:
                logger.info(f"📊 Poucos feedbacks para análise: {len(recent_feedback)} < {self.feedback_thresholds['min_feedbacks_for_analysis']}")
                return None
            
            # Calcular métricas
            metrics = self._calculate_feedback_metrics(recent_feedback)
            
            # Analisar tendência
            trend = self._analyze_feedback_trend(feedback_df)
            metrics['feedback_trend'] = trend
            
            feedback_metrics = FeedbackMetrics(
                total_feedbacks=len(recent_feedback),
                model_accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                false_positive_rate=metrics['false_positive_rate'],
                false_negative_rate=metrics['false_negative_rate'],
                feedback_trend=trend,
                last_updated=datetime.now().isoformat()
            )
            
            self.last_feedback_analysis = feedback_metrics
            
            logger.info(f"📊 Análise de feedback concluída: "
                       f"Accuracy={metrics['accuracy']:.3f}, "
                       f"Precision={metrics['precision']:.3f}, "
                       f"Recall={metrics['recall']:.3f}, "
                       f"Trend={trend}")
            
            return feedback_metrics
            
        except Exception as e:
            logger.error(f"❌ Erro na análise de feedback: {e}")
            return None
    
    def _calculate_feedback_metrics(self, feedback_df: pd.DataFrame) -> Dict[str, float]:
        """Calcula métricas baseadas no feedback"""
        total = len(feedback_df)
        
        # Matriz de confusão
        true_positives = len(feedback_df[
            (feedback_df['model_prediction'] == 1) & (feedback_df['actual_label'] == 1)
        ])
        true_negatives = len(feedback_df[
            (feedback_df['model_prediction'] == 0) & (feedback_df['actual_label'] == 0)
        ])
        false_positives = len(feedback_df[
            (feedback_df['model_prediction'] == 1) & (feedback_df['actual_label'] == 0)
        ])
        false_negatives = len(feedback_df[
            (feedback_df['model_prediction'] == 0) & (feedback_df['actual_label'] == 1)
        ])
        
        # Métricas
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        false_positive_rate = false_positives / total if total > 0 else 0
        false_negative_rate = false_negatives / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def _analyze_feedback_trend(self, feedback_df: pd.DataFrame) -> str:
        """Analisa a tendência do feedback ao longo do tempo"""
        try:
            if len(feedback_df) < self.feedback_thresholds['min_feedback_trend_samples']:
                return 'insufficient_data'
            
            # Ordenar por timestamp
            feedback_df = feedback_df.sort_values('feedback_timestamp')
            
            # Dividir em duas metades para comparar
            mid_point = len(feedback_df) // 2
            first_half = feedback_df.iloc[:mid_point]
            second_half = feedback_df.iloc[mid_point:]
            
            # Calcular accuracy para cada metade
            first_metrics = self._calculate_feedback_metrics(first_half)
            second_metrics = self._calculate_feedback_metrics(second_half)
            
            accuracy_change = second_metrics['accuracy'] - first_metrics['accuracy']
            
            # Determinar tendência
            if accuracy_change > 0.02:  # Melhoria de 2%
                return 'improving'
            elif accuracy_change < -0.02:  # Degradação de 2%
                return 'degrading'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"❌ Erro na análise de tendência: {e}")
            return 'unknown'
    
    def _check_feedback_triggers(self, feedback_metrics: FeedbackMetrics):
        """Verifica se o feedback indica necessidade de retreinamento"""
        triggers = []
        
        # Verificar degradação de accuracy
        if feedback_metrics.model_accuracy < (0.95 - self.feedback_thresholds['accuracy_degradation_threshold']):
            triggers.append(RetrainingTrigger(
                trigger_type='feedback_accuracy_degradation',
                threshold_value=0.95 - self.feedback_thresholds['accuracy_degradation_threshold'],
                current_value=feedback_metrics.model_accuracy,
                triggered_at=datetime.now().isoformat(),
                severity='high' if feedback_metrics.model_accuracy < 0.85 else 'medium'
            ))
        
        # Verificar alta taxa de falsos positivos
        if feedback_metrics.false_positive_rate > self.feedback_thresholds['false_positive_rate_threshold']:
            triggers.append(RetrainingTrigger(
                trigger_type='high_false_positive_rate',
                threshold_value=self.feedback_thresholds['false_positive_rate_threshold'],
                current_value=feedback_metrics.false_positive_rate,
                triggered_at=datetime.now().isoformat(),
                severity='medium'
            ))
        
        # Verificar alta taxa de falsos negativos
        if feedback_metrics.false_negative_rate > self.feedback_thresholds['false_negative_rate_threshold']:
            triggers.append(RetrainingTrigger(
                trigger_type='high_false_negative_rate',
                threshold_value=self.feedback_thresholds['false_negative_rate_threshold'],
                current_value=feedback_metrics.false_negative_rate,
                triggered_at=datetime.now().isoformat(),
                severity='critical'  # Falsos negativos são críticos em detecção de fraude
            ))
        
        # Verificar tendência de degradação
        if feedback_metrics.feedback_trend == 'degrading':
            triggers.append(RetrainingTrigger(
                trigger_type='feedback_trend_degradation',
                threshold_value=0.0,  # Qualquer degradação
                current_value=-1.0,  # Valor simbólico para degradação
                triggered_at=datetime.now().isoformat(),
                severity='medium'
            ))
        
        # Adicionar triggers ao lifecycle manager
        for trigger in triggers:
            self.lifecycle_manager.retraining_triggers.append(trigger)
            
            logger.warning(f"⚠️ Trigger de feedback: {trigger.trigger_type} "
                         f"({trigger.severity}) - {trigger.current_value:.3f}")
        
        # Verificar se deve iniciar retreinamento automático
        if triggers:
            critical_triggers = [t for t in triggers if t.severity == 'critical']
            high_triggers = [t for t in triggers if t.severity == 'high']
            
            if critical_triggers or len(high_triggers) >= 2:
                logger.warning("🔄 Iniciando retreinamento baseado em feedback")
                self._trigger_feedback_based_retraining(triggers)
    
    def _trigger_feedback_based_retraining(self, triggers: List[RetrainingTrigger]):
        """Inicia retreinamento baseado no feedback"""
        try:
            logger.info("🤖 Retreinamento baseado em feedback iniciado")
            
            # Preparar dados de retreinamento com feedback
            training_data = self._prepare_feedback_training_data()
            
            if training_data is None:
                logger.error("❌ Não foi possível preparar dados de retreinamento")
                return
            
            # Criar job de retreinamento
            retraining_job = {
                'job_id': f"feedback_retrain_{int(time.time())}",
                'triggered_at': datetime.now().isoformat(),
                'trigger_source': 'human_feedback',
                'triggers': [trigger.__dict__ for trigger in triggers],
                'feedback_metrics': self.last_feedback_analysis.__dict__ if self.last_feedback_analysis else {},
                'status': 'queued',
                'training_data_size': len(training_data) if training_data is not None else 0
            }
            
            # Salvar job (em um ambiente real, isso seria enviado para uma fila de processamento)
            import json
            import os
            
            job_path = f"/tmp/feedback_retraining_job_{retraining_job['job_id']}.json"
            with open(job_path, 'w') as f:
                json.dump(retraining_job, f, indent=2, default=str)
            
            logger.info(f"📋 Job de retreinamento baseado em feedback criado: {retraining_job['job_id']}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao iniciar retreinamento baseado em feedback: {e}")
    
    def _prepare_feedback_training_data(self) -> Optional[pd.DataFrame]:
        """Prepara dados de treinamento incorporando o feedback humano"""
        try:
            feedback_df = self.feedback_module.get_feedback()
            
            if feedback_df.empty:
                return None
            
            # Filtrar apenas feedbacks com labels corretos (actual_label)
            # Estes serão usados como dados de treinamento corrigidos
            corrected_data = feedback_df[['transaction_id', 'actual_label']].copy()
            corrected_data = corrected_data.rename(columns={'actual_label': 'is_fraud'})
            
            logger.info(f"📊 Dados de retreinamento preparados: {len(corrected_data)} amostras corrigidas")
            
            return corrected_data
            
        except Exception as e:
            logger.error(f"❌ Erro ao preparar dados de retreinamento: {e}")
            return None
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Retorna resumo do status do feedback"""
        try:
            feedback_metrics = self.analyze_recent_feedback()
            
            if not feedback_metrics:
                return {
                    'status': 'insufficient_data',
                    'message': 'Dados de feedback insuficientes para análise'
                }
            
            return {
                'status': 'active',
                'metrics': {
                    'total_feedbacks': feedback_metrics.total_feedbacks,
                    'model_accuracy': feedback_metrics.model_accuracy,
                    'precision': feedback_metrics.precision,
                    'recall': feedback_metrics.recall,
                    'f1_score': feedback_metrics.f1_score,
                    'false_positive_rate': feedback_metrics.false_positive_rate,
                    'false_negative_rate': feedback_metrics.false_negative_rate,
                    'feedback_trend': feedback_metrics.feedback_trend
                },
                'monitoring_active': self.monitoring_active,
                'last_analysis': feedback_metrics.last_updated,
                'thresholds': self.feedback_thresholds
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter resumo de feedback: {e}")
            return {
                'status': 'error',
                'message': f'Erro ao obter resumo: {str(e)}'
            }

