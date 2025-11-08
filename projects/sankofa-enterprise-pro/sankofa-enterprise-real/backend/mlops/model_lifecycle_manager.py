#!/usr/bin/env python3
"""
Sistema de MLOps para Gerenciamento do Ciclo de Vida dos Modelos
Sankofa Enterprise Pro - Model Lifecycle Manager
"""

import os
import json
import time
import pickle
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Métricas de performance do modelo"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    throughput_tps: float
    latency_p95_ms: float
    timestamp: str
    model_version: str
    data_drift_score: float = 0.0
    concept_drift_score: float = 0.0

@dataclass
class ModelVersion:
    """Versão do modelo"""
    version: str
    model_path: str
    training_data_hash: str
    performance_metrics: ModelMetrics
    created_at: str
    status: str  # 'training', 'validation', 'production', 'deprecated'
    deployment_config: Dict[str, Any]

@dataclass
class RetrainingTrigger:
    """Gatilho para retreinamento do modelo"""
    trigger_type: str  # 'performance_degradation', 'data_drift', 'concept_drift', 'scheduled'
    threshold_value: float
    current_value: float
    triggered_at: str
    severity: str  # 'low', 'medium', 'high', 'critical'

class ModelLifecycleManager:
    """Gerenciador do ciclo de vida dos modelos de ML"""
    
    def __init__(self, models_dir: str = "models", metrics_dir: str = "metrics"):
        self.models_dir = models_dir
        self.metrics_dir = metrics_dir
        self.model_versions: Dict[str, List[ModelVersion]] = {}
        self.current_production_models: Dict[str, ModelVersion] = {}
        self.performance_history: List[ModelMetrics] = []
        self.retraining_triggers: List[RetrainingTrigger] = []
        
        # Configurações de thresholds para retreinamento
        self.performance_thresholds = {
            'accuracy_min': 0.85,
            'precision_min': 0.80,
            'recall_min': 0.75,
            'f1_score_min': 0.70,
            'throughput_min': 100.0,  # TPS
            'latency_max': 50.0,  # ms
            'data_drift_max': 0.3,
            'concept_drift_max': 0.25
        }
        
        # Configurações de retreinamento
        self.retraining_config = {
            'auto_retrain_enabled': True,
            'validation_holdout_ratio': 0.2,
            'min_training_samples': 10000,
            'max_model_versions': 10,
            'performance_window_hours': 24,
            'drift_detection_window_hours': 12
        }
        
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Criar diretórios se não existirem
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        logger.info("🔄 Model Lifecycle Manager inicializado")
        logger.info(f"📁 Diretório de modelos: {self.models_dir}")
        logger.info(f"📊 Diretório de métricas: {self.metrics_dir}")
    
    def register_model_version(self, model_name: str, model_path: str, 
                             training_data_hash: str, metrics: ModelMetrics,
                             deployment_config: Dict[str, Any] = None) -> str:
        """Registra uma nova versão do modelo"""
        version = f"v{int(time.time())}"
        
        model_version = ModelVersion(
            version=version,
            model_path=model_path,
            training_data_hash=training_data_hash,
            performance_metrics=metrics,
            created_at=datetime.now().isoformat(),
            status='validation',
            deployment_config=deployment_config or {}
        )
        
        if model_name not in self.model_versions:
            self.model_versions[model_name] = []
        
        self.model_versions[model_name].append(model_version)
        
        # Manter apenas as últimas N versões
        max_versions = self.retraining_config['max_model_versions']
        if len(self.model_versions[model_name]) > max_versions:
            # Depreciar versões antigas
            old_versions = self.model_versions[model_name][:-max_versions]
            for old_version in old_versions:
                old_version.status = 'deprecated'
            
            self.model_versions[model_name] = self.model_versions[model_name][-max_versions:]
        
        self._save_model_registry()
        
        logger.info(f"📝 Modelo {model_name} {version} registrado")
        logger.info(f"🎯 Métricas: Acc={metrics.accuracy:.3f}, Prec={metrics.precision:.3f}, "
                   f"Rec={metrics.recall:.3f}, F1={metrics.f1_score:.3f}")
        
        return version
    
    def promote_to_production(self, model_name: str, version: str) -> bool:
        """Promove uma versão do modelo para produção"""
        if model_name not in self.model_versions:
            logger.error(f"❌ Modelo {model_name} não encontrado")
            return False
        
        model_version = None
        for mv in self.model_versions[model_name]:
            if mv.version == version:
                model_version = mv
                break
        
        if not model_version:
            logger.error(f"❌ Versão {version} do modelo {model_name} não encontrada")
            return False
        
        # Validar métricas antes da promoção
        metrics = model_version.performance_metrics
        if not self._validate_production_readiness(metrics):
            logger.error(f"❌ Modelo {model_name} {version} não atende aos critérios de produção")
            return False
        
        # Depreciar modelo atual em produção
        if model_name in self.current_production_models:
            old_model = self.current_production_models[model_name]
            old_model.status = 'deprecated'
            logger.info(f"📤 Modelo anterior {model_name} {old_model.version} depreciado")
        
        # Promover novo modelo
        model_version.status = 'production'
        self.current_production_models[model_name] = model_version
        
        self._save_model_registry()
        
        logger.info(f"🚀 Modelo {model_name} {version} promovido para produção")
        logger.info(f"✅ Métricas de produção: Acc={metrics.accuracy:.3f}, "
                   f"Prec={metrics.precision:.3f}, Rec={metrics.recall:.3f}")
        
        return True
    
    def record_performance_metrics(self, metrics: ModelMetrics):
        """Registra métricas de performance do modelo em produção"""
        self.performance_history.append(metrics)
        
        # Manter apenas métricas das últimas 24h
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.performance_history = [
            m for m in self.performance_history 
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
        
        # Verificar se precisa de retreinamento
        self._check_retraining_triggers(metrics)
        
        # Salvar métricas
        self._save_performance_metrics(metrics)
        
        logger.debug(f"📊 Métricas registradas: Acc={metrics.accuracy:.3f}, "
                    f"TPS={metrics.throughput_tps:.1f}")
    
    def _validate_production_readiness(self, metrics: ModelMetrics) -> bool:
        """Valida se o modelo está pronto para produção"""
        thresholds = self.performance_thresholds
        
        checks = [
            metrics.accuracy >= thresholds['accuracy_min'],
            metrics.precision >= thresholds['precision_min'],
            metrics.recall >= thresholds['recall_min'],
            metrics.f1_score >= thresholds['f1_score_min'],
            metrics.throughput_tps >= thresholds['throughput_min'],
            metrics.latency_p95_ms <= thresholds['latency_max']
        ]
        
        passed_checks = sum(checks)
        total_checks = len(checks)
        
        logger.info(f"🔍 Validação de produção: {passed_checks}/{total_checks} critérios atendidos")
        
        return passed_checks >= (total_checks * 0.8)  # 80% dos critérios devem ser atendidos
    
    def _check_retraining_triggers(self, current_metrics: ModelMetrics):
        """Verifica se algum gatilho de retreinamento foi acionado"""
        thresholds = self.performance_thresholds
        triggers = []
        
        # Verificar degradação de performance
        if current_metrics.accuracy < thresholds['accuracy_min']:
            triggers.append(RetrainingTrigger(
                trigger_type='performance_degradation',
                threshold_value=thresholds['accuracy_min'],
                current_value=current_metrics.accuracy,
                triggered_at=datetime.now().isoformat(),
                severity='high' if current_metrics.accuracy < 0.7 else 'medium'
            ))
        
        if current_metrics.precision < thresholds['precision_min']:
            triggers.append(RetrainingTrigger(
                trigger_type='performance_degradation',
                threshold_value=thresholds['precision_min'],
                current_value=current_metrics.precision,
                triggered_at=datetime.now().isoformat(),
                severity='high' if current_metrics.precision < 0.6 else 'medium'
            ))
        
        if current_metrics.recall < thresholds['recall_min']:
            triggers.append(RetrainingTrigger(
                trigger_type='performance_degradation',
                threshold_value=thresholds['recall_min'],
                current_value=current_metrics.recall,
                triggered_at=datetime.now().isoformat(),
                severity='critical' if current_metrics.recall < 0.5 else 'high'
            ))
        
        # Verificar drift de dados
        if current_metrics.data_drift_score > thresholds['data_drift_max']:
            triggers.append(RetrainingTrigger(
                trigger_type='data_drift',
                threshold_value=thresholds['data_drift_max'],
                current_value=current_metrics.data_drift_score,
                triggered_at=datetime.now().isoformat(),
                severity='medium'
            ))
        
        # Verificar drift de conceito
        if current_metrics.concept_drift_score > thresholds['concept_drift_max']:
            triggers.append(RetrainingTrigger(
                trigger_type='concept_drift',
                threshold_value=thresholds['concept_drift_max'],
                current_value=current_metrics.concept_drift_score,
                triggered_at=datetime.now().isoformat(),
                severity='high'
            ))
        
        # Adicionar triggers à lista
        self.retraining_triggers.extend(triggers)
        
        # Log triggers
        for trigger in triggers:
            logger.warning(f"⚠️ Gatilho de retreinamento: {trigger.trigger_type} "
                         f"({trigger.severity}) - {trigger.current_value:.3f} vs "
                         f"{trigger.threshold_value:.3f}")
        
        # Iniciar retreinamento automático se habilitado
        if triggers and self.retraining_config['auto_retrain_enabled']:
            critical_triggers = [t for t in triggers if t.severity == 'critical']
            high_triggers = [t for t in triggers if t.severity == 'high']
            
            if critical_triggers or len(high_triggers) >= 2:
                logger.warning("🔄 Iniciando retreinamento automático devido a triggers críticos")
                self._trigger_automatic_retraining()
    
    def _trigger_automatic_retraining(self):
        """Inicia o processo de retreinamento automático"""
        logger.info("🤖 Retreinamento automático iniciado")
        
        # Em um ambiente real, isso dispararia um pipeline de retreinamento
        # Por enquanto, apenas logamos a ação
        retraining_job = {
            'job_id': f"retrain_{int(time.time())}",
            'triggered_at': datetime.now().isoformat(),
            'triggers': [asdict(t) for t in self.retraining_triggers[-5:]],  # Últimos 5 triggers
            'status': 'queued'
        }
        
        # Salvar job de retreinamento
        job_path = os.path.join(self.metrics_dir, f"retraining_job_{retraining_job['job_id']}.json")
        with open(job_path, 'w') as f:
            json.dump(retraining_job, f, indent=2)
        
        logger.info(f"📋 Job de retreinamento criado: {retraining_job['job_id']}")
    
    def get_model_performance_summary(self, model_name: str = None) -> Dict[str, Any]:
        """Retorna resumo de performance dos modelos"""
        if not self.performance_history:
            return {"error": "Nenhuma métrica de performance disponível"}
        
        recent_metrics = self.performance_history[-10:]  # Últimas 10 medições
        
        summary = {
            "current_models": {
                name: {
                    "version": model.version,
                    "status": model.status,
                    "accuracy": model.performance_metrics.accuracy,
                    "precision": model.performance_metrics.precision,
                    "recall": model.performance_metrics.recall,
                    "f1_score": model.performance_metrics.f1_score
                }
                for name, model in self.current_production_models.items()
            },
            "recent_performance": {
                "avg_accuracy": np.mean([m.accuracy for m in recent_metrics]),
                "avg_precision": np.mean([m.precision for m in recent_metrics]),
                "avg_recall": np.mean([m.recall for m in recent_metrics]),
                "avg_f1_score": np.mean([m.f1_score for m in recent_metrics]),
                "avg_throughput": np.mean([m.throughput_tps for m in recent_metrics]),
                "avg_latency": np.mean([m.latency_p95_ms for m in recent_metrics])
            },
            "active_triggers": len([t for t in self.retraining_triggers 
                                  if datetime.fromisoformat(t.triggered_at) > 
                                  datetime.now() - timedelta(hours=1)]),
            "total_model_versions": sum(len(versions) for versions in self.model_versions.values()),
            "last_updated": datetime.now().isoformat()
        }
        
        return summary
    
    def start_monitoring(self, interval_seconds: int = 300):
        """Inicia o monitoramento contínuo dos modelos"""
        if self.is_monitoring:
            logger.warning("⚠️ Monitoramento já está ativo")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"👁️ Monitoramento iniciado (intervalo: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Para o monitoramento contínuo"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        logger.info("🛑 Monitoramento parado")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Loop principal de monitoramento"""
        while self.is_monitoring:
            try:
                # Verificar saúde dos modelos em produção
                for model_name, model_version in self.current_production_models.items():
                    self._check_model_health(model_name, model_version)
                
                # Limpar triggers antigos (mais de 24h)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.retraining_triggers = [
                    t for t in self.retraining_triggers 
                    if datetime.fromisoformat(t.triggered_at) > cutoff_time
                ]
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"❌ Erro no loop de monitoramento: {e}")
                time.sleep(60)  # Aguardar 1 minuto antes de tentar novamente
    
    def _check_model_health(self, model_name: str, model_version: ModelVersion):
        """Verifica a saúde de um modelo específico"""
        # Verificar se o arquivo do modelo ainda existe
        if not os.path.exists(model_version.model_path):
            logger.error(f"❌ Arquivo do modelo {model_name} {model_version.version} não encontrado")
            return
        
        # Verificar idade do modelo
        created_at = datetime.fromisoformat(model_version.created_at)
        age_hours = (datetime.now() - created_at).total_seconds() / 3600
        
        if age_hours > 24 * 7:  # Mais de 7 dias
            logger.warning(f"⏰ Modelo {model_name} {model_version.version} está antigo ({age_hours:.1f}h)")
    
    def _save_model_registry(self):
        """Salva o registro de modelos"""
        registry_path = os.path.join(self.models_dir, "model_registry.json")
        
        registry_data = {
            "model_versions": {
                name: [asdict(mv) for mv in versions]
                for name, versions in self.model_versions.items()
            },
            "current_production_models": {
                name: asdict(model)
                for name, model in self.current_production_models.items()
            },
            "last_updated": datetime.now().isoformat()
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def _save_performance_metrics(self, metrics: ModelMetrics):
        """Salva métricas de performance"""
        metrics_file = os.path.join(
            self.metrics_dir, 
            f"performance_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')

# Instância global do gerenciador
model_lifecycle_manager = ModelLifecycleManager()

if __name__ == "__main__":
    # Teste do sistema de MLOps
    manager = ModelLifecycleManager()
    
    # Simular registro de modelo
    test_metrics = ModelMetrics(
        accuracy=0.89,
        precision=0.85,
        recall=0.92,
        f1_score=0.88,
        auc_roc=0.94,
        throughput_tps=150.0,
        latency_p95_ms=25.0,
        timestamp=datetime.now().isoformat(),
        model_version="v1",
        data_drift_score=0.1,
        concept_drift_score=0.05
    )
    
    version = manager.register_model_version(
        "fraud_detector", 
        "models/fraud_detector_v1.pkl",
        "abc123",
        test_metrics
    )
    
    # Promover para produção
    manager.promote_to_production("fraud_detector", version)
    
    # Simular degradação de performance
    degraded_metrics = ModelMetrics(
        accuracy=0.70,  # Abaixo do threshold
        precision=0.65,
        recall=0.95,
        f1_score=0.77,
        auc_roc=0.82,
        throughput_tps=120.0,
        latency_p95_ms=35.0,
        timestamp=datetime.now().isoformat(),
        model_version=version,
        data_drift_score=0.4,  # Acima do threshold
        concept_drift_score=0.15
    )
    
    manager.record_performance_metrics(degraded_metrics)
    
    # Obter resumo
    summary = manager.get_model_performance_summary()
    print("📊 Resumo de Performance:")
    print(json.dumps(summary, indent=2))
    
    print("\n🔄 Sistema de MLOps testado com sucesso!")
