#!/usr/bin/env python3
"""
Sistema de Canary Deployment para Modelos de Detecção de Fraude
Sankofa Enterprise Pro - Canary Deployment Manager
"""

import logging
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    """Status do deployment canary"""
    PENDING = "pending"
    STARTING = "starting"
    ACTIVE = "active"
    PROMOTING = "promoting"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"

class HealthCheckStatus(Enum):
    """Status dos health checks"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class CanaryConfig:
    """Configuração de deployment canary"""
    deployment_id: str
    model_name: str
    current_version: str
    canary_version: str
    initial_traffic_percentage: float
    target_traffic_percentage: float
    promotion_steps: List[float]  # Porcentagens de tráfego para cada step
    step_duration_minutes: int
    success_criteria: Dict[str, float]
    rollback_criteria: Dict[str, float]
    health_check_interval_seconds: int
    created_by: str
    created_at: str
    metadata: Dict[str, Any] = None

@dataclass
class DeploymentStep:
    """Step de um deployment canary"""
    step_number: int
    traffic_percentage: float
    started_at: str
    completed_at: Optional[str]
    status: str
    metrics: Dict[str, float]
    health_status: HealthCheckStatus

@dataclass
class CanaryMetrics:
    """Métricas de um deployment canary"""
    deployment_id: str
    version: str
    traffic_percentage: float
    total_requests: int
    successful_requests: int
    error_rate: float
    avg_response_time_ms: float
    fraud_detection_rate: float
    false_positive_rate: float
    false_negative_rate: float
    accuracy: float
    precision: float
    recall: float
    timestamp: str

@dataclass
class CanaryDeployment:
    """Deployment canary completo"""
    config: CanaryConfig
    status: DeploymentStatus
    current_step: int
    steps: List[DeploymentStep]
    current_metrics: Dict[str, CanaryMetrics]
    alerts: List[Dict[str, Any]]
    started_at: Optional[str]
    completed_at: Optional[str]
    rollback_reason: Optional[str]

class CanaryDeploymentManager:
    """Gerenciador de deployments canary"""
    
    def __init__(self, storage_path: str = "/home/ubuntu/sankofa-enterprise-real/data/canary_deployments"):
        self.storage_path = storage_path
        self.deployments_file = os.path.join(storage_path, "canary_deployments.json")
        self.metrics_file = os.path.join(storage_path, "canary_metrics.csv")
        
        # Criar diretório se não existir
        os.makedirs(storage_path, exist_ok=True)
        
        # Inicializar arquivos
        self._initialize_storage()
        
        # Deployments ativos
        self.active_deployments = {}
        self._load_active_deployments()
        
        # Thread de monitoramento
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Callbacks para health checks
        self.health_check_callbacks = {}
        
        logger.info("🚀 Canary Deployment Manager inicializado")
    
    def _initialize_storage(self):
        """Inicializa arquivos de armazenamento"""
        if not os.path.exists(self.deployments_file):
            with open(self.deployments_file, 'w') as f:
                json.dump({}, f)
        
        if not os.path.exists(self.metrics_file):
            metrics_df = pd.DataFrame(columns=[
                'deployment_id', 'version', 'traffic_percentage', 'total_requests',
                'successful_requests', 'error_rate', 'avg_response_time_ms',
                'fraud_detection_rate', 'false_positive_rate', 'false_negative_rate',
                'accuracy', 'precision', 'recall', 'timestamp'
            ])
            metrics_df.to_csv(self.metrics_file, index=False)
    
    def _load_active_deployments(self):
        """Carrega deployments ativos"""
        try:
            with open(self.deployments_file, 'r') as f:
                all_deployments = json.load(f)
            
            active_statuses = [DeploymentStatus.ACTIVE.value, DeploymentStatus.STARTING.value, 
                             DeploymentStatus.PROMOTING.value]
            
            self.active_deployments = {
                dep_id: dep_data for dep_id, dep_data in all_deployments.items()
                if dep_data.get('status') in active_statuses
            }
            
            logger.info(f"🚀 {len(self.active_deployments)} deployments canary ativos carregados")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar deployments ativos: {e}")
            self.active_deployments = {}
    
    def create_canary_deployment(self, config: CanaryConfig) -> bool:
        """Cria um novo deployment canary"""
        try:
            # Validar configuração
            if not self._validate_canary_config(config):
                return False
            
            # Criar deployment
            deployment = CanaryDeployment(
                config=config,
                status=DeploymentStatus.PENDING,
                current_step=0,
                steps=[],
                current_metrics={},
                alerts=[],
                started_at=None,
                completed_at=None,
                rollback_reason=None
            )
            
            # Salvar
            self._save_deployment(deployment)
            
            logger.info(f"✅ Deployment canary criado: {config.deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar deployment canary: {e}")
            return False
    
    def _validate_canary_config(self, config: CanaryConfig) -> bool:
        """Valida configuração do deployment canary"""
        # Verificar porcentagens
        if not (0 < config.initial_traffic_percentage <= 100):
            logger.error("❌ Porcentagem inicial de tráfego deve estar entre 0 e 100")
            return False
        
        if not (0 < config.target_traffic_percentage <= 100):
            logger.error("❌ Porcentagem alvo de tráfego deve estar entre 0 e 100")
            return False
        
        # Verificar steps de promoção
        if not config.promotion_steps:
            logger.error("❌ Deve haver pelo menos um step de promoção")
            return False
        
        if max(config.promotion_steps) != config.target_traffic_percentage:
            logger.error("❌ Último step deve ser igual à porcentagem alvo")
            return False
        
        # Verificar critérios
        if not config.success_criteria or not config.rollback_criteria:
            logger.error("❌ Critérios de sucesso e rollback são obrigatórios")
            return False
        
        return True
    
    def start_canary_deployment(self, deployment_id: str) -> bool:
        """Inicia um deployment canary"""
        try:
            deployment = self._load_deployment(deployment_id)
            if not deployment:
                return False
            
            if deployment.status != DeploymentStatus.PENDING:
                logger.error(f"❌ Deployment {deployment_id} não está em status PENDING")
                return False
            
            # Atualizar status
            deployment.status = DeploymentStatus.STARTING
            deployment.started_at = datetime.now().isoformat()
            
            # Criar primeiro step
            first_step = DeploymentStep(
                step_number=1,
                traffic_percentage=deployment.config.initial_traffic_percentage,
                started_at=datetime.now().isoformat(),
                completed_at=None,
                status="active",
                metrics={},
                health_status=HealthCheckStatus.UNKNOWN
            )
            
            deployment.steps = [first_step]
            deployment.current_step = 1
            deployment.status = DeploymentStatus.ACTIVE
            
            # Salvar
            self._save_deployment(deployment)
            
            # Adicionar ao cache de ativos
            self.active_deployments[deployment_id] = asdict(deployment)
            
            # Iniciar monitoramento se não estiver ativo
            if not self.monitoring_active:
                self.start_monitoring()
            
            logger.info(f"🚀 Deployment canary iniciado: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao iniciar deployment canary: {e}")
            return False
    
    def start_monitoring(self):
        """Inicia monitoramento de deployments canary"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("👁️ Monitoramento de deployments canary iniciado")
    
    def stop_monitoring(self):
        """Para monitoramento de deployments canary"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=30)
        
        logger.info("🛑 Monitoramento de deployments canary parado")
    
    def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        while self.monitoring_active:
            try:
                # Recarregar deployments ativos
                self._load_active_deployments()
                
                # Monitorar cada deployment ativo
                for deployment_id in list(self.active_deployments.keys()):
                    self._monitor_deployment(deployment_id)
                
                # Aguardar próximo ciclo
                time.sleep(30)  # Verificar a cada 30 segundos
                
            except Exception as e:
                logger.error(f"❌ Erro no loop de monitoramento: {e}")
                time.sleep(60)  # Aguardar mais tempo em caso de erro
    
    def _monitor_deployment(self, deployment_id: str):
        """Monitora um deployment específico"""
        try:
            deployment = self._load_deployment(deployment_id)
            if not deployment or deployment.status not in [DeploymentStatus.ACTIVE, DeploymentStatus.PROMOTING]:
                return
            
            # Coletar métricas atuais
            current_metrics = self._collect_deployment_metrics(deployment)
            
            # Atualizar métricas do deployment
            deployment.current_metrics = current_metrics
            
            # Verificar health checks
            health_status = self._perform_health_checks(deployment)
            
            # Atualizar step atual
            if deployment.steps:
                current_step = deployment.steps[deployment.current_step - 1]
                current_step.health_status = health_status
                current_step.metrics = {
                    version: asdict(metrics) for version, metrics in current_metrics.items()
                }
            
            # Verificar critérios de rollback
            if self._should_rollback(deployment, current_metrics):
                self._initiate_rollback(deployment, "Critérios de rollback atingidos")
                return
            
            # Verificar se deve promover para próximo step
            if self._should_promote_step(deployment, current_metrics):
                self._promote_to_next_step(deployment)
            
            # Salvar alterações
            self._save_deployment(deployment)
            
        except Exception as e:
            logger.error(f"❌ Erro ao monitorar deployment {deployment_id}: {e}")
    
    def _collect_deployment_metrics(self, deployment: CanaryDeployment) -> Dict[str, CanaryMetrics]:
        """Coleta métricas do deployment"""
        # Simulação de coleta de métricas
        # Em produção, isso seria integrado com sistemas de monitoramento reais
        
        current_time = datetime.now().isoformat()
        metrics = {}
        
        # Métricas da versão atual (controle)
        metrics['current'] = CanaryMetrics(
            deployment_id=deployment.config.deployment_id,
            version=deployment.config.current_version,
            traffic_percentage=100 - deployment.steps[deployment.current_step - 1].traffic_percentage,
            total_requests=np.random.randint(1000, 5000),
            successful_requests=np.random.randint(950, 1000),
            error_rate=np.random.uniform(0.001, 0.01),
            avg_response_time_ms=np.random.uniform(50, 100),
            fraud_detection_rate=np.random.uniform(0.02, 0.05),
            false_positive_rate=np.random.uniform(0.001, 0.005),
            false_negative_rate=np.random.uniform(0.001, 0.003),
            accuracy=np.random.uniform(0.95, 0.98),
            precision=np.random.uniform(0.90, 0.95),
            recall=np.random.uniform(0.85, 0.92),
            timestamp=current_time
        )
        
        # Métricas da versão canary
        metrics['canary'] = CanaryMetrics(
            deployment_id=deployment.config.deployment_id,
            version=deployment.config.canary_version,
            traffic_percentage=deployment.steps[deployment.current_step - 1].traffic_percentage,
            total_requests=np.random.randint(100, 1000),
            successful_requests=np.random.randint(95, 100),
            error_rate=np.random.uniform(0.001, 0.02),
            avg_response_time_ms=np.random.uniform(45, 110),
            fraud_detection_rate=np.random.uniform(0.02, 0.06),
            false_positive_rate=np.random.uniform(0.001, 0.008),
            false_negative_rate=np.random.uniform(0.001, 0.004),
            accuracy=np.random.uniform(0.94, 0.99),
            precision=np.random.uniform(0.88, 0.97),
            recall=np.random.uniform(0.83, 0.95),
            timestamp=current_time
        )
        
        return metrics
    
    def _perform_health_checks(self, deployment: CanaryDeployment) -> HealthCheckStatus:
        """Executa health checks"""
        try:
            # Health check básico baseado nas métricas
            if deployment.current_metrics:
                canary_metrics = deployment.current_metrics.get('canary')
                if canary_metrics:
                    # Verificar taxa de erro
                    if canary_metrics.error_rate > 0.05:  # 5% de erro
                        return HealthCheckStatus.UNHEALTHY
                    elif canary_metrics.error_rate > 0.02:  # 2% de erro
                        return HealthCheckStatus.DEGRADED
                    else:
                        return HealthCheckStatus.HEALTHY
            
            return HealthCheckStatus.UNKNOWN
            
        except Exception as e:
            logger.error(f"❌ Erro no health check: {e}")
            return HealthCheckStatus.UNKNOWN
    
    def _should_rollback(self, deployment: CanaryDeployment, metrics: Dict[str, CanaryMetrics]) -> bool:
        """Verifica se deve fazer rollback"""
        try:
            canary_metrics = metrics.get('canary')
            if not canary_metrics:
                return False
            
            rollback_criteria = deployment.config.rollback_criteria
            
            # Verificar cada critério de rollback
            for metric_name, threshold in rollback_criteria.items():
                metric_value = getattr(canary_metrics, metric_name, None)
                if metric_value is not None:
                    if metric_name in ['error_rate', 'false_positive_rate', 'false_negative_rate']:
                        # Para métricas onde menor é melhor
                        if metric_value > threshold:
                            logger.warning(f"⚠️ Critério de rollback atingido: {metric_name} = {metric_value} > {threshold}")
                            return True
                    else:
                        # Para métricas onde maior é melhor
                        if metric_value < threshold:
                            logger.warning(f"⚠️ Critério de rollback atingido: {metric_name} = {metric_value} < {threshold}")
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erro ao verificar critérios de rollback: {e}")
            return False
    
    def _should_promote_step(self, deployment: CanaryDeployment, metrics: Dict[str, CanaryMetrics]) -> bool:
        """Verifica se deve promover para próximo step"""
        try:
            # Verificar se o step atual já durou tempo suficiente
            current_step = deployment.steps[deployment.current_step - 1]
            step_start = datetime.fromisoformat(current_step.started_at)
            step_duration = datetime.now() - step_start
            
            if step_duration.total_seconds() < deployment.config.step_duration_minutes * 60:
                return False
            
            # Verificar critérios de sucesso
            canary_metrics = metrics.get('canary')
            if not canary_metrics:
                return False
            
            success_criteria = deployment.config.success_criteria
            
            for metric_name, threshold in success_criteria.items():
                metric_value = getattr(canary_metrics, metric_name, None)
                if metric_value is not None:
                    if metric_name in ['error_rate', 'false_positive_rate', 'false_negative_rate']:
                        # Para métricas onde menor é melhor
                        if metric_value > threshold:
                            return False
                    else:
                        # Para métricas onde maior é melhor
                        if metric_value < threshold:
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao verificar critérios de promoção: {e}")
            return False
    
    def _promote_to_next_step(self, deployment: CanaryDeployment):
        """Promove para o próximo step"""
        try:
            # Completar step atual
            current_step = deployment.steps[deployment.current_step - 1]
            current_step.completed_at = datetime.now().isoformat()
            current_step.status = "completed"
            
            # Verificar se há próximo step
            if deployment.current_step >= len(deployment.config.promotion_steps):
                # Deployment completo
                deployment.status = DeploymentStatus.COMPLETED
                deployment.completed_at = datetime.now().isoformat()
                
                # Remover do cache de ativos
                if deployment.config.deployment_id in self.active_deployments:
                    del self.active_deployments[deployment.config.deployment_id]
                
                logger.info(f"✅ Deployment canary completado: {deployment.config.deployment_id}")
                return
            
            # Criar próximo step
            next_traffic_percentage = deployment.config.promotion_steps[deployment.current_step]
            next_step = DeploymentStep(
                step_number=deployment.current_step + 1,
                traffic_percentage=next_traffic_percentage,
                started_at=datetime.now().isoformat(),
                completed_at=None,
                status="active",
                metrics={},
                health_status=HealthCheckStatus.UNKNOWN
            )
            
            deployment.steps.append(next_step)
            deployment.current_step += 1
            deployment.status = DeploymentStatus.PROMOTING
            
            logger.info(f"📈 Deployment {deployment.config.deployment_id} promovido para step {deployment.current_step} ({next_traffic_percentage}% tráfego)")
            
        except Exception as e:
            logger.error(f"❌ Erro ao promover step: {e}")
    
    def _initiate_rollback(self, deployment: CanaryDeployment, reason: str):
        """Inicia rollback do deployment"""
        try:
            deployment.status = DeploymentStatus.ROLLING_BACK
            deployment.rollback_reason = reason
            
            # Completar step atual como falha
            if deployment.steps:
                current_step = deployment.steps[deployment.current_step - 1]
                current_step.completed_at = datetime.now().isoformat()
                current_step.status = "failed"
            
            # Simular rollback (em produção, isso reverteria o tráfego)
            time.sleep(5)
            
            deployment.status = DeploymentStatus.ROLLED_BACK
            deployment.completed_at = datetime.now().isoformat()
            
            # Remover do cache de ativos
            if deployment.config.deployment_id in self.active_deployments:
                del self.active_deployments[deployment.config.deployment_id]
            
            logger.warning(f"🔄 Rollback executado para deployment {deployment.config.deployment_id}: {reason}")
            
        except Exception as e:
            logger.error(f"❌ Erro durante rollback: {e}")
            deployment.status = DeploymentStatus.FAILED
    
    def _load_deployment(self, deployment_id: str) -> Optional[CanaryDeployment]:
        """Carrega um deployment específico"""
        try:
            with open(self.deployments_file, 'r') as f:
                deployments = json.load(f)
            
            if deployment_id not in deployments:
                return None
            
            deployment_data = deployments[deployment_id]
            
            # Reconstruir objeto CanaryDeployment
            config_data = deployment_data['config']
            config = CanaryConfig(**config_data)
            
            steps = [DeploymentStep(**step_data) for step_data in deployment_data.get('steps', [])]
            
            current_metrics = {}
            for version, metrics_data in deployment_data.get('current_metrics', {}).items():
                current_metrics[version] = CanaryMetrics(**metrics_data)
            
            deployment = CanaryDeployment(
                config=config,
                status=DeploymentStatus(deployment_data['status']),
                current_step=deployment_data['current_step'],
                steps=steps,
                current_metrics=current_metrics,
                alerts=deployment_data.get('alerts', []),
                started_at=deployment_data.get('started_at'),
                completed_at=deployment_data.get('completed_at'),
                rollback_reason=deployment_data.get('rollback_reason')
            )
            
            return deployment
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar deployment {deployment_id}: {e}")
            return None
    
    def _save_deployment(self, deployment: CanaryDeployment):
        """Salva um deployment"""
        try:
            # Carregar deployments existentes
            with open(self.deployments_file, 'r') as f:
                deployments = json.load(f)
            
            # Converter para dict
            deployment_dict = asdict(deployment)
            
            # Salvar
            deployments[deployment.config.deployment_id] = deployment_dict
            
            with open(self.deployments_file, 'w') as f:
                json.dump(deployments, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar deployment: {e}")
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Obtém status de um deployment"""
        deployment = self._load_deployment(deployment_id)
        if not deployment:
            return {'error': f'Deployment {deployment_id} não encontrado'}
        
        return asdict(deployment)
    
    def list_deployments(self, status_filter: Optional[DeploymentStatus] = None) -> List[Dict[str, Any]]:
        """Lista deployments"""
        try:
            with open(self.deployments_file, 'r') as f:
                deployments = json.load(f)
            
            if status_filter:
                filtered_deployments = {
                    dep_id: dep_data for dep_id, dep_data in deployments.items()
                    if dep_data.get('status') == status_filter.value
                }
            else:
                filtered_deployments = deployments
            
            return list(filtered_deployments.values())
            
        except Exception as e:
            logger.error(f"❌ Erro ao listar deployments: {e}")
            return []
    
    def force_rollback(self, deployment_id: str, reason: str = "Manual rollback") -> bool:
        """Força rollback de um deployment"""
        try:
            deployment = self._load_deployment(deployment_id)
            if not deployment:
                return False
            
            if deployment.status not in [DeploymentStatus.ACTIVE, DeploymentStatus.PROMOTING]:
                logger.error(f"❌ Deployment {deployment_id} não está em status que permite rollback")
                return False
            
            self._initiate_rollback(deployment, reason)
            self._save_deployment(deployment)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao forçar rollback: {e}")
            return False

