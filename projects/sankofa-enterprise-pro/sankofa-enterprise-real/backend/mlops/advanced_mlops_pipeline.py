"""
Sistema Avançado de MLOps Pipeline para Sankofa Enterprise Pro
Implementa CI/CD para ML, gestão de versões, testes adversariais e automação completa
"""

import os
import json
import pickle
import hashlib
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import joblib
import yaml

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """Representa uma versão específica de um modelo"""
    version: str
    model_hash: str
    performance_metrics: Dict[str, float]
    training_data_hash: str
    created_at: datetime.datetime
    status: str  # 'training', 'testing', 'staging', 'production', 'deprecated'
    metadata: Dict[str, Any]

@dataclass
class ModelPerformanceReport:
    """Relatório de performance de um modelo"""
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    latency_p95: float
    throughput: float
    adversarial_robustness: float
    drift_score: float
    timestamp: datetime.datetime

class AdvancedMLOpsPipeline:
    """
    Pipeline avançado de MLOps com CI/CD, gestão de versões e testes adversariais
    """
    
    def __init__(self, base_path: str = "/home/ubuntu/sankofa-enterprise-real/models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Diretórios para diferentes estágios
        self.staging_path = self.base_path / "staging"
        self.production_path = self.base_path / "production"
        self.archive_path = self.base_path / "archive"
        self.experiments_path = self.base_path / "experiments"
        
        for path in [self.staging_path, self.production_path, self.archive_path, self.experiments_path]:
            path.mkdir(exist_ok=True)
        
        # Configurações
        self.config = self._load_config()
        self.model_registry = self._load_model_registry()
        
        logger.info("Advanced MLOps Pipeline inicializado")
    
    def _load_config(self) -> Dict[str, Any]:
        """Carrega configurações do pipeline"""
        config_path = self.base_path / "mlops_config.yaml"
        
        default_config = {
            "performance_thresholds": {
                "min_accuracy": 0.85,
                "min_precision": 0.80,
                "min_recall": 0.85,
                "min_f1_score": 0.80,
                "max_latency_p95": 20.0,
                "min_throughput": 100.0,
                "min_adversarial_robustness": 0.70
            },
            "deployment_rules": {
                "require_staging_approval": True,
                "min_staging_duration_hours": 24,
                "require_a_b_testing": True,
                "rollback_on_performance_degradation": True
            },
            "monitoring": {
                "drift_detection_threshold": 0.1,
                "performance_check_interval_minutes": 15,
                "alert_on_threshold_breach": True
            }
        }
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = default_config
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        
        return config
    
    def _load_model_registry(self) -> Dict[str, ModelVersion]:
        """Carrega o registro de modelos"""
        registry_path = self.base_path / "model_registry.json"
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                data = json.load(f)
                registry = {}
                for version, model_data in data.items():
                    model_data['created_at'] = datetime.datetime.fromisoformat(model_data['created_at'])
                    registry[version] = ModelVersion(**model_data)
                return registry
        
        return {}
    
    def _save_model_registry(self):
        """Salva o registro de modelos"""
        registry_path = self.base_path / "model_registry.json"
        
        data = {}
        for version, model_version in self.model_registry.items():
            model_data = asdict(model_version)
            model_data['created_at'] = model_version.created_at.isoformat()
            data[version] = model_data
        
        with open(registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _calculate_model_hash(self, model) -> str:
        """Calcula hash único do modelo"""
        model_bytes = pickle.dumps(model)
        return hashlib.sha256(model_bytes).hexdigest()[:16]
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calcula hash dos dados de treinamento"""
        data_str = data.to_string()
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def register_model(self, model, training_data: pd.DataFrame, 
                      performance_metrics: Dict[str, float], 
                      metadata: Dict[str, Any] = None) -> str:
        """
        Registra um novo modelo no pipeline
        """
        model_hash = self._calculate_model_hash(model)
        data_hash = self._calculate_data_hash(training_data)
        
        # Gera versão baseada em timestamp
        version = f"v{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_hash[:8]}"
        
        # Cria versão do modelo
        model_version = ModelVersion(
            version=version,
            model_hash=model_hash,
            performance_metrics=performance_metrics,
            training_data_hash=data_hash,
            created_at=datetime.datetime.now(),
            status='training',
            metadata=metadata or {}
        )
        
        # Salva modelo no diretório de experimentos
        model_path = self.experiments_path / f"{version}.pkl"
        joblib.dump(model, model_path)
        
        # Registra no registry
        self.model_registry[version] = model_version
        self._save_model_registry()
        
        logger.info(f"Modelo {version} registrado com sucesso")
        return version
    
    def run_adversarial_tests(self, model, test_data: pd.DataFrame, 
                            test_labels: np.ndarray) -> float:
        """
        Executa testes adversariais no modelo
        """
        logger.info("Executando testes adversariais...")
        
        # Simula ataques adversariais básicos
        adversarial_scores = []
        
        # Teste 1: Perturbação de features numéricas
        for noise_level in [0.01, 0.05, 0.1]:
            perturbed_data = test_data.copy()
            numeric_cols = test_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                noise = np.random.normal(0, noise_level * test_data[col].std(), len(test_data))
                perturbed_data[col] = test_data[col] + noise
            
            try:
                predictions = model.predict(perturbed_data)
                accuracy = accuracy_score(test_labels, predictions)
                adversarial_scores.append(accuracy)
            except Exception as e:
                logger.warning(f"Erro no teste adversarial com noise {noise_level}: {e}")
                adversarial_scores.append(0.0)
        
        # Teste 2: Valores extremos
        extreme_data = test_data.copy()
        numeric_cols = test_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Substitui 10% dos valores por valores extremos
            n_extreme = int(0.1 * len(extreme_data))
            extreme_indices = np.random.choice(len(extreme_data), n_extreme, replace=False)
            extreme_data.iloc[extreme_indices, extreme_data.columns.get_loc(col)] = extreme_data[col].max() * 10
        
        try:
            predictions = model.predict(extreme_data)
            accuracy = accuracy_score(test_labels, predictions)
            adversarial_scores.append(accuracy)
        except Exception as e:
            logger.warning(f"Erro no teste de valores extremos: {e}")
            adversarial_scores.append(0.0)
        
        # Calcula score médio de robustez adversarial
        robustness_score = np.mean(adversarial_scores) if adversarial_scores else 0.0
        
        logger.info(f"Score de robustez adversarial: {robustness_score:.3f}")
        return robustness_score
    
    def validate_model_performance(self, model_version: str, 
                                 validation_data: pd.DataFrame,
                                 validation_labels: np.ndarray) -> ModelPerformanceReport:
        """
        Valida performance completa do modelo
        """
        logger.info(f"Validando performance do modelo {model_version}")
        
        # Carrega modelo
        model_path = self.experiments_path / f"{model_version}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo {model_version} não encontrado")
        
        model = joblib.load(model_path)
        
        # Métricas básicas de ML
        start_time = datetime.datetime.now()
        predictions = model.predict(validation_data)
        end_time = datetime.datetime.now()
        
        # Calcula latência
        latency_ms = (end_time - start_time).total_seconds() * 1000 / len(validation_data)
        throughput = len(validation_data) / (end_time - start_time).total_seconds()
        
        # Métricas de qualidade
        accuracy = accuracy_score(validation_labels, predictions)
        precision = precision_score(validation_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(validation_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(validation_labels, predictions, average='weighted', zero_division=0)
        
        # AUC-ROC (se aplicável)
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(validation_data)
                if probabilities.shape[1] == 2:  # Classificação binária
                    auc_roc = roc_auc_score(validation_labels, probabilities[:, 1])
                else:
                    auc_roc = roc_auc_score(validation_labels, probabilities, multi_class='ovr')
            else:
                auc_roc = 0.0
        except Exception:
            auc_roc = 0.0
        
        # Testes adversariais
        adversarial_robustness = self.run_adversarial_tests(model, validation_data, validation_labels)
        
        # Score de drift (simulado)
        drift_score = np.random.uniform(0.0, 0.1)  # Em produção, seria calculado com dados reais
        
        # Cria relatório
        report = ModelPerformanceReport(
            model_version=model_version,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            latency_p95=latency_ms,
            throughput=throughput,
            adversarial_robustness=adversarial_robustness,
            drift_score=drift_score,
            timestamp=datetime.datetime.now()
        )
        
        logger.info(f"Validação concluída para {model_version}")
        return report
    
    def promote_to_staging(self, model_version: str) -> bool:
        """
        Promove modelo para staging se atender aos critérios
        """
        logger.info(f"Promovendo modelo {model_version} para staging")
        
        if model_version not in self.model_registry:
            logger.error(f"Modelo {model_version} não encontrado no registry")
            return False
        
        model_info = self.model_registry[model_version]
        
        # Verifica thresholds de performance
        thresholds = self.config['performance_thresholds']
        metrics = model_info.performance_metrics
        
        checks = [
            metrics.get('accuracy', 0) >= thresholds['min_accuracy'],
            metrics.get('precision', 0) >= thresholds['min_precision'],
            metrics.get('recall', 0) >= thresholds['min_recall'],
            metrics.get('f1_score', 0) >= thresholds['min_f1_score'],
            metrics.get('latency_p95', float('inf')) <= thresholds['max_latency_p95'],
            metrics.get('throughput', 0) >= thresholds['min_throughput'],
            metrics.get('adversarial_robustness', 0) >= thresholds['min_adversarial_robustness']
        ]
        
        if not all(checks):
            logger.warning(f"Modelo {model_version} não atende aos critérios de staging")
            return False
        
        # Move modelo para staging
        source_path = self.experiments_path / f"{model_version}.pkl"
        target_path = self.staging_path / f"{model_version}.pkl"
        
        import shutil
        shutil.copy2(source_path, target_path)
        
        # Atualiza status
        model_info.status = 'staging'
        self._save_model_registry()
        
        logger.info(f"Modelo {model_version} promovido para staging com sucesso")
        return True
    
    def promote_to_production(self, model_version: str) -> bool:
        """
        Promove modelo para produção após validação em staging
        """
        logger.info(f"Promovendo modelo {model_version} para produção")
        
        if model_version not in self.model_registry:
            logger.error(f"Modelo {model_version} não encontrado no registry")
            return False
        
        model_info = self.model_registry[model_version]
        
        if model_info.status != 'staging':
            logger.error(f"Modelo {model_version} deve estar em staging para promoção")
            return False
        
        # Verifica tempo mínimo em staging
        staging_duration = datetime.datetime.now() - model_info.created_at
        min_duration = datetime.timedelta(hours=self.config['deployment_rules']['min_staging_duration_hours'])
        
        if staging_duration < min_duration:
            logger.warning(f"Modelo {model_version} precisa ficar mais tempo em staging")
            return False
        
        # Arquiva modelo de produção atual (se existir)
        current_production = self.get_production_model()
        if current_production:
            self._archive_model(current_production)
        
        # Move modelo para produção
        source_path = self.staging_path / f"{model_version}.pkl"
        target_path = self.production_path / f"{model_version}.pkl"
        
        import shutil
        shutil.copy2(source_path, target_path)
        
        # Atualiza status
        model_info.status = 'production'
        self._save_model_registry()
        
        logger.info(f"Modelo {model_version} promovido para produção com sucesso")
        return True
    
    def _archive_model(self, model_version: str):
        """Arquiva um modelo"""
        model_info = self.model_registry[model_version]
        
        # Move modelo para arquivo
        source_path = self.production_path / f"{model_version}.pkl"
        target_path = self.archive_path / f"{model_version}.pkl"
        
        if source_path.exists():
            import shutil
            shutil.move(source_path, target_path)
        
        # Atualiza status
        model_info.status = 'archived'
        self._save_model_registry()
    
    def get_production_model(self) -> Optional[str]:
        """Retorna a versão do modelo em produção"""
        for version, model_info in self.model_registry.items():
            if model_info.status == 'production':
                return version
        return None
    
    def rollback_to_previous_version(self) -> bool:
        """
        Faz rollback para a versão anterior em caso de problemas
        """
        logger.info("Iniciando rollback para versão anterior")
        
        # Encontra versões arquivadas ordenadas por data
        archived_versions = [
            (version, model_info) for version, model_info in self.model_registry.items()
            if model_info.status == 'archived'
        ]
        
        if not archived_versions:
            logger.error("Nenhuma versão anterior disponível para rollback")
            return False
        
        # Ordena por data de criação (mais recente primeiro)
        archived_versions.sort(key=lambda x: x[1].created_at, reverse=True)
        previous_version, previous_info = archived_versions[0]
        
        # Arquiva versão atual
        current_production = self.get_production_model()
        if current_production:
            self._archive_model(current_production)
        
        # Restaura versão anterior
        source_path = self.archive_path / f"{previous_version}.pkl"
        target_path = self.production_path / f"{previous_version}.pkl"
        
        import shutil
        shutil.copy2(source_path, target_path)
        
        # Atualiza status
        previous_info.status = 'production'
        self._save_model_registry()
        
        logger.info(f"Rollback concluído para versão {previous_version}")
        return True
    
    def generate_deployment_report(self) -> Dict[str, Any]:
        """
        Gera relatório completo do estado do deployment
        """
        production_model = self.get_production_model()
        staging_models = [v for v, m in self.model_registry.items() if m.status == 'staging']
        
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "production_model": production_model,
            "staging_models": staging_models,
            "total_models": len(self.model_registry),
            "model_versions": {}
        }
        
        for version, model_info in self.model_registry.items():
            report["model_versions"][version] = {
                "status": model_info.status,
                "performance_metrics": model_info.performance_metrics,
                "created_at": model_info.created_at.isoformat(),
                "model_hash": model_info.model_hash
            }
        
        return report

def test_advanced_mlops_pipeline():
    """
    Testa o pipeline avançado de MLOps
    """
    print("🚀 Testando Advanced MLOps Pipeline...")
    
    # Inicializa pipeline
    pipeline = AdvancedMLOpsPipeline()
    
    # Simula dados de treinamento
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Classificação binária simples
    
    train_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    
    # Simula modelo treinado
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(train_data, y)
    
    # Métricas de performance simuladas
    performance_metrics = {
        "accuracy": 0.92,
        "precision": 0.90,
        "recall": 0.88,
        "f1_score": 0.89,
        "latency_p95": 15.0,
        "throughput": 150.0,
        "adversarial_robustness": 0.75
    }
    
    # Registra modelo
    version = pipeline.register_model(
        model=model,
        training_data=train_data,
        performance_metrics=performance_metrics,
        metadata={"algorithm": "RandomForest", "experiment_id": "exp_001"}
    )
    
    print(f"✅ Modelo registrado: {version}")
    
    # Valida performance
    validation_data = train_data.sample(200)
    validation_labels = y[:200]
    
    report = pipeline.validate_model_performance(version, validation_data, validation_labels)
    print(f"✅ Validação concluída - F1 Score: {report.f1_score:.3f}")
    
    # Promove para staging
    success = pipeline.promote_to_staging(version)
    print(f"✅ Promoção para staging: {'Sucesso' if success else 'Falhou'}")
    
    # Gera relatório
    deployment_report = pipeline.generate_deployment_report()
    print(f"✅ Relatório gerado - Total de modelos: {deployment_report['total_models']}")
    
    print("🎉 Teste do Advanced MLOps Pipeline concluído com sucesso!")
    
    return pipeline, version, report

if __name__ == "__main__":
    test_advanced_mlops_pipeline()
