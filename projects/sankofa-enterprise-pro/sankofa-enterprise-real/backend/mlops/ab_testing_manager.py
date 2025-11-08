#!/usr/bin/env python3
"""
Sistema de A/B Testing para Modelos de Detecção de Fraude
Sankofa Enterprise Pro - A/B Testing Manager
"""

import logging
import json
import hashlib
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """Status dos testes A/B"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class TrafficSplitStrategy(Enum):
    """Estratégias de divisão de tráfego"""
    RANDOM = "random"
    HASH_BASED = "hash_based"
    GEOGRAPHIC = "geographic"
    TIME_BASED = "time_based"
    RISK_BASED = "risk_based"

@dataclass
class ModelVariant:
    """Variante de modelo para teste A/B"""
    variant_id: str
    model_name: str
    model_version: str
    model_path: str
    traffic_percentage: float
    description: str
    is_control: bool = False
    metadata: Dict[str, Any] = None

@dataclass
class ABTestConfig:
    """Configuração de teste A/B"""
    test_id: str
    test_name: str
    description: str
    variants: List[ModelVariant]
    traffic_split_strategy: TrafficSplitStrategy
    start_date: str
    end_date: str
    success_metrics: List[str]
    minimum_sample_size: int
    confidence_level: float
    status: TestStatus
    created_by: str
    created_at: str
    metadata: Dict[str, Any] = None

@dataclass
class ABTestResult:
    """Resultado de uma transação no teste A/B"""
    test_id: str
    transaction_id: str
    variant_id: str
    model_prediction: float
    model_decision: str
    actual_label: Optional[int]
    processing_time_ms: float
    timestamp: str
    metadata: Dict[str, Any] = None

@dataclass
class ABTestMetrics:
    """Métricas de um teste A/B"""
    test_id: str
    variant_id: str
    total_transactions: int
    fraud_detected: int
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_processing_time_ms: float
    confidence_interval: Tuple[float, float]
    statistical_significance: bool
    calculated_at: str

class ABTestingManager:
    """Gerenciador de testes A/B para modelos de detecção de fraude"""
    
    def __init__(self, storage_path: str = "/home/ubuntu/sankofa-enterprise-real/data/ab_testing"):
        self.storage_path = storage_path
        self.tests_file = os.path.join(storage_path, "ab_tests.json")
        self.results_file = os.path.join(storage_path, "ab_results.csv")
        self.metrics_file = os.path.join(storage_path, "ab_metrics.json")
        
        # Criar diretório se não existir
        os.makedirs(storage_path, exist_ok=True)
        
        # Inicializar arquivos
        self._initialize_storage()
        
        # Cache de testes ativos
        self.active_tests = {}
        self._load_active_tests()
        
        logger.info("🧪 A/B Testing Manager inicializado")
    
    def _initialize_storage(self):
        """Inicializa arquivos de armazenamento"""
        # Arquivo de testes
        if not os.path.exists(self.tests_file):
            with open(self.tests_file, 'w') as f:
                json.dump({}, f)
        
        # Arquivo de resultados
        if not os.path.exists(self.results_file):
            results_df = pd.DataFrame(columns=[
                'test_id', 'transaction_id', 'variant_id', 'model_prediction',
                'model_decision', 'actual_label', 'processing_time_ms', 'timestamp'
            ])
            results_df.to_csv(self.results_file, index=False)
        
        # Arquivo de métricas
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w') as f:
                json.dump({}, f)
    
    def _load_active_tests(self):
        """Carrega testes ativos na memória"""
        try:
            with open(self.tests_file, 'r') as f:
                all_tests = json.load(f)
            
            self.active_tests = {
                test_id: test_config for test_id, test_config in all_tests.items()
                if test_config.get('status') == TestStatus.ACTIVE.value
            }
            
            logger.info(f"📊 {len(self.active_tests)} testes A/B ativos carregados")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar testes ativos: {e}")
            self.active_tests = {}
    
    def create_ab_test(self, config: ABTestConfig) -> bool:
        """Cria um novo teste A/B"""
        try:
            # Validar configuração
            if not self._validate_test_config(config):
                return False
            
            # Carregar testes existentes
            with open(self.tests_file, 'r') as f:
                tests = json.load(f)
            
            # Adicionar novo teste
            tests[config.test_id] = asdict(config)
            
            # Salvar
            with open(self.tests_file, 'w') as f:
                json.dump(tests, f, indent=2, default=str)
            
            # Atualizar cache se o teste estiver ativo
            if config.status == TestStatus.ACTIVE:
                self.active_tests[config.test_id] = asdict(config)
            
            logger.info(f"✅ Teste A/B criado: {config.test_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar teste A/B: {e}")
            return False
    
    def _validate_test_config(self, config: ABTestConfig) -> bool:
        """Valida configuração do teste A/B"""
        # Verificar se soma das porcentagens é 100%
        total_percentage = sum(variant.traffic_percentage for variant in config.variants)
        if abs(total_percentage - 100.0) > 0.01:
            logger.error(f"❌ Soma das porcentagens de tráfego deve ser 100%, atual: {total_percentage}%")
            return False
        
        # Verificar se há pelo menos uma variante de controle
        control_variants = [v for v in config.variants if v.is_control]
        if len(control_variants) != 1:
            logger.error(f"❌ Deve haver exatamente uma variante de controle, encontradas: {len(control_variants)}")
            return False
        
        # Verificar se as datas são válidas
        try:
            start_date = datetime.fromisoformat(config.start_date)
            end_date = datetime.fromisoformat(config.end_date)
            if end_date <= start_date:
                logger.error("❌ Data de fim deve ser posterior à data de início")
                return False
        except ValueError as e:
            logger.error(f"❌ Formato de data inválido: {e}")
            return False
        
        return True
    
    def get_variant_for_transaction(self, transaction_data: Dict[str, Any]) -> Optional[str]:
        """Determina qual variante usar para uma transação"""
        try:
            # Verificar se há testes ativos
            if not self.active_tests:
                return None
            
            # Para simplicidade, usar o primeiro teste ativo
            # Em produção, poderia haver lógica mais complexa para múltiplos testes
            test_id = list(self.active_tests.keys())[0]
            test_config = self.active_tests[test_id]
            
            # Verificar se o teste ainda está no período válido
            now = datetime.now()
            start_date = datetime.fromisoformat(test_config['start_date'])
            end_date = datetime.fromisoformat(test_config['end_date'])
            
            if not (start_date <= now <= end_date):
                return None
            
            # Determinar variante baseada na estratégia
            strategy = TrafficSplitStrategy(test_config['traffic_split_strategy'])
            variant_id = self._select_variant(transaction_data, test_config, strategy)
            
            return variant_id
            
        except Exception as e:
            logger.error(f"❌ Erro ao determinar variante: {e}")
            return None
    
    def _select_variant(self, transaction_data: Dict[str, Any], test_config: Dict, strategy: TrafficSplitStrategy) -> str:
        """Seleciona variante baseada na estratégia"""
        variants = test_config['variants']
        
        if strategy == TrafficSplitStrategy.RANDOM:
            return self._random_selection(variants)
        
        elif strategy == TrafficSplitStrategy.HASH_BASED:
            return self._hash_based_selection(transaction_data, variants)
        
        elif strategy == TrafficSplitStrategy.RISK_BASED:
            return self._risk_based_selection(transaction_data, variants)
        
        else:
            # Fallback para seleção aleatória
            return self._random_selection(variants)
    
    def _random_selection(self, variants: List[Dict]) -> str:
        """Seleção aleatória baseada nas porcentagens"""
        rand = random.random() * 100
        cumulative = 0
        
        for variant in variants:
            cumulative += variant['traffic_percentage']
            if rand <= cumulative:
                return variant['variant_id']
        
        # Fallback para a primeira variante
        return variants[0]['variant_id']
    
    def _hash_based_selection(self, transaction_data: Dict[str, Any], variants: List[Dict]) -> str:
        """Seleção baseada em hash para consistência"""
        # Usar ID da transação ou CPF para hash consistente
        hash_key = transaction_data.get('id', transaction_data.get('cliente_cpf', 'default'))
        hash_value = int(hashlib.md5(str(hash_key).encode()).hexdigest(), 16) % 100
        
        cumulative = 0
        for variant in variants:
            cumulative += variant['traffic_percentage']
            if hash_value < cumulative:
                return variant['variant_id']
        
        return variants[0]['variant_id']
    
    def _risk_based_selection(self, transaction_data: Dict[str, Any], variants: List[Dict]) -> str:
        """Seleção baseada no risco da transação"""
        # Calcular score de risco simples
        amount = float(transaction_data.get('valor', 0))
        risk_score = min(amount / 10000, 1.0)  # Normalizar para 0-1
        
        # Transações de alto risco vão para variante de controle
        if risk_score > 0.8:
            control_variant = next(v for v in variants if v['is_control'])
            return control_variant['variant_id']
        
        # Outras transações seguem distribuição normal
        return self._random_selection(variants)
    
    def record_ab_result(self, result: ABTestResult) -> bool:
        """Registra resultado de teste A/B"""
        try:
            # Carregar resultados existentes
            results_df = pd.read_csv(self.results_file)
            
            # Adicionar novo resultado
            new_result = pd.DataFrame([asdict(result)])
            results_df = pd.concat([results_df, new_result], ignore_index=True)
            
            # Salvar
            results_df.to_csv(self.results_file, index=False)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao registrar resultado A/B: {e}")
            return False
    
    def calculate_test_metrics(self, test_id: str) -> Dict[str, ABTestMetrics]:
        """Calcula métricas para todas as variantes de um teste"""
        try:
            # Carregar resultados
            results_df = pd.read_csv(self.results_file)
            test_results = results_df[results_df['test_id'] == test_id]
            
            if test_results.empty:
                logger.warning(f"⚠️ Nenhum resultado encontrado para teste {test_id}")
                return {}
            
            metrics = {}
            
            # Calcular métricas para cada variante
            for variant_id in test_results['variant_id'].unique():
                variant_results = test_results[test_results['variant_id'] == variant_id]
                variant_metrics = self._calculate_variant_metrics(test_id, variant_id, variant_results)
                metrics[variant_id] = variant_metrics
            
            # Salvar métricas
            self._save_metrics(test_id, metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Erro ao calcular métricas do teste: {e}")
            return {}
    
    def _calculate_variant_metrics(self, test_id: str, variant_id: str, results_df: pd.DataFrame) -> ABTestMetrics:
        """Calcula métricas para uma variante específica"""
        total_transactions = len(results_df)
        
        # Filtrar apenas resultados com labels conhecidos
        labeled_results = results_df.dropna(subset=['actual_label'])
        
        if labeled_results.empty:
            # Retornar métricas vazias se não há labels
            return ABTestMetrics(
                test_id=test_id,
                variant_id=variant_id,
                total_transactions=total_transactions,
                fraud_detected=0,
                false_positives=0,
                false_negatives=0,
                true_positives=0,
                true_negatives=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                avg_processing_time_ms=results_df['processing_time_ms'].mean(),
                confidence_interval=(0.0, 0.0),
                statistical_significance=False,
                calculated_at=datetime.now().isoformat()
            )
        
        # Calcular matriz de confusão
        true_positives = len(labeled_results[
            (labeled_results['model_decision'] == 'block') & (labeled_results['actual_label'] == 1)
        ])
        true_negatives = len(labeled_results[
            (labeled_results['model_decision'] == 'approve') & (labeled_results['actual_label'] == 0)
        ])
        false_positives = len(labeled_results[
            (labeled_results['model_decision'] == 'block') & (labeled_results['actual_label'] == 0)
        ])
        false_negatives = len(labeled_results[
            (labeled_results['model_decision'] == 'approve') & (labeled_results['actual_label'] == 1)
        ])
        
        # Calcular métricas
        accuracy = (true_positives + true_negatives) / len(labeled_results) if len(labeled_results) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calcular intervalo de confiança para accuracy (aproximação simples)
        if len(labeled_results) > 0:
            std_error = np.sqrt(accuracy * (1 - accuracy) / len(labeled_results))
            confidence_interval = (
                max(0, accuracy - 1.96 * std_error),
                min(1, accuracy + 1.96 * std_error)
            )
        else:
            confidence_interval = (0.0, 0.0)
        
        # Determinar significância estatística (simplificado)
        statistical_significance = len(labeled_results) >= 1000 and abs(accuracy - 0.5) > 0.05
        
        return ABTestMetrics(
            test_id=test_id,
            variant_id=variant_id,
            total_transactions=total_transactions,
            fraud_detected=len(results_df[results_df['model_decision'] == 'block']),
            false_positives=false_positives,
            false_negatives=false_negatives,
            true_positives=true_positives,
            true_negatives=true_negatives,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            avg_processing_time_ms=results_df['processing_time_ms'].mean(),
            confidence_interval=confidence_interval,
            statistical_significance=statistical_significance,
            calculated_at=datetime.now().isoformat()
        )
    
    def _save_metrics(self, test_id: str, metrics: Dict[str, ABTestMetrics]):
        """Salva métricas calculadas"""
        try:
            # Carregar métricas existentes
            with open(self.metrics_file, 'r') as f:
                all_metrics = json.load(f)
            
            # Converter métricas para dict
            metrics_dict = {variant_id: asdict(metric) for variant_id, metric in metrics.items()}
            all_metrics[test_id] = metrics_dict
            
            # Salvar
            with open(self.metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"❌ Erro ao salvar métricas: {e}")
    
    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """Obtém status completo de um teste A/B"""
        try:
            # Carregar configuração do teste
            with open(self.tests_file, 'r') as f:
                tests = json.load(f)
            
            if test_id not in tests:
                return {'error': f'Teste {test_id} não encontrado'}
            
            test_config = tests[test_id]
            
            # Calcular métricas atuais
            metrics = self.calculate_test_metrics(test_id)
            
            # Carregar resultados para estatísticas
            results_df = pd.read_csv(self.results_file)
            test_results = results_df[results_df['test_id'] == test_id]
            
            return {
                'test_config': test_config,
                'metrics': {variant_id: asdict(metric) for variant_id, metric in metrics.items()},
                'total_transactions': len(test_results),
                'variants_performance': {
                    variant_id: {
                        'transactions': len(test_results[test_results['variant_id'] == variant_id]),
                        'avg_processing_time': test_results[test_results['variant_id'] == variant_id]['processing_time_ms'].mean()
                    }
                    for variant_id in test_results['variant_id'].unique()
                },
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter status do teste: {e}")
            return {'error': str(e)}
    
    def stop_test(self, test_id: str, reason: str = "Manual stop") -> bool:
        """Para um teste A/B"""
        try:
            # Carregar testes
            with open(self.tests_file, 'r') as f:
                tests = json.load(f)
            
            if test_id not in tests:
                logger.error(f"❌ Teste {test_id} não encontrado")
                return False
            
            # Atualizar status
            tests[test_id]['status'] = TestStatus.COMPLETED.value
            tests[test_id]['stopped_at'] = datetime.now().isoformat()
            tests[test_id]['stop_reason'] = reason
            
            # Salvar
            with open(self.tests_file, 'w') as f:
                json.dump(tests, f, indent=2, default=str)
            
            # Remover do cache de testes ativos
            if test_id in self.active_tests:
                del self.active_tests[test_id]
            
            logger.info(f"🛑 Teste A/B parado: {test_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao parar teste: {e}")
            return False
    
    def list_tests(self, status_filter: Optional[TestStatus] = None) -> List[Dict[str, Any]]:
        """Lista todos os testes A/B"""
        try:
            with open(self.tests_file, 'r') as f:
                tests = json.load(f)
            
            if status_filter:
                filtered_tests = {
                    test_id: test_config for test_id, test_config in tests.items()
                    if test_config.get('status') == status_filter.value
                }
            else:
                filtered_tests = tests
            
            return list(filtered_tests.values())
            
        except Exception as e:
            logger.error(f"❌ Erro ao listar testes: {e}")
            return []

