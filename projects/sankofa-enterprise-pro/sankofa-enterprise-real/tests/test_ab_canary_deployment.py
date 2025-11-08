#!/usr/bin/env python3
"""
Testes para A/B Testing e Canary Deployment
Sankofa Enterprise Pro - A/B Testing & Canary Deployment Tests
"""

import unittest
import tempfile
import os
import json
import time
from datetime import datetime, timedelta
import sys

# Adicionar o diretório raiz ao path
sys.path.append('/home/ubuntu/sankofa-enterprise-real')

from backend.mlops.ab_testing_manager import (
    ABTestingManager, ABTestConfig, ModelVariant, ABTestResult,
    TestStatus, TrafficSplitStrategy
)
from backend.mlops.canary_deployment_manager import (
    CanaryDeploymentManager, CanaryConfig, DeploymentStatus
)

class TestABTesting(unittest.TestCase):
    """Testes para A/B Testing"""
    
    def setUp(self):
        """Configuração inicial dos testes"""
        self.temp_dir = tempfile.mkdtemp()
        self.ab_manager = ABTestingManager(self.temp_dir)
        
        # Configuração de teste padrão
        self.test_config = ABTestConfig(
            test_id="test_fraud_model_v1_vs_v2",
            test_name="Teste Modelo V1 vs V2",
            description="Comparação entre modelo atual e nova versão",
            variants=[
                ModelVariant(
                    variant_id="control",
                    model_name="fraud_model",
                    model_version="v1.0",
                    model_path="/models/fraud_model_v1.pkl",
                    traffic_percentage=70.0,
                    description="Modelo atual em produção",
                    is_control=True
                ),
                ModelVariant(
                    variant_id="treatment",
                    model_name="fraud_model",
                    model_version="v2.0",
                    model_path="/models/fraud_model_v2.pkl",
                    traffic_percentage=30.0,
                    description="Nova versão do modelo",
                    is_control=False
                )
            ],
            traffic_split_strategy=TrafficSplitStrategy.RANDOM,
            start_date=datetime.now().isoformat(),
            end_date=(datetime.now() + timedelta(days=7)).isoformat(),
            success_metrics=["accuracy", "precision", "recall"],
            minimum_sample_size=1000,
            confidence_level=0.95,
            status=TestStatus.DRAFT,
            created_by="test_user",
            created_at=datetime.now().isoformat()
        )
    
    def tearDown(self):
        """Limpeza após os testes"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_ab_test(self):
        """Testa criação de teste A/B"""
        result = self.ab_manager.create_ab_test(self.test_config)
        self.assertTrue(result)
        
        # Verificar se foi salvo
        self.assertTrue(os.path.exists(self.ab_manager.tests_file))
        
        with open(self.ab_manager.tests_file, 'r') as f:
            tests = json.load(f)
        
        self.assertIn(self.test_config.test_id, tests)
        self.assertEqual(tests[self.test_config.test_id]['test_name'], self.test_config.test_name)
    
    def test_invalid_traffic_percentage(self):
        """Testa validação de porcentagem de tráfego inválida"""
        # Configuração com porcentagens que não somam 100%
        invalid_config = self.test_config
        invalid_config.variants[0].traffic_percentage = 50.0
        invalid_config.variants[1].traffic_percentage = 40.0  # Total = 90%
        
        result = self.ab_manager.create_ab_test(invalid_config)
        self.assertFalse(result)
    
    def test_variant_selection_random(self):
        """Testa seleção aleatória de variantes"""
        # Criar teste ativo
        self.test_config.status = TestStatus.ACTIVE
        self.ab_manager.create_ab_test(self.test_config)
        
        # Simular múltiplas seleções
        transaction_data = {'id': 'TXN001', 'valor': 1000}
        selections = []
        
        for _ in range(100):
            variant = self.ab_manager.get_variant_for_transaction(transaction_data)
            if variant:
                selections.append(variant)
        
        # Verificar se ambas as variantes foram selecionadas
        unique_variants = set(selections)
        self.assertGreater(len(unique_variants), 0)
    
    def test_variant_selection_hash_based(self):
        """Testa seleção baseada em hash"""
        # Configurar estratégia hash
        self.test_config.traffic_split_strategy = TrafficSplitStrategy.HASH_BASED
        self.test_config.status = TestStatus.ACTIVE
        self.ab_manager.create_ab_test(self.test_config)
        
        # Mesma transação deve sempre retornar a mesma variante
        transaction_data = {'id': 'TXN001', 'valor': 1000}
        
        variant1 = self.ab_manager.get_variant_for_transaction(transaction_data)
        variant2 = self.ab_manager.get_variant_for_transaction(transaction_data)
        
        self.assertEqual(variant1, variant2)
    
    def test_record_ab_result(self):
        """Testa registro de resultado A/B"""
        result = ABTestResult(
            test_id=self.test_config.test_id,
            transaction_id="TXN001",
            variant_id="control",
            model_prediction=0.8,
            model_decision="block",
            actual_label=1,
            processing_time_ms=50.0,
            timestamp=datetime.now().isoformat()
        )
        
        success = self.ab_manager.record_ab_result(result)
        self.assertTrue(success)
        
        # Verificar se foi salvo
        import pandas as pd
        results_df = pd.read_csv(self.ab_manager.results_file)
        self.assertEqual(len(results_df), 1)
        self.assertEqual(results_df.iloc[0]['transaction_id'], "TXN001")
    
    def test_calculate_metrics(self):
        """Testa cálculo de métricas"""
        # Criar teste
        self.ab_manager.create_ab_test(self.test_config)
        
        # Adicionar alguns resultados
        results = [
            ABTestResult("test1", "TXN001", "control", 0.8, "block", 1, 50.0, datetime.now().isoformat()),
            ABTestResult("test1", "TXN002", "control", 0.2, "approve", 0, 45.0, datetime.now().isoformat()),
            ABTestResult("test1", "TXN003", "treatment", 0.9, "block", 1, 55.0, datetime.now().isoformat()),
            ABTestResult("test1", "TXN004", "treatment", 0.1, "approve", 0, 48.0, datetime.now().isoformat()),
        ]
        
        for result in results:
            self.ab_manager.record_ab_result(result)
        
        # Calcular métricas
        metrics = self.ab_manager.calculate_test_metrics("test1")
        
        # Verificar se métricas foram calculadas para ambas as variantes
        self.assertIn("control", metrics)
        self.assertIn("treatment", metrics)
        
        # Verificar se accuracy foi calculada corretamente (100% para ambas)
        self.assertEqual(metrics["control"].accuracy, 1.0)
        self.assertEqual(metrics["treatment"].accuracy, 1.0)

class TestCanaryDeployment(unittest.TestCase):
    """Testes para Canary Deployment"""
    
    def setUp(self):
        """Configuração inicial dos testes"""
        self.temp_dir = tempfile.mkdtemp()
        self.canary_manager = CanaryDeploymentManager(self.temp_dir)
        
        # Configuração de deployment padrão
        self.canary_config = CanaryConfig(
            deployment_id="canary_fraud_model_v3",
            model_name="fraud_model",
            current_version="v2.0",
            canary_version="v3.0",
            initial_traffic_percentage=5.0,
            target_traffic_percentage=100.0,
            promotion_steps=[5.0, 10.0, 25.0, 50.0, 100.0],
            step_duration_minutes=5,  # 5 minutos para testes
            success_criteria={
                "accuracy": 0.95,
                "precision": 0.90,
                "error_rate": 0.01
            },
            rollback_criteria={
                "error_rate": 0.05,
                "false_positive_rate": 0.10
            },
            health_check_interval_seconds=30,
            created_by="test_user",
            created_at=datetime.now().isoformat()
        )
    
    def tearDown(self):
        """Limpeza após os testes"""
        self.canary_manager.stop_monitoring()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_canary_deployment(self):
        """Testa criação de deployment canary"""
        result = self.canary_manager.create_canary_deployment(self.canary_config)
        self.assertTrue(result)
        
        # Verificar se foi salvo
        self.assertTrue(os.path.exists(self.canary_manager.deployments_file))
        
        with open(self.canary_manager.deployments_file, 'r') as f:
            deployments = json.load(f)
        
        self.assertIn(self.canary_config.deployment_id, deployments)
    
    def test_invalid_canary_config(self):
        """Testa validação de configuração inválida"""
        # Configuração com porcentagem inicial inválida
        invalid_config = self.canary_config
        invalid_config.initial_traffic_percentage = 150.0  # Inválido
        
        result = self.canary_manager.create_canary_deployment(invalid_config)
        self.assertFalse(result)
    
    def test_start_canary_deployment(self):
        """Testa início de deployment canary"""
        # Criar deployment
        self.canary_manager.create_canary_deployment(self.canary_config)
        
        # Iniciar deployment
        result = self.canary_manager.start_canary_deployment(self.canary_config.deployment_id)
        self.assertTrue(result)
        
        # Verificar status
        status = self.canary_manager.get_deployment_status(self.canary_config.deployment_id)
        self.assertEqual(status['status'], DeploymentStatus.ACTIVE.value)
        self.assertEqual(len(status['steps']), 1)
        self.assertEqual(status['current_step'], 1)
    
    def test_deployment_monitoring(self):
        """Testa monitoramento de deployment"""
        # Criar e iniciar deployment
        self.canary_manager.create_canary_deployment(self.canary_config)
        self.canary_manager.start_canary_deployment(self.canary_config.deployment_id)
        
        # Iniciar monitoramento
        self.canary_manager.start_monitoring()
        
        # Aguardar um ciclo de monitoramento
        time.sleep(2)
        
        # Verificar se deployment está sendo monitorado
        self.assertTrue(self.canary_manager.monitoring_active)
        self.assertIn(self.canary_config.deployment_id, self.canary_manager.active_deployments)
    
    def test_force_rollback(self):
        """Testa rollback forçado"""
        # Criar e iniciar deployment
        self.canary_manager.create_canary_deployment(self.canary_config)
        self.canary_manager.start_canary_deployment(self.canary_config.deployment_id)
        
        # Forçar rollback
        result = self.canary_manager.force_rollback(
            self.canary_config.deployment_id, 
            "Teste de rollback"
        )
        self.assertTrue(result)
        
        # Verificar status após rollback
        status = self.canary_manager.get_deployment_status(self.canary_config.deployment_id)
        self.assertEqual(status['status'], DeploymentStatus.ROLLED_BACK.value)
        self.assertEqual(status['rollback_reason'], "Teste de rollback")
    
    def test_deployment_steps_progression(self):
        """Testa progressão através dos steps"""
        # Configurar deployment com steps rápidos para teste
        quick_config = self.canary_config
        quick_config.step_duration_minutes = 0.1  # 6 segundos
        quick_config.promotion_steps = [5.0, 10.0]  # Apenas 2 steps
        quick_config.target_traffic_percentage = 10.0
        
        # Criar e iniciar
        self.canary_manager.create_canary_deployment(quick_config)
        self.canary_manager.start_canary_deployment(quick_config.deployment_id)
        
        # Verificar step inicial
        status = self.canary_manager.get_deployment_status(quick_config.deployment_id)
        self.assertEqual(status['current_step'], 1)
        self.assertEqual(status['steps'][0]['traffic_percentage'], 5.0)
    
    def test_list_deployments(self):
        """Testa listagem de deployments"""
        # Criar múltiplos deployments
        configs = []
        for i in range(3):
            config = CanaryConfig(
                deployment_id=f"test_deployment_{i}",
                model_name="fraud_model",
                current_version="v1.0",
                canary_version=f"v1.{i}",
                initial_traffic_percentage=5.0,
                target_traffic_percentage=100.0,
                promotion_steps=[5.0, 100.0],
                step_duration_minutes=5,
                success_criteria={"accuracy": 0.95},
                rollback_criteria={"error_rate": 0.05},
                health_check_interval_seconds=30,
                created_by="test_user",
                created_at=datetime.now().isoformat()
            )
            configs.append(config)
            self.canary_manager.create_canary_deployment(config)
        
        # Listar todos
        all_deployments = self.canary_manager.list_deployments()
        self.assertEqual(len(all_deployments), 3)
        
        # Iniciar um deployment
        self.canary_manager.start_canary_deployment(configs[0].deployment_id)
        
        # Listar apenas ativos
        active_deployments = self.canary_manager.list_deployments(DeploymentStatus.ACTIVE)
        self.assertEqual(len(active_deployments), 1)

def run_ab_canary_tests():
    """Executa todos os testes de A/B testing e canary deployment"""
    print("🧪 Iniciando testes de A/B Testing e Canary Deployment...")
    
    # Criar suite de testes
    test_suite = unittest.TestSuite()
    
    # Adicionar testes de A/B Testing
    ab_tests = unittest.TestLoader().loadTestsFromTestCase(TestABTesting)
    test_suite.addTests(ab_tests)
    
    # Adicionar testes de Canary Deployment
    canary_tests = unittest.TestLoader().loadTestsFromTestCase(TestCanaryDeployment)
    test_suite.addTests(canary_tests)
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Resumo dos resultados
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n📊 Resultados dos Testes de A/B Testing e Canary Deployment:")
    print(f"   Total de testes: {total_tests}")
    print(f"   Sucessos: {total_tests - failures - errors}")
    print(f"   Falhas: {failures}")
    print(f"   Erros: {errors}")
    print(f"   Taxa de sucesso: {success_rate:.1f}%")
    
    if failures > 0:
        print(f"\n❌ Falhas detectadas:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"   - {test}: {error_msg}")
    
    if errors > 0:
        print(f"\n💥 Erros detectados:")
        for test, traceback in result.errors:
            error_lines = traceback.split('\n')
            error_msg = error_lines[-2] if len(error_lines) > 1 else 'Erro desconhecido'
            print(f"   - {test}: {error_msg}")
    
    return success_rate >= 80.0  # Considerar sucesso se >= 80% dos testes passarem

if __name__ == '__main__':
    success = run_ab_canary_tests()
    exit(0 if success else 1)

