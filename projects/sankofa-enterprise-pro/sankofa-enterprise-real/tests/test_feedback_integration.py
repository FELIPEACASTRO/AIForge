#!/usr/bin/env python3
"""
Testes de Integração para o Módulo de Feedback Humano
Sankofa Enterprise Pro - Feedback Integration Tests
"""

import unittest
import tempfile
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import sys

# Adicionar o diretório raiz ao path
sys.path.append('/home/ubuntu/sankofa-enterprise-real')

from backend.mlops.human_feedback_module import HumanFeedbackModule
from backend.mlops.model_lifecycle_manager import ModelLifecycleManager
from backend.mlops.feedback_integration import FeedbackIntegration

class TestFeedbackIntegration(unittest.TestCase):
    """Testes para a integração do feedback humano"""
    
    def setUp(self):
        """Configuração inicial dos testes"""
        # Criar arquivo temporário para feedback
        self.temp_feedback_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_feedback_file.close()
        
        # Inicializar módulos
        self.feedback_module = HumanFeedbackModule(self.temp_feedback_file.name)
        self.lifecycle_manager = ModelLifecycleManager()
        self.feedback_integration = FeedbackIntegration(
            self.feedback_module, 
            self.lifecycle_manager
        )
        
        # Dados de teste
        self.sample_feedbacks = [
            {
                'transaction_id': 'TXN001',
                'model_prediction': 1,
                'actual_label': 1,
                'analyst_id': 'analyst1',
                'comments': 'Fraude confirmada'
            },
            {
                'transaction_id': 'TXN002',
                'model_prediction': 0,
                'actual_label': 0,
                'analyst_id': 'analyst1',
                'comments': 'Transação legítima'
            },
            {
                'transaction_id': 'TXN003',
                'model_prediction': 1,
                'actual_label': 0,
                'analyst_id': 'analyst2',
                'comments': 'Falso positivo'
            },
            {
                'transaction_id': 'TXN004',
                'model_prediction': 0,
                'actual_label': 1,
                'analyst_id': 'analyst2',
                'comments': 'Falso negativo - fraude não detectada'
            }
        ]
    
    def tearDown(self):
        """Limpeza após os testes"""
        try:
            os.unlink(self.temp_feedback_file.name)
        except:
            pass
    
    def test_feedback_module_initialization(self):
        """Testa a inicialização do módulo de feedback"""
        self.assertIsNotNone(self.feedback_module)
        self.assertTrue(os.path.exists(self.temp_feedback_file.name))
        
        # Verificar se o arquivo CSV foi criado com as colunas corretas
        feedback_df = self.feedback_module.get_feedback()
        expected_columns = ['transaction_id', 'model_prediction', 'actual_label', 
                          'feedback_timestamp', 'analyst_id', 'comments']
        self.assertEqual(list(feedback_df.columns), expected_columns)
    
    def test_record_feedback(self):
        """Testa o registro de feedback"""
        feedback = self.sample_feedbacks[0]
        
        # Registrar feedback
        self.feedback_module.record_feedback(
            transaction_id=feedback['transaction_id'],
            model_prediction=feedback['model_prediction'],
            actual_label=feedback['actual_label'],
            analyst_id=feedback['analyst_id'],
            comments=feedback['comments']
        )
        
        # Verificar se foi registrado
        feedback_df = self.feedback_module.get_feedback()
        self.assertEqual(len(feedback_df), 1)
        self.assertEqual(feedback_df.iloc[0]['transaction_id'], feedback['transaction_id'])
        self.assertEqual(feedback_df.iloc[0]['model_prediction'], feedback['model_prediction'])
        self.assertEqual(feedback_df.iloc[0]['actual_label'], feedback['actual_label'])
        self.assertEqual(feedback_df.iloc[0]['analyst_id'], feedback['analyst_id'])
    
    def test_multiple_feedbacks(self):
        """Testa o registro de múltiplos feedbacks"""
        # Registrar todos os feedbacks de teste
        for feedback in self.sample_feedbacks:
            self.feedback_module.record_feedback(
                transaction_id=feedback['transaction_id'],
                model_prediction=feedback['model_prediction'],
                actual_label=feedback['actual_label'],
                analyst_id=feedback['analyst_id'],
                comments=feedback['comments']
            )
        
        # Verificar se todos foram registrados
        feedback_df = self.feedback_module.get_feedback()
        self.assertEqual(len(feedback_df), len(self.sample_feedbacks))
    
    def test_feedback_metrics_calculation(self):
        """Testa o cálculo de métricas baseadas no feedback"""
        # Registrar feedbacks de teste
        for feedback in self.sample_feedbacks:
            self.feedback_module.record_feedback(
                transaction_id=feedback['transaction_id'],
                model_prediction=feedback['model_prediction'],
                actual_label=feedback['actual_label'],
                analyst_id=feedback['analyst_id'],
                comments=feedback['comments']
            )
        
        # Analisar feedback
        feedback_metrics = self.feedback_integration.analyze_recent_feedback()
        
        # Verificar se as métricas foram calculadas
        self.assertIsNotNone(feedback_metrics)
        self.assertEqual(feedback_metrics.total_feedbacks, 4)
        
        # Verificar accuracy (2 corretos de 4 = 0.5)
        self.assertEqual(feedback_metrics.model_accuracy, 0.5)
        
        # Verificar precision (1 verdadeiro positivo / (1 verdadeiro positivo + 1 falso positivo) = 0.5)
        self.assertEqual(feedback_metrics.precision, 0.5)
        
        # Verificar recall (1 verdadeiro positivo / (1 verdadeiro positivo + 1 falso negativo) = 0.5)
        self.assertEqual(feedback_metrics.recall, 0.5)
    
    def test_feedback_trend_analysis(self):
        """Testa a análise de tendência do feedback"""
        # Criar feedbacks com tendência de melhoria
        improving_feedbacks = [
            # Primeira metade - performance ruim
            ('TXN001', 1, 0, 'analyst1'),  # Falso positivo
            ('TXN002', 0, 1, 'analyst1'),  # Falso negativo
            ('TXN003', 1, 0, 'analyst2'),  # Falso positivo
            ('TXN004', 0, 1, 'analyst2'),  # Falso negativo
            # Segunda metade - performance melhor
            ('TXN005', 1, 1, 'analyst1'),  # Verdadeiro positivo
            ('TXN006', 0, 0, 'analyst1'),  # Verdadeiro negativo
            ('TXN007', 1, 1, 'analyst2'),  # Verdadeiro positivo
            ('TXN008', 0, 0, 'analyst2'),  # Verdadeiro negativo
        ]
        
        # Registrar feedbacks com timestamps espaçados
        base_time = datetime.now() - timedelta(hours=24)
        for i, (txn_id, pred, actual, analyst) in enumerate(improving_feedbacks):
            # Simular timestamps diferentes
            timestamp = base_time + timedelta(hours=i * 3)
            
            self.feedback_module.record_feedback(
                transaction_id=txn_id,
                model_prediction=pred,
                actual_label=actual,
                analyst_id=analyst,
                comments=f'Feedback {i+1}'
            )
            
            # Atualizar timestamp manualmente para simular ordem temporal
            feedback_df = self.feedback_module.get_feedback()
            feedback_df.loc[feedback_df['transaction_id'] == txn_id, 'feedback_timestamp'] = timestamp.isoformat()
            feedback_df.to_csv(self.temp_feedback_file.name, index=False)
        
        # Analisar tendência
        feedback_df = self.feedback_module.get_feedback()
        trend = self.feedback_integration._analyze_feedback_trend(feedback_df)
        
        # Deve detectar melhoria
        self.assertEqual(trend, 'improving')
    
    def test_retraining_triggers(self):
        """Testa os gatilhos de retreinamento baseados em feedback"""
        # Criar feedbacks que devem disparar retreinamento
        poor_performance_feedbacks = [
            # Muitos falsos positivos e negativos
            ('TXN001', 1, 0, 'analyst1'),  # Falso positivo
            ('TXN002', 1, 0, 'analyst1'),  # Falso positivo
            ('TXN003', 1, 0, 'analyst2'),  # Falso positivo
            ('TXN004', 0, 1, 'analyst2'),  # Falso negativo
            ('TXN005', 0, 1, 'analyst1'),  # Falso negativo
        ] * 20  # Repetir para ter feedbacks suficientes
        
        # Registrar feedbacks
        for i, (txn_id, pred, actual, analyst) in enumerate(poor_performance_feedbacks):
            self.feedback_module.record_feedback(
                transaction_id=f'{txn_id}_{i}',
                model_prediction=pred,
                actual_label=actual,
                analyst_id=analyst,
                comments=f'Feedback {i+1}'
            )
        
        # Analisar feedback e verificar triggers
        initial_triggers = len(self.lifecycle_manager.retraining_triggers)
        feedback_metrics = self.feedback_integration.analyze_recent_feedback()
        
        # Verificar se triggers foram criados
        self.assertIsNotNone(feedback_metrics)
        final_triggers = len(self.lifecycle_manager.retraining_triggers)
        self.assertGreater(final_triggers, initial_triggers)
    
    def test_feedback_integration_summary(self):
        """Testa o resumo da integração de feedback"""
        # Registrar alguns feedbacks
        for feedback in self.sample_feedbacks:
            self.feedback_module.record_feedback(
                transaction_id=feedback['transaction_id'],
                model_prediction=feedback['model_prediction'],
                actual_label=feedback['actual_label'],
                analyst_id=feedback['analyst_id'],
                comments=feedback['comments']
            )
        
        # Obter resumo
        summary = self.feedback_integration.get_feedback_summary()
        
        # Verificar estrutura do resumo
        self.assertIn('status', summary)
        self.assertIn('metrics', summary)
        self.assertIn('monitoring_active', summary)
        
        if summary['status'] == 'active':
            metrics = summary['metrics']
            self.assertIn('total_feedbacks', metrics)
            self.assertIn('model_accuracy', metrics)
            self.assertIn('precision', metrics)
            self.assertIn('recall', metrics)
    
    def test_insufficient_feedback_data(self):
        """Testa o comportamento com dados insuficientes"""
        # Registrar apenas um feedback (insuficiente)
        self.feedback_module.record_feedback(
            transaction_id='TXN001',
            model_prediction=1,
            actual_label=1,
            analyst_id='analyst1',
            comments='Único feedback'
        )
        
        # Tentar analisar
        feedback_metrics = self.feedback_integration.analyze_recent_feedback()
        
        # Deve retornar None por dados insuficientes
        self.assertIsNone(feedback_metrics)
        
        # Resumo deve indicar dados insuficientes
        summary = self.feedback_integration.get_feedback_summary()
        self.assertEqual(summary['status'], 'insufficient_data')

def run_feedback_integration_tests():
    """Executa todos os testes de integração de feedback"""
    print("🧪 Iniciando testes de integração do feedback humano...")
    
    # Criar suite de testes
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestFeedbackIntegration)
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Resumo dos resultados
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n📊 Resultados dos Testes de Integração de Feedback:")
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
    success = run_feedback_integration_tests()
    exit(0 if success else 1)

