#!/usr/bin/env python3
"""
Teste QA Otimizado - Foco em Qualidade
Sistema de teste focado na melhoria de precisão, recall e F1-score
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures
import threading
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Adicionar o diretório backend ao path
sys.path.append('/home/ubuntu/sankofa-enterprise-real/backend')

from data.real_time_transaction_generator import RealTimeTransactionGenerator
from ml_engine.optimized_fraud_analyzer import OptimizedFraudAnalyzer
from cache.redis_cache_system import RedisCacheSystem

class OptimizedQATest:
    """Teste QA otimizado focado em qualidade de detecção"""
    
    def __init__(self, total_transactions=100_000):
        self.transaction_generator = RealTimeTransactionGenerator()
        self.fraud_analyzer = OptimizedFraudAnalyzer()
        self.cache_system = RedisCacheSystem()
        
        # Configurações do teste
        self.total_transactions = total_transactions
        self.batch_size = 5_000
        self.num_batches = self.total_transactions // self.batch_size
        
        # Métricas
        self.performance_metrics = {
            'processing_times': [],
            'throughput_per_second': [],
        }
        
        self.quality_metrics = {
            'true_labels': [],
            'predicted_labels': [],
            'fraud_scores': [],
            'processing_times_per_transaction': []
        }
        
        self.stats = {
            'total_processed': 0,
            'total_frauds_detected': 0,
            'total_legitimate': 0,
            'start_time': None,
            'end_time': None,
            'errors': []
        }
        
        print("🔧 Teste QA Otimizado Inicializado")
        print(f"📊 Configuração: {self.total_transactions:,} transações em {self.num_batches} lotes")
    
    def generate_balanced_dataset(self):
        """Gera dataset balanceado para melhor avaliação"""
        print("\n🏗️ Gerando dataset balanceado...")
        
        transactions = []
        fraud_labels = []
        
        # Configurar proporção balanceada (20% fraudes para melhor teste)
        fraud_rate = 0.20
        num_frauds = int(self.total_transactions * fraud_rate)
        num_legitimate = self.total_transactions - num_frauds
        
        print(f"📈 Gerando {num_legitimate:,} transações legítimas")
        print(f"🚨 Gerando {num_frauds:,} transações fraudulentas")
        
        # Gerar transações com labels mais precisos
        for i in range(self.total_transactions):
            tx = self.transaction_generator.generate_transaction()
            
            # Converter para dict se necessário
            if hasattr(tx, '__dict__'):
                transaction = tx.__dict__
            else:
                transaction = tx
            
            # Determinar label de fraude baseado em critérios mais rigorosos
            is_fraud = self._determine_fraud_label(transaction, i < num_frauds)
            
            transactions.append(transaction)
            fraud_labels.append(1 if is_fraud else 0)
            
            if (i + 1) % 10000 == 0:
                frauds_so_far = sum(fraud_labels)
                print(f"✅ Geradas {i + 1:,} transações ({frauds_so_far:,} fraudes)")
        
        # Embaralhar o dataset
        combined = list(zip(transactions, fraud_labels))
        np.random.shuffle(combined)
        transactions, fraud_labels = zip(*combined)
        
        final_fraud_count = sum(fraud_labels)
        print(f"✅ Dataset balanceado gerado: {len(transactions):,} transações ({final_fraud_count:,} fraudes - {final_fraud_count/len(transactions)*100:.1f}%)")
        
        return list(transactions), list(fraud_labels)
    
    def _determine_fraud_label(self, transaction, force_fraud=False):
        """Determina se uma transação deve ser rotulada como fraude"""
        if force_fraud:
            return True
        
        # Critérios mais rigorosos para determinar fraude real
        valor = transaction.get('valor', 0)
        canal = transaction.get('canal', '').upper()
        fraud_score = transaction.get('fraud_score', 0.0)
        
        # Critérios de fraude baseados em padrões reais
        fraud_indicators = 0
        
        # Valor muito alto
        if valor > 50000:
            fraud_indicators += 2
        elif valor > 20000:
            fraud_indicators += 1
        
        # Canal de risco
        if canal in ['INTERNET', 'MOBILE'] and valor > 10000:
            fraud_indicators += 1
        
        # Score original alto
        if fraud_score > 0.8:
            fraud_indicators += 2
        elif fraud_score > 0.6:
            fraud_indicators += 1
        
        # Horário suspeito
        try:
            data_hora = transaction.get('data_hora', '')
            if ':' in data_hora:
                hora = int(data_hora.split(' ')[-1].split(':')[0])
                if 2 <= hora <= 5:  # Madrugada
                    fraud_indicators += 1
        except:
            pass
        
        # Valores redondos suspeitos
        if valor >= 1000 and valor % 1000 == 0:
            fraud_indicators += 1
        
        # Determinar fraude baseado no número de indicadores
        return fraud_indicators >= 2
    
    def process_batch_optimized(self, batch_transactions, batch_labels, batch_id):
        """Processa um lote com análise otimizada"""
        batch_start_time = time.time()
        batch_predictions = []
        batch_scores = []
        batch_processing_times = []
        
        for transaction in batch_transactions:
            tx_start_time = time.time()
            
            try:
                # Analisar com algoritmo otimizado
                result = self.fraud_analyzer.analyze_transaction(transaction)
                
                fraud_score = result.get('fraud_score', 0.0)
                is_fraud = fraud_score > 0.45  # Threshold otimizado
                
                batch_predictions.append(1 if is_fraud else 0)
                batch_scores.append(fraud_score)
                
                processing_time = (time.time() - tx_start_time) * 1000
                batch_processing_times.append(processing_time)
                
            except Exception as e:
                self.stats['errors'].append(f"Batch {batch_id}, Transaction error: {str(e)}")
                batch_predictions.append(0)
                batch_scores.append(0.0)
                batch_processing_times.append(0.0)
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        batch_throughput = len(batch_transactions) / batch_duration
        
        # Calcular métricas do lote
        batch_accuracy = accuracy_score(batch_labels, batch_predictions)
        batch_precision = precision_score(batch_labels, batch_predictions, zero_division=0)
        batch_recall = recall_score(batch_labels, batch_predictions, zero_division=0)
        batch_f1 = f1_score(batch_labels, batch_predictions, zero_division=0)
        
        return {
            'batch_id': batch_id,
            'predictions': batch_predictions,
            'scores': batch_scores,
            'processing_times': batch_processing_times,
            'duration': batch_duration,
            'throughput': batch_throughput,
            'accuracy': batch_accuracy,
            'precision': batch_precision,
            'recall': batch_recall,
            'f1_score': batch_f1
        }
    
    def run_optimized_test(self, transactions, labels):
        """Executa teste otimizado"""
        print("\n🚀 Iniciando teste otimizado...")
        
        self.stats['start_time'] = datetime.now()
        
        # Dividir em lotes
        batches = []
        for i in range(0, len(transactions), self.batch_size):
            batch_transactions = transactions[i:i + self.batch_size]
            batch_labels = labels[i:i + self.batch_size]
            batches.append((batch_transactions, batch_labels, i // self.batch_size))
        
        print(f"📦 Processando {len(batches)} lotes de {self.batch_size:,} transações cada")
        
        # Processamento com menos paralelismo para melhor qualidade
        batch_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_batch = {
                executor.submit(self.process_batch_optimized, batch_tx, batch_labels, batch_id): batch_id
                for batch_tx, batch_labels, batch_id in batches
            }
            
            completed_batches = 0
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    result = future.result()
                    batch_results.append(result)
                    completed_batches += 1
                    
                    progress = (completed_batches / len(batches)) * 100
                    print(f"⏳ Progresso: {progress:.1f}% - Lote {batch_id + 1}/{len(batches)}")
                    print(f"   📊 TPS: {result['throughput']:.1f}, Acc: {result['accuracy']:.3f}, Prec: {result['precision']:.3f}, Rec: {result['recall']:.3f}, F1: {result['f1_score']:.3f}")
                    
                except Exception as e:
                    self.stats['errors'].append(f"Batch {batch_id} failed: {str(e)}")
                    print(f"❌ Erro no lote {batch_id}: {str(e)}")
        
        self.stats['end_time'] = datetime.now()
        
        # Consolidar resultados
        self.consolidate_optimized_results(batch_results, labels)
        
        return batch_results
    
    def consolidate_optimized_results(self, batch_results, true_labels):
        """Consolida resultados otimizados"""
        print("\n📊 Consolidando resultados otimizados...")
        
        all_predictions = []
        all_scores = []
        all_processing_times = []
        
        for result in batch_results:
            all_predictions.extend(result['predictions'])
            all_scores.extend(result['scores'])
            all_processing_times.extend(result['processing_times'])
        
        self.quality_metrics['true_labels'] = true_labels
        self.quality_metrics['predicted_labels'] = all_predictions
        self.quality_metrics['fraud_scores'] = all_scores
        self.quality_metrics['processing_times_per_transaction'] = all_processing_times
        
        self.stats['total_processed'] = len(all_predictions)
        self.stats['total_frauds_detected'] = sum(all_predictions)
        self.stats['total_legitimate'] = len(all_predictions) - sum(all_predictions)
        
        total_duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        overall_throughput = self.stats['total_processed'] / total_duration
        
        self.performance_metrics['overall_throughput'] = overall_throughput
        self.performance_metrics['total_duration'] = total_duration
        self.performance_metrics['avg_processing_time'] = np.mean(all_processing_times)
        self.performance_metrics['p95_processing_time'] = np.percentile(all_processing_times, 95)
        self.performance_metrics['p99_processing_time'] = np.percentile(all_processing_times, 99)
        
        print(f"✅ Consolidação concluída: {self.stats['total_processed']:,} transações processadas")
    
    def calculate_optimized_quality_metrics(self):
        """Calcula métricas de qualidade otimizadas"""
        print("\n🎯 Calculando métricas de qualidade otimizadas...")
        
        true_labels = self.quality_metrics['true_labels']
        predicted_labels = self.quality_metrics['predicted_labels']
        fraud_scores = self.quality_metrics['fraud_scores']
        
        # Métricas básicas
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(true_labels, fraud_scores)
        except:
            auc_roc = 0.0
        
        # Matriz de confusão
        cm = confusion_matrix(true_labels, predicted_labels)
        tn, fp, fn, tp = cm.ravel()
        
        # Métricas derivadas
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        quality_report = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'specificity': specificity,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
        }
        
        return quality_report
    
    def generate_optimized_report(self, quality_report):
        """Gera relatório otimizado"""
        print("\n📋 Gerando relatório otimizado...")
        
        total_duration = self.performance_metrics['total_duration']
        overall_throughput = self.performance_metrics['overall_throughput']
        avg_processing_time = self.performance_metrics['avg_processing_time']
        p95_processing_time = self.performance_metrics['p95_processing_time']
        p99_processing_time = self.performance_metrics['p99_processing_time']
        
        report = {
            'test_configuration': {
                'total_transactions': self.total_transactions,
                'batch_size': self.batch_size,
                'num_batches': self.num_batches,
                'test_start': self.stats['start_time'].isoformat(),
                'test_end': self.stats['end_time'].isoformat(),
                'total_duration_seconds': total_duration,
                'test_type': 'optimized_quality_focused'
            },
            'performance_metrics': {
                'overall_throughput_tps': round(overall_throughput, 2),
                'avg_processing_time_ms': round(avg_processing_time, 2),
                'p95_processing_time_ms': round(p95_processing_time, 2),
                'p99_processing_time_ms': round(p99_processing_time, 2),
                'total_processed': self.stats['total_processed'],
                'processing_errors': len(self.stats['errors'])
            },
            'quality_metrics': quality_report,
            'fraud_detection_stats': {
                'total_frauds_in_dataset': sum(self.quality_metrics['true_labels']),
                'total_frauds_detected': self.stats['total_frauds_detected'],
                'total_legitimate': self.stats['total_legitimate'],
                'fraud_detection_rate': quality_report['recall'],
                'false_alarm_rate': quality_report['false_positive_rate']
            },
            'system_requirements_check': {
                'target_throughput_100_tps': overall_throughput >= 100,
                'target_latency_50ms': p95_processing_time <= 50,
                'target_accuracy_90pct': quality_report['accuracy'] >= 0.90,
                'target_precision_85pct': quality_report['precision'] >= 0.85,
                'target_recall_80pct': quality_report['recall'] >= 0.80,
                'target_f1_75pct': quality_report['f1_score'] >= 0.75
            },
            'optimization_results': {
                'algorithm_version': 'OptimizedFraudAnalyzer v2.0',
                'threshold_high_risk': 0.45,
                'threshold_medium_risk': 0.25,
                'fraud_rate_in_dataset': sum(self.quality_metrics['true_labels']) / len(self.quality_metrics['true_labels'])
            }
        }
        
        return report
    
    def print_optimized_summary(self, report):
        """Exibe resumo otimizado dos resultados"""
        print("\n📊 RESUMO EXECUTIVO - TESTE QA OTIMIZADO")
        print("-" * 60)
        
        perf = report['performance_metrics']
        qual = report['quality_metrics']
        req_check = report['system_requirements_check']
        
        print(f"🔢 Transações Processadas: {perf['total_processed']:,}")
        print(f"⚡ Throughput: {perf['overall_throughput_tps']:.1f} TPS")
        print(f"⏱️  Latência P95: {perf['p95_processing_time_ms']:.1f}ms")
        print(f"🎯 Accuracy: {qual['accuracy']:.3f} ({qual['accuracy']*100:.1f}%)")
        print(f"🎯 Precision: {qual['precision']:.3f} ({qual['precision']*100:.1f}%)")
        print(f"🎯 Recall: {qual['recall']:.3f} ({qual['recall']*100:.1f}%)")
        print(f"🎯 F1-Score: {qual['f1_score']:.3f} ({qual['f1_score']*100:.1f}%)")
        print(f"🎯 AUC-ROC: {qual['auc_roc']:.3f}")
        
        print("\n✅ REQUISITOS DE SISTEMA:")
        print(f"   Throughput ≥100 TPS: {'✅ PASS' if req_check['target_throughput_100_tps'] else '❌ FAIL'}")
        print(f"   Latência ≤50ms: {'✅ PASS' if req_check['target_latency_50ms'] else '❌ FAIL'}")
        print(f"   Accuracy ≥90%: {'✅ PASS' if req_check['target_accuracy_90pct'] else '❌ FAIL'}")
        print(f"   Precision ≥85%: {'✅ PASS' if req_check['target_precision_85pct'] else '❌ FAIL'}")
        print(f"   Recall ≥80%: {'✅ PASS' if req_check['target_recall_80pct'] else '❌ FAIL'}")
        print(f"   F1-Score ≥75%: {'✅ PASS' if req_check['target_f1_75pct'] else '❌ FAIL'}")
        
        # Calcular score geral
        passed_requirements = sum(req_check.values())
        total_requirements = len(req_check)
        overall_score = (passed_requirements / total_requirements) * 100
        
        print(f"\n🏆 SCORE GERAL: {overall_score:.1f}% ({passed_requirements}/{total_requirements} requisitos)")
        
        if overall_score >= 80:
            print("🎉 SISTEMA OTIMIZADO APROVADO PARA PRODUÇÃO!")
        elif overall_score >= 60:
            print("⚠️ SISTEMA MELHORADO - REQUER AJUSTES FINAIS")
        else:
            print("❌ SISTEMA AINDA REQUER OTIMIZAÇÕES SIGNIFICATIVAS")
    
    def run_complete_optimized_test(self):
        """Executa o teste completo otimizado"""
        print("🚀 INICIANDO TESTE QA OTIMIZADO - FOCO EM QUALIDADE")
        print("=" * 70)
        
        try:
            # 1. Gerar dataset balanceado
            transactions, labels = self.generate_balanced_dataset()
            
            # 2. Executar teste otimizado
            batch_results = self.run_optimized_test(transactions, labels)
            
            # 3. Calcular métricas de qualidade
            quality_report = self.calculate_optimized_quality_metrics()
            
            # 4. Gerar relatório otimizado
            performance_report = self.generate_optimized_report(quality_report)
            
            # 5. Salvar relatório final
            report_path = '/home/ubuntu/sankofa-enterprise-real/tests/qa_report_optimized.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(performance_report, f, indent=2, ensure_ascii=False, default=str)
            
            print("\n" + "=" * 70)
            print("✅ TESTE QA OTIMIZADO FINALIZADO COM SUCESSO!")
            print("=" * 70)
            
            # Exibir resumo final
            self.print_optimized_summary(performance_report)
            
            return performance_report, report_path
            
        except Exception as e:
            print(f"❌ Erro durante o teste QA otimizado: {str(e)}")
            raise

def main():
    """Função principal para executar o teste QA otimizado"""
    print("🔧 Inicializando Teste QA Otimizado - Sankofa Enterprise Pro")
    
    # Criar instância do teste otimizado
    qa_test = OptimizedQATest(total_transactions=100_000)
    
    # Executar teste completo
    try:
        report, report_path = qa_test.run_complete_optimized_test()
        
        print(f"\n📄 Relatório otimizado salvo em: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Falha no teste QA otimizado: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
