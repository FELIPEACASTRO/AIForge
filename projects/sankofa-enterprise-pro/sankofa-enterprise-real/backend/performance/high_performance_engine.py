#!/usr/bin/env python3
"""
Motor de Alta Performance para Detecção de Fraudes
Sankofa Enterprise Pro - High Performance Engine
Otimizado para processar 5 milhões de requisições/dia
"""

import logging
import asyncio
import aioredis
import asyncpg
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import pandas as pd
from functools import lru_cache
import pickle
import hashlib
import os
from queue import Queue, Empty
import multiprocessing as mp

logger = logging.getLogger(__name__)

@dataclass
class TransactionRequest:
    """Request de transação otimizado"""
    id: str
    customer_id: str
    amount: float
    merchant_id: str
    timestamp: str
    features: Dict[str, float]
    priority: int = 1  # 1=normal, 2=high, 3=critical

@dataclass
class FraudPrediction:
    """Resultado de predição otimizado"""
    transaction_id: str
    fraud_score: float
    decision: str
    confidence: float
    processing_time_ms: float
    model_version: str
    features_used: List[str]

class HighPerformanceCache:
    """Sistema de cache de alta performance"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_pool = None
        self.local_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0
        }
        
    async def initialize(self):
        """Inicializa conexão Redis"""
        try:
            self.redis_pool = aioredis.ConnectionPool.from_url(
                self.redis_url, 
                max_connections=100,
                retry_on_timeout=True
            )
            logger.info("✅ Cache Redis inicializado")
        except Exception as e:
            logger.warning(f"⚠️ Redis não disponível, usando cache local: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Busca valor no cache"""
        try:
            # Tentar Redis primeiro
            if self.redis_pool:
                redis = aioredis.Redis(connection_pool=self.redis_pool)
                value = await redis.get(key)
                if value:
                    self.cache_stats['hits'] += 1
                    return pickle.loads(value)
            
            # Fallback para cache local
            if key in self.local_cache:
                self.cache_stats['hits'] += 1
                return self.local_cache[key]
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"❌ Erro no cache get: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Define valor no cache"""
        try:
            # Tentar Redis primeiro
            if self.redis_pool:
                redis = aioredis.Redis(connection_pool=self.redis_pool)
                await redis.set(key, pickle.dumps(value), ex=ttl)
            
            # Sempre manter no cache local também
            self.local_cache[key] = value
            self.cache_stats['sets'] += 1
            
            # Limitar tamanho do cache local
            if len(self.local_cache) > 10000:
                # Remove 20% dos itens mais antigos
                keys_to_remove = list(self.local_cache.keys())[:2000]
                for k in keys_to_remove:
                    del self.local_cache[k]
                    
        except Exception as e:
            logger.error(f"❌ Erro no cache set: {e}")

class AsyncDatabasePool:
    """Pool de conexões assíncronas com PostgreSQL"""
    
    def __init__(self, database_url: str, min_size: int = 10, max_size: int = 100):
        self.database_url = database_url
        self.min_size = min_size
        self.max_size = max_size
        self.pool = None
        
    async def initialize(self):
        """Inicializa pool de conexões"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=5,
                server_settings={
                    'jit': 'off',  # Desabilitar JIT para queries rápidas
                    'application_name': 'sankofa_fraud_engine'
                }
            )
            logger.info(f"✅ Pool de banco de dados inicializado ({self.min_size}-{self.max_size} conexões)")
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar pool de banco: {e}")
            raise
    
    async def execute_query(self, query: str, *args) -> List[Dict]:
        """Executa query com pool de conexões"""
        if not self.pool:
            raise RuntimeError("Pool não inicializado")
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def execute_batch(self, query: str, args_list: List[Tuple]) -> None:
        """Executa batch de queries"""
        if not self.pool:
            raise RuntimeError("Pool não inicializado")
        
        async with self.pool.acquire() as conn:
            await conn.executemany(query, args_list)

class ModelPredictor:
    """Preditor de modelo otimizado para alta performance"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.feature_cache = {}
        self.prediction_cache = {}
        
    def load_model(self):
        """Carrega modelo em memória"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✅ Modelo carregado: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            # Usar modelo fallback simples
            self.model = self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Cria modelo fallback simples"""
        class FallbackModel:
            def predict_proba(self, X):
                # Modelo simples baseado em regras
                scores = []
                for row in X:
                    amount = row[0] if len(row) > 0 else 100
                    # Score baseado no valor da transação
                    score = min(amount / 10000, 0.9)
                    scores.append([1-score, score])
                return np.array(scores)
        
        return FallbackModel()
    
    @lru_cache(maxsize=10000)
    def extract_features(self, transaction_data: str) -> np.ndarray:
        """Extrai features com cache"""
        # Parse dos dados da transação
        data = json.loads(transaction_data)
        
        # Features básicas otimizadas
        features = [
            float(data.get('amount', 0)),
            float(data.get('hour', 12)),
            float(data.get('day_of_week', 1)),
            float(data.get('is_weekend', 0)),
            float(data.get('merchant_risk', 0.1)),
            float(data.get('location_risk', 0.1)),
            float(data.get('device_risk', 0.1)),
            float(data.get('channel_risk', 0.1)),
            float(data.get('amount_log', np.log1p(data.get('amount', 0)))),
            float(data.get('velocity_1h', 0)),
            float(data.get('velocity_24h', 0)),
            float(data.get('customer_age_days', 365)),
            float(data.get('avg_transaction_amount', 100)),
            float(data.get('transaction_count_30d', 10))
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, transaction_data: Dict[str, Any]) -> FraudPrediction:
        """Predição otimizada"""
        start_time = time.time()
        
        try:
            # Criar hash para cache
            data_str = json.dumps(transaction_data, sort_keys=True)
            cache_key = hashlib.md5(data_str.encode()).hexdigest()
            
            # Verificar cache de predição
            if cache_key in self.prediction_cache:
                cached_result = self.prediction_cache[cache_key]
                cached_result.processing_time_ms = (time.time() - start_time) * 1000
                return cached_result
            
            # Extrair features
            features = self.extract_features(data_str)
            
            # Predição
            if self.model:
                pred_proba = self.model.predict_proba(features)[0]
                fraud_score = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
                confidence = max(pred_proba)
            else:
                fraud_score = 0.5
                confidence = 0.5
            
            # Decisão baseada no score
            if fraud_score > 0.8:
                decision = 'block'
            elif fraud_score > 0.3:
                decision = 'review'
            else:
                decision = 'approve'
            
            # Criar resultado
            result = FraudPrediction(
                transaction_id=transaction_data.get('id', 'unknown'),
                fraud_score=float(fraud_score),
                decision=decision,
                confidence=float(confidence),
                processing_time_ms=(time.time() - start_time) * 1000,
                model_version='high_performance_v1.0',
                features_used=['amount', 'merchant_risk', 'location_risk']
            )
            
            # Cache do resultado
            self.prediction_cache[cache_key] = result
            
            # Limitar tamanho do cache
            if len(self.prediction_cache) > 5000:
                # Remove 20% dos itens mais antigos
                keys_to_remove = list(self.prediction_cache.keys())[:1000]
                for k in keys_to_remove:
                    del self.prediction_cache[k]
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro na predição: {e}")
            # Retornar predição fallback
            return FraudPrediction(
                transaction_id=transaction_data.get('id', 'unknown'),
                fraud_score=0.5,
                decision='review',
                confidence=0.5,
                processing_time_ms=(time.time() - start_time) * 1000,
                model_version='fallback_v1.0',
                features_used=['fallback']
            )

class HighPerformanceFraudEngine:
    """Motor de detecção de fraudes de alta performance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Componentes principais
        self.cache = HighPerformanceCache(config.get('redis_url', 'redis://localhost:6379'))
        self.db_pool = AsyncDatabasePool(
            config.get('database_url', 'postgresql://user:pass@localhost/db'),
            min_size=config.get('db_min_connections', 20),
            max_size=config.get('db_max_connections', 200)
        )
        
        # Predictors (múltiplas instâncias para paralelismo)
        self.predictors = []
        model_path = config.get('model_path', '/tmp/fraud_model.pkl')
        for i in range(config.get('predictor_instances', mp.cpu_count())):
            predictor = ModelPredictor(model_path)
            predictor.load_model()
            self.predictors.append(predictor)
        
        # Thread pools
        self.prediction_executor = ThreadPoolExecutor(
            max_workers=config.get('prediction_threads', mp.cpu_count() * 2)
        )
        self.io_executor = ThreadPoolExecutor(
            max_workers=config.get('io_threads', 50)
        )
        
        # Filas de processamento
        self.high_priority_queue = asyncio.Queue(maxsize=10000)
        self.normal_priority_queue = asyncio.Queue(maxsize=50000)
        
        # Métricas de performance
        self.metrics = {
            'requests_processed': 0,
            'requests_per_second': 0,
            'avg_processing_time_ms': 0,
            'cache_hit_rate': 0,
            'error_rate': 0,
            'queue_sizes': {'high': 0, 'normal': 0}
        }
        
        # Workers assíncronos
        self.workers = []
        self.running = False
        
        logger.info("🚀 High Performance Fraud Engine inicializado")
    
    async def initialize(self):
        """Inicializa todos os componentes"""
        await self.cache.initialize()
        await self.db_pool.initialize()
        
        # Iniciar workers
        self.running = True
        num_workers = self.config.get('async_workers', 20)
        
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Iniciar worker de métricas
        metrics_worker = asyncio.create_task(self._metrics_worker())
        self.workers.append(metrics_worker)
        
        logger.info(f"✅ {num_workers} workers assíncronos iniciados")
    
    async def shutdown(self):
        """Para todos os componentes"""
        self.running = False
        
        # Aguardar workers terminarem
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Fechar executors
        self.prediction_executor.shutdown(wait=True)
        self.io_executor.shutdown(wait=True)
        
        logger.info("🛑 High Performance Fraud Engine parado")
    
    async def process_transaction(self, transaction_data: Dict[str, Any], priority: int = 1) -> FraudPrediction:
        """Processa transação com prioridade"""
        request = TransactionRequest(
            id=transaction_data.get('id', f"txn_{int(time.time())}"),
            customer_id=transaction_data.get('customer_id', 'unknown'),
            amount=float(transaction_data.get('amount', 0)),
            merchant_id=transaction_data.get('merchant_id', 'unknown'),
            timestamp=transaction_data.get('timestamp', datetime.now().isoformat()),
            features=transaction_data,
            priority=priority
        )
        
        # Adicionar à fila apropriada
        if priority >= 2:
            await self.high_priority_queue.put(request)
        else:
            await self.normal_priority_queue.put(request)
        
        # Para este exemplo, retornar predição direta
        # Em produção, isso seria assíncrono com callbacks
        return await self._predict_transaction(request)
    
    async def _worker(self, worker_id: str):
        """Worker assíncrono para processar transações"""
        while self.running:
            try:
                # Priorizar fila de alta prioridade
                request = None
                try:
                    request = await asyncio.wait_for(
                        self.high_priority_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    try:
                        request = await asyncio.wait_for(
                            self.normal_priority_queue.get(), timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        continue
                
                if request:
                    # Processar transação
                    result = await self._predict_transaction(request)
                    
                    # Salvar resultado assincronamente
                    asyncio.create_task(self._save_result(request, result))
                    
                    # Atualizar métricas
                    self.metrics['requests_processed'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Erro no worker {worker_id}: {e}")
                await asyncio.sleep(1)
    
    async def _predict_transaction(self, request: TransactionRequest) -> FraudPrediction:
        """Executa predição para uma transação"""
        try:
            # Verificar cache primeiro
            cache_key = f"prediction:{request.id}"
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Executar predição em thread separada
            loop = asyncio.get_event_loop()
            predictor = self.predictors[hash(request.id) % len(self.predictors)]
            
            result = await loop.run_in_executor(
                self.prediction_executor,
                predictor.predict,
                request.features
            )
            
            # Cache do resultado
            await self.cache.set(cache_key, result, ttl=300)  # 5 minutos
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro na predição da transação {request.id}: {e}")
            # Retornar resultado fallback
            return FraudPrediction(
                transaction_id=request.id,
                fraud_score=0.5,
                decision='review',
                confidence=0.5,
                processing_time_ms=0,
                model_version='error_fallback',
                features_used=['error']
            )
    
    async def _save_result(self, request: TransactionRequest, result: FraudPrediction):
        """Salva resultado no banco de dados"""
        try:
            query = """
                INSERT INTO fraud_predictions 
                (transaction_id, customer_id, amount, fraud_score, decision, 
                 confidence, processing_time_ms, model_version, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """
            
            await self.db_pool.execute_query(
                query,
                request.id,
                request.customer_id,
                request.amount,
                result.fraud_score,
                result.decision,
                result.confidence,
                result.processing_time_ms,
                result.model_version,
                datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar resultado: {e}")
    
    async def _metrics_worker(self):
        """Worker para calcular métricas de performance"""
        last_requests = 0
        last_time = time.time()
        
        while self.running:
            try:
                await asyncio.sleep(10)  # Atualizar a cada 10 segundos
                
                current_time = time.time()
                current_requests = self.metrics['requests_processed']
                
                # Calcular RPS
                time_diff = current_time - last_time
                requests_diff = current_requests - last_requests
                
                if time_diff > 0:
                    self.metrics['requests_per_second'] = requests_diff / time_diff
                
                # Atualizar cache hit rate
                cache_stats = self.cache.cache_stats
                total_cache_ops = cache_stats['hits'] + cache_stats['misses']
                if total_cache_ops > 0:
                    self.metrics['cache_hit_rate'] = cache_stats['hits'] / total_cache_ops
                
                # Atualizar tamanhos das filas
                self.metrics['queue_sizes'] = {
                    'high': self.high_priority_queue.qsize(),
                    'normal': self.normal_priority_queue.qsize()
                }
                
                last_requests = current_requests
                last_time = current_time
                
                # Log das métricas
                if current_requests % 1000 == 0 and current_requests > 0:
                    logger.info(
                        f"📊 Métricas: {self.metrics['requests_per_second']:.1f} RPS, "
                        f"Cache Hit Rate: {self.metrics['cache_hit_rate']:.2%}, "
                        f"Filas: H={self.metrics['queue_sizes']['high']}, "
                        f"N={self.metrics['queue_sizes']['normal']}"
                    )
                
            except Exception as e:
                logger.error(f"❌ Erro no worker de métricas: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas atuais"""
        return self.metrics.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Verifica saúde do sistema"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        try:
            # Verificar cache
            test_key = f"health_check_{int(time.time())}"
            await self.cache.set(test_key, 'ok', ttl=10)
            cached_value = await self.cache.get(test_key)
            health['components']['cache'] = 'healthy' if cached_value == 'ok' else 'degraded'
            
            # Verificar banco de dados
            result = await self.db_pool.execute_query("SELECT 1 as test")
            health['components']['database'] = 'healthy' if result else 'unhealthy'
            
            # Verificar workers
            active_workers = sum(1 for w in self.workers if not w.done())
            health['components']['workers'] = f"{active_workers}/{len(self.workers)} active"
            
            # Verificar filas
            total_queue_size = (self.high_priority_queue.qsize() + 
                              self.normal_priority_queue.qsize())
            if total_queue_size > 40000:  # 80% da capacidade
                health['components']['queues'] = 'overloaded'
                health['status'] = 'degraded'
            else:
                health['components']['queues'] = 'healthy'
            
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
        
        return health

# Função para criar instância configurada
def create_high_performance_engine(config: Optional[Dict[str, Any]] = None) -> HighPerformanceFraudEngine:
    """Cria instância do motor de alta performance"""
    default_config = {
        'redis_url': 'redis://localhost:6379',
        'database_url': 'postgresql://sankofa:sankofa123@localhost/sankofa_fraud',
        'model_path': '/home/ubuntu/sankofa-enterprise-real/models/fraud_model_v4.pkl',
        'db_min_connections': 20,
        'db_max_connections': 200,
        'predictor_instances': mp.cpu_count(),
        'prediction_threads': mp.cpu_count() * 2,
        'io_threads': 50,
        'async_workers': 20
    }
    
    if config:
        default_config.update(config)
    
    return HighPerformanceFraudEngine(default_config)

# Exemplo de uso
async def main():
    """Exemplo de uso do motor de alta performance"""
    engine = create_high_performance_engine()
    
    try:
        await engine.initialize()
        
        # Simular transações
        for i in range(100):
            transaction = {
                'id': f'TXN_{i:06d}',
                'customer_id': f'CUST_{i % 1000}',
                'amount': np.random.uniform(10, 10000),
                'merchant_id': f'MERCH_{i % 100}',
                'timestamp': datetime.now().isoformat()
            }
            
            result = await engine.process_transaction(transaction)
            print(f"Transação {result.transaction_id}: {result.decision} (score: {result.fraud_score:.3f})")
            
            if i % 10 == 0:
                metrics = engine.get_metrics()
                print(f"RPS: {metrics['requests_per_second']:.1f}")
        
        # Health check
        health = await engine.health_check()
        print(f"Health: {health}")
        
    finally:
        await engine.shutdown()

if __name__ == '__main__':
    asyncio.run(main())

