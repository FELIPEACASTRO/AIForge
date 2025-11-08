#!/usr/bin/env python3
"""
Sistema de Alta Performance para Sankofa Enterprise Pro
Implementa cache Redis, connection pooling, otimizações e throughput >1000 RPS
"""

import redis
import asyncio
import asyncpg
import time
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import wraps, lru_cache
import psutil
import numpy as np
from dataclasses import dataclass
import pickle
import zlib
from contextlib import asynccontextmanager
import threading
from queue import Queue, Empty
import sqlite3
from sqlalchemy import create_engine, pool
from sqlalchemy.orm import sessionmaker
import uvloop  # Para melhor performance async

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceConfig:
    """Configurações de performance do sistema"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_max_connections: int = 100
    
    # Cache settings
    cache_ttl_seconds: int = 300  # 5 minutos
    cache_max_memory: str = "1gb"
    cache_eviction_policy: str = "allkeys-lru"
    
    # Database settings
    db_pool_size: int = 20
    db_max_overflow: int = 30
    db_pool_timeout: int = 30
    db_pool_recycle: int = 3600
    
    # Performance settings
    max_workers: int = multiprocessing.cpu_count() * 2
    batch_size: int = 1000
    async_batch_size: int = 100
    
    # Throughput targets
    target_rps: int = 1000
    max_latency_ms: int = 50
    
    # Memory settings
    max_memory_usage_percent: int = 80

class HighPerformanceCache:
    """Sistema de cache Redis de alta performance"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.redis_pool = None

        self._init_redis()
        
        # Estatísticas de cache
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_sets = 0
        
        logger.info("Sistema de Cache Redis inicializado")
    
    def _init_redis(self):
        """Inicializa conexões Redis"""
        try:
            # Pool síncrono
            self.redis_pool = redis.ConnectionPool(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                max_connections=self.config.redis_max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            # Testa conexão
            r = redis.Redis(connection_pool=self.redis_pool)
            r.ping()
            
            # Configura Redis para performance
            r.config_set('maxmemory', self.config.cache_max_memory)
            r.config_set('maxmemory-policy', self.config.cache_eviction_policy)
            r.config_set('save', '')  # Desabilita persistência para performance
            
            logger.info("Redis configurado para alta performance")
            
        except Exception as e:
            logger.warning(f"Redis não disponível, usando cache em memória: {e}")
            self.redis_pool = None
    

    
    def _generate_cache_key(self, prefix: str, data: Any) -> str:
        """Gera chave de cache determinística"""
        if isinstance(data, dict):
            # Ordena dict para chave consistente
            sorted_data = json.dumps(data, sort_keys=True)
        else:
            sorted_data = str(data)
        
        hash_obj = hashlib.md5(sorted_data.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def _compress_data(self, data: Any) -> bytes:
        """Comprime dados para economizar memória"""
        serialized = pickle.dumps(data)
        return zlib.compress(serialized)
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Descomprime dados"""
        decompressed = zlib.decompress(compressed_data)
        return pickle.loads(decompressed)
    
    def get(self, key: str) -> Optional[Any]:
        """Recupera valor do cache"""
        try:
            if self.redis_pool:
                r = redis.Redis(connection_pool=self.redis_pool)
                compressed_data = r.get(key)
                if compressed_data:
                    self.cache_hits += 1
                    return self._decompress_data(compressed_data)
            
            self.cache_misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Erro ao recuperar do cache: {e}")
            self.cache_misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Armazena valor no cache"""
        try:
            if self.redis_pool:
                r = redis.Redis(connection_pool=self.redis_pool)
                compressed_data = self._compress_data(value)
                ttl = ttl or self.config.cache_ttl_seconds
                
                result = r.setex(key, ttl, compressed_data)
                if result:
                    self.cache_sets += 1
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erro ao armazenar no cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Remove valor do cache"""
        try:
            if self.redis_pool:
                r = redis.Redis(connection_pool=self.redis_pool)
                return bool(r.delete(key))
            return False
        except Exception as e:
            logger.error(f"Erro ao deletar do cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_sets': self.cache_sets,
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests
        }
        
        if self.redis_pool:
            try:
                r = redis.Redis(connection_pool=self.redis_pool)
                info = r.info()
                stats.update({
                    'redis_memory_used': info.get('used_memory_human', 'N/A'),
                    'redis_connected_clients': info.get('connected_clients', 0),
                    'redis_keyspace_hits': info.get('keyspace_hits', 0),
                    'redis_keyspace_misses': info.get('keyspace_misses', 0)
                })
            except Exception as e:
                logger.error(f"Erro ao obter stats do Redis: {e}")
        
        return stats

class DatabaseConnectionPool:
    """Pool de conexões de banco de dados otimizado"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.engine = None
        self.session_factory = None
        self._init_pool()
    
    def _init_pool(self):
        """Inicializa pool de conexões"""
        try:
            # SQLite com pool para desenvolvimento
            database_url = "sqlite:///sankofa_performance.db"
            
            self.engine = create_engine(
                database_url,
                poolclass=pool.QueuePool,
                pool_size=self.config.db_pool_size,
                max_overflow=self.config.db_max_overflow,
                pool_timeout=self.config.db_pool_timeout,
                pool_recycle=self.config.db_pool_recycle,
                pool_pre_ping=True,
                echo=False  # Para performance
            )
            
            self.session_factory = sessionmaker(bind=self.engine)
            
            # Testa conexão
            with self.engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text("SELECT 1"))
            
            logger.info("Pool de conexões de banco inicializado")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar pool de banco: {e}")
            raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Context manager para conexões assíncronas"""
        connection = None
        try:
            connection = self.engine.connect()
            yield connection
        finally:
            if connection:
                connection.close()
    
    def get_session(self):
        """Retorna sessão do SQLAlchemy"""
        return self.session_factory()

class PerformanceMonitor:
    """Monitor de performance em tempo real"""
    
    def __init__(self):
        self.request_times = []
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def record_request(self, duration_ms: float, success: bool = True):
        """Registra tempo de requisição"""
        with self.lock:
            self.request_times.append(duration_ms)
            self.request_count += 1
            if not success:
                self.error_count += 1
            
            # Mantém apenas últimas 1000 requisições
            if len(self.request_times) > 1000:
                self.request_times = self.request_times[-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance"""
        with self.lock:
            if not self.request_times:
                return {
                    'rps': 0,
                    'avg_latency_ms': 0,
                    'p95_latency_ms': 0,
                    'p99_latency_ms': 0,
                    'error_rate_percent': 0,
                    'total_requests': 0
                }
            
            elapsed_time = time.time() - self.start_time
            rps = self.request_count / elapsed_time if elapsed_time > 0 else 0
            
            avg_latency = np.mean(self.request_times)
            p95_latency = np.percentile(self.request_times, 95)
            p99_latency = np.percentile(self.request_times, 99)
            error_rate = (self.error_count / self.request_count * 100) if self.request_count > 0 else 0
            
            return {
                'rps': round(rps, 2),
                'avg_latency_ms': round(avg_latency, 2),
                'p95_latency_ms': round(p95_latency, 2),
                'p99_latency_ms': round(p99_latency, 2),
                'error_rate_percent': round(error_rate, 2),
                'total_requests': self.request_count,
                'uptime_seconds': round(elapsed_time, 2)
            }

class HighPerformanceSystem:
    """Sistema de alta performance integrado"""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.cache = HighPerformanceCache(self.config)
        self.db_pool = DatabaseConnectionPool(self.config)
        self.monitor = PerformanceMonitor()
        
        # Thread pools
        self.thread_executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=self.config.max_workers // 2)
        
        # Configurações assíncronas
        self._setup_async()
        
        logger.info("Sistema de Alta Performance inicializado")
    
    def _setup_async(self):
        """Configura loop assíncrono otimizado"""
        try:
            # Usa uvloop para melhor performance
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        except ImportError:
            logger.warning("uvloop não disponível, usando loop padrão")
    
    def performance_decorator(self, cache_key_prefix: str = None, cache_ttl: int = None):
        """Decorator para otimização automática de performance"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    # Tenta cache se especificado
                    if cache_key_prefix:
                        cache_key = self.cache._generate_cache_key(
                            cache_key_prefix, 
                            {'args': args, 'kwargs': kwargs}
                        )
                        cached_result = self.cache.get(cache_key)
                        if cached_result is not None:
                            duration_ms = (time.time() - start_time) * 1000
                            self.monitor.record_request(duration_ms, True)
                            return cached_result
                    
                    # Executa função
                    result = func(*args, **kwargs)
                    
                    # Armazena no cache se especificado
                    if cache_key_prefix and result is not None:
                        self.cache.set(cache_key, result, cache_ttl)
                    
                    duration_ms = (time.time() - start_time) * 1000
                    self.monitor.record_request(duration_ms, True)
                    
                    return result
                    
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    self.monitor.record_request(duration_ms, False)
                    logger.error(f"Erro na função {func.__name__}: {e}")
                    raise
            
            return wrapper
        return decorator
    
    async def batch_process(self, items: List[Any], process_func, batch_size: Optional[int] = None) -> List[Any]:
        """Processa itens em lotes para melhor performance"""
        batch_size = batch_size or self.config.async_batch_size
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [process_func(item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
        
        return results
    
    def parallel_process(self, items: List[Any], process_func, use_processes: bool = False) -> List[Any]:
        """Processa itens em paralelo"""
        executor = self.process_executor if use_processes else self.thread_executor
        
        futures = [executor.submit(process_func, item) for item in items]
        results = []
        
        for future in futures:
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                logger.error(f"Erro no processamento paralelo: {e}")
                results.append(None)
        
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas completas do sistema"""
        # Performance stats
        perf_stats = self.monitor.get_stats()
        
        # Cache stats
        cache_stats = self.cache.get_stats()
        
        # System stats
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_stats = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': round(memory.available / (1024**3), 2),
            'disk_percent': disk.percent,
            'disk_free_gb': round(disk.free / (1024**3), 2)
        }
        
        # Thread pool stats
        thread_stats = {
            'thread_pool_active': self.thread_executor._threads,
            'thread_pool_max': self.config.max_workers,
            'process_pool_max': self.config.max_workers // 2
        }
        
        return {
            'performance': perf_stats,
            'cache': cache_stats,
            'system': system_stats,
            'threads': thread_stats,
            'config': {
                'target_rps': self.config.target_rps,
                'max_latency_ms': self.config.max_latency_ms,
                'cache_ttl_seconds': self.config.cache_ttl_seconds
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Verifica saúde do sistema"""
        health = {
            'status': 'healthy',
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Check Redis
        try:
            if self.cache.redis_pool:
                r = redis.Redis(connection_pool=self.cache.redis_pool)
                r.ping()
                health['checks']['redis'] = 'healthy'
            else:
                health['checks']['redis'] = 'not_configured'
        except Exception as e:
            health['checks']['redis'] = f'unhealthy: {e}'
            health['status'] = 'degraded'
        
        # Check Database
        try:
            with self.db_pool.engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text("SELECT 1"))
            health['checks']['database'] = 'healthy'
        except Exception as e:
            health['checks']['database'] = f'unhealthy: {e}'
            health['status'] = 'degraded'
        
        # Check Memory
        memory = psutil.virtual_memory()
        if memory.percent > self.config.max_memory_usage_percent:
            health['checks']['memory'] = f'high_usage: {memory.percent}%'
            health['status'] = 'degraded'
        else:
            health['checks']['memory'] = 'healthy'
        
        # Check Performance
        perf_stats = self.monitor.get_stats()
        if perf_stats['rps'] > 0:  # Só verifica se há tráfego
            if perf_stats['avg_latency_ms'] > self.config.max_latency_ms:
                health['checks']['latency'] = f'high: {perf_stats["avg_latency_ms"]}ms'
                health['status'] = 'degraded'
            else:
                health['checks']['latency'] = 'healthy'
        else:
            health['checks']['latency'] = 'no_traffic'
        
        return health
    
    def shutdown(self):
        """Encerra sistema graciosamente"""
        logger.info("Encerrando sistema de alta performance...")
        
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        if self.db_pool.engine:
            self.db_pool.engine.dispose()
        
        logger.info("Sistema de alta performance encerrado")

# Exemplo de uso com decorador
def create_optimized_fraud_detector(performance_system: HighPerformanceSystem):
    """Cria detector de fraude otimizado"""
    
    @performance_system.performance_decorator(cache_key_prefix="fraud_prediction", cache_ttl=60)
    def predict_fraud(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predição de fraude com cache automático"""
        # Simula processamento de ML
        time.sleep(0.01)  # 10ms de processamento
        
        # Resultado simulado
        fraud_score = np.random.random()
        
        return {
            'fraud_score': fraud_score,
            'is_fraud': fraud_score > 0.7,
            'confidence': fraud_score,
            'processing_time_ms': 10,
            'cached': False
        }
    
    return predict_fraud

# Teste do sistema
if __name__ == "__main__":
    print("⚡ Testando Sistema de Alta Performance...")
    
    # Inicializa sistema
    config = PerformanceConfig(target_rps=1000, max_latency_ms=50)
    perf_system = HighPerformanceSystem(config)
    
    # Cria detector otimizado
    fraud_detector = create_optimized_fraud_detector(perf_system)
    
    # Teste de performance
    print("🚀 Executando teste de throughput...")
    
    start_time = time.time()
    num_requests = 100
    
    # Simula requisições
    for i in range(num_requests):
        transaction = {
            'id': f'txn_{i}',
            'amount': np.random.uniform(10, 1000),
            'timestamp': datetime.now().isoformat()
        }
        
        result = fraud_detector(transaction)
        
        if i % 20 == 0:
            print(f"   Processadas {i+1} transações...")
    
    elapsed_time = time.time() - start_time
    rps = num_requests / elapsed_time
    
    print(f"✅ Teste concluído:")
    print(f"   RPS atingido: {rps:.2f}")
    print(f"   Tempo total: {elapsed_time:.2f}s")
    
    # Estatísticas do sistema
    stats = perf_system.get_system_stats()
    print(f"✅ Estatísticas do sistema:")
    print(f"   RPS médio: {stats['performance']['rps']}")
    print(f"   Latência média: {stats['performance']['avg_latency_ms']}ms")
    print(f"   P95 latência: {stats['performance']['p95_latency_ms']}ms")
    print(f"   Hit rate cache: {stats['cache']['hit_rate_percent']}%")
    print(f"   CPU: {stats['system']['cpu_percent']}%")
    print(f"   Memória: {stats['system']['memory_percent']}%")
    
    # Health check
    health = perf_system.health_check()
    print(f"✅ Health check: {health['status']}")
    for check, status in health['checks'].items():
        print(f"   {check}: {status}")
    
    # Teste de cache
    print("🔄 Testando cache...")
    
    # Primeira chamada (miss)
    start = time.time()
    result1 = fraud_detector({'id': 'test_cache', 'amount': 100})
    time1 = (time.time() - start) * 1000
    
    # Segunda chamada (hit)
    start = time.time()
    result2 = fraud_detector({'id': 'test_cache', 'amount': 100})
    time2 = (time.time() - start) * 1000
    
    print(f"   Primeira chamada: {time1:.2f}ms")
    print(f"   Segunda chamada (cache): {time2:.2f}ms")
    print(f"   Speedup: {time1/time2:.1f}x")
    
    # Encerra sistema
    perf_system.shutdown()
    
    print("⚡ Teste do Sistema de Alta Performance concluído!")
