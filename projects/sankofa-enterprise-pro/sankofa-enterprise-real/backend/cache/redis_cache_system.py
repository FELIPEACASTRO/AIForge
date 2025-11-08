#!/usr/bin/env python3
"""
Sistema de Cache Redis Enterprise para Sankofa Enterprise Pro
Implementa cache distribuído de alta performance para análise de fraude
"""

import redis
import json
import pickle
import hashlib
import time
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuração do sistema de cache"""
    host: str = 'localhost'
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 100
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    default_ttl: int = 3600  # 1 hora
    max_memory_policy: str = 'allkeys-lru'

class RedisConnectionManager:
    """Gerenciador de conexões Redis com pool e failover"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.pool = None
        self.async_pool = None
        self._lock = threading.Lock()
        self._health_check_thread = None
        self._is_healthy = True
        
        self._init_connection_pool()
        self._start_health_check()
    
    def _init_connection_pool(self):
        """Inicializa pool de conexões Redis"""
        try:
            self.pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                decode_responses=False  # Para suportar dados binários
            )
            
            # Testa conexão
            client = redis.Redis(connection_pool=self.pool)
            client.ping()
            
            logger.info(f"Pool de conexões Redis inicializado - {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar pool Redis: {e}")
            self._is_healthy = False
            raise
    
    def _init_async_pool(self):
        """Pool assíncrono não implementado nesta versão"""
        pass
    
    def get_client(self) -> redis.Redis:
        """Obtém cliente Redis do pool"""
        if not self._is_healthy:
            raise ConnectionError("Redis não está saudável")
        
        return redis.Redis(connection_pool=self.pool)
    
    def get_async_client(self):
        """Cliente assíncrono não implementado nesta versão"""
        raise NotImplementedError("Cliente assíncrono não disponível")
    
    def _start_health_check(self):
        """Inicia thread de health check"""
        def health_check():
            while True:
                try:
                    client = redis.Redis(connection_pool=self.pool)
                    client.ping()
                    if not self._is_healthy:
                        logger.info("Redis voltou a ficar saudável")
                        self._is_healthy = True
                except Exception as e:
                    if self._is_healthy:
                        logger.error(f"Redis ficou não saudável: {e}")
                        self._is_healthy = False
                
                time.sleep(self.config.health_check_interval)
        
        self._health_check_thread = threading.Thread(target=health_check, daemon=True)
        self._health_check_thread.start()
    
    def is_healthy(self) -> bool:
        """Verifica se Redis está saudável"""
        return self._is_healthy

class CacheSerializer:
    """Serializador otimizado para diferentes tipos de dados"""
    
    @staticmethod
    def serialize(data: Any) -> bytes:
        """Serializa dados para armazenamento"""
        if isinstance(data, (str, int, float, bool)):
            return json.dumps(data).encode('utf-8')
        elif isinstance(data, (dict, list, tuple)):
            return json.dumps(data, default=str).encode('utf-8')
        else:
            # Usa pickle para objetos complexos
            return pickle.dumps(data)
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Deserializa dados do cache"""
        try:
            # Tenta JSON primeiro (mais rápido)
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fallback para pickle
            return pickle.loads(data)

class CacheKeyManager:
    """Gerenciador de chaves de cache com namespaces"""
    
    def __init__(self, namespace: str = "sankofa"):
        self.namespace = namespace
    
    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Gera chave de cache determinística"""
        # Cria hash dos argumentos
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        
        return f"{self.namespace}:{prefix}:{key_hash}"
    
    def pattern_key(self, prefix: str) -> str:
        """Gera padrão para busca de chaves"""
        return f"{self.namespace}:{prefix}:*"

class RedisCacheSystem:
    """Sistema de cache Redis enterprise com recursos avançados"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.connection_manager = RedisConnectionManager(self.config)
        self.serializer = CacheSerializer()
        self.key_manager = CacheKeyManager()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Métricas
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        
        logger.info("Sistema de Cache Redis Enterprise inicializado")
    
    def _update_stats(self, operation: str):
        """Atualiza estatísticas"""
        self.stats[operation] = self.stats.get(operation, 0) + 1
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtém valor do cache"""
        try:
            client = self.connection_manager.get_client()
            data = client.get(key)
            
            if data is None:
                self._update_stats('misses')
                return default
            
            self._update_stats('hits')
            return self.serializer.deserialize(data)
            
        except Exception as e:
            logger.error(f"Erro ao obter cache {key}: {e}")
            self._update_stats('errors')
            return default
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Define valor no cache"""
        try:
            client = self.connection_manager.get_client()
            serialized_data = self.serializer.serialize(value)
            
            ttl = ttl or self.config.default_ttl
            result = client.setex(key, ttl, serialized_data)
            
            self._update_stats('sets')
            return result
            
        except Exception as e:
            logger.error(f"Erro ao definir cache {key}: {e}")
            self._update_stats('errors')
            return False
    
    def delete(self, key: str) -> bool:
        """Remove valor do cache"""
        try:
            client = self.connection_manager.get_client()
            result = client.delete(key) > 0
            
            self._update_stats('deletes')
            return result
            
        except Exception as e:
            logger.error(f"Erro ao deletar cache {key}: {e}")
            self._update_stats('errors')
            return False
    
    def exists(self, key: str) -> bool:
        """Verifica se chave existe"""
        try:
            client = self.connection_manager.get_client()
            return client.exists(key) > 0
        except Exception as e:
            logger.error(f"Erro ao verificar existência {key}: {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """Define TTL para chave existente"""
        try:
            client = self.connection_manager.get_client()
            return client.expire(key, ttl)
        except Exception as e:
            logger.error(f"Erro ao definir TTL {key}: {e}")
            return False
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Incrementa valor numérico"""
        try:
            client = self.connection_manager.get_client()
            return client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Erro ao incrementar {key}: {e}")
            return 0
    
    def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """Obtém múltiplos valores"""
        try:
            client = self.connection_manager.get_client()
            values = client.mget(keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self.serializer.deserialize(value)
                    self._update_stats('hits')
                else:
                    self._update_stats('misses')
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao obter múltiplos valores: {e}")
            self._update_stats('errors')
            return {}
    
    def set_multiple(self, mapping: Dict[str, Any], ttl: int = None) -> bool:
        """Define múltiplos valores"""
        try:
            client = self.connection_manager.get_client()
            
            # Serializa todos os valores
            serialized_mapping = {
                key: self.serializer.serialize(value)
                for key, value in mapping.items()
            }
            
            # Define valores
            pipe = client.pipeline()
            for key, value in serialized_mapping.items():
                ttl_value = ttl or self.config.default_ttl
                pipe.setex(key, ttl_value, value)
            
            results = pipe.execute()
            
            self._update_stats('sets')
            return all(results)
            
        except Exception as e:
            logger.error(f"Erro ao definir múltiplos valores: {e}")
            self._update_stats('errors')
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Remove chaves por padrão"""
        try:
            client = self.connection_manager.get_client()
            keys = client.keys(pattern)
            
            if keys:
                deleted = client.delete(*keys)
                self._update_stats('deletes')
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Erro ao limpar padrão {pattern}: {e}")
            self._update_stats('errors')
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do cache"""
        try:
            client = self.connection_manager.get_client()
            info = client.info()
            
            hit_rate = 0
            if self.stats['hits'] + self.stats['misses'] > 0:
                hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses'])
            
            return {
                'operations': self.stats.copy(),
                'hit_rate': hit_rate,
                'redis_info': {
                    'used_memory': info.get('used_memory_human'),
                    'connected_clients': info.get('connected_clients'),
                    'total_commands_processed': info.get('total_commands_processed'),
                    'keyspace_hits': info.get('keyspace_hits'),
                    'keyspace_misses': info.get('keyspace_misses')
                },
                'health': self.connection_manager.is_healthy()
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {e}")
            return {'error': str(e)}

class FraudCacheManager:
    """Gerenciador de cache específico para detecção de fraude"""
    
    def __init__(self, cache_system: RedisCacheSystem):
        self.cache = cache_system
        self.key_manager = CacheKeyManager("fraud")
        
        # TTLs específicos por tipo de dados
        self.ttls = {
            'transaction_analysis': 300,      # 5 minutos
            'user_profile': 3600,            # 1 hora
            'merchant_profile': 7200,        # 2 horas
            'model_predictions': 1800,       # 30 minutos
            'feature_vectors': 600,          # 10 minutos
            'risk_scores': 900,              # 15 minutos
            'blacklist': 86400,              # 24 horas
            'whitelist': 86400,              # 24 horas
            'velocity_counters': 3600,       # 1 hora
            'session_data': 1800             # 30 minutos
        }
    
    def cache_transaction_analysis(self, transaction_id: str, analysis_result: Dict[str, Any]) -> bool:
        """Cache resultado de análise de transação"""
        key = self.key_manager.generate_key('transaction_analysis', transaction_id)
        return self.cache.set(key, analysis_result, self.ttls['transaction_analysis'])
    
    def get_transaction_analysis(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Obtém análise de transação do cache"""
        key = self.key_manager.generate_key('transaction_analysis', transaction_id)
        return self.cache.get(key)
    
    def cache_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """Cache perfil de usuário"""
        key = self.key_manager.generate_key('user_profile', user_id)
        return self.cache.set(key, profile_data, self.ttls['user_profile'])
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtém perfil de usuário do cache"""
        key = self.key_manager.generate_key('user_profile', user_id)
        return self.cache.get(key)
    
    def cache_model_prediction(self, model_name: str, input_hash: str, prediction: Dict[str, Any]) -> bool:
        """Cache predição de modelo"""
        key = self.key_manager.generate_key('model_predictions', model_name, input_hash)
        return self.cache.set(key, prediction, self.ttls['model_predictions'])
    
    def get_model_prediction(self, model_name: str, input_hash: str) -> Optional[Dict[str, Any]]:
        """Obtém predição de modelo do cache"""
        key = self.key_manager.generate_key('model_predictions', model_name, input_hash)
        return self.cache.get(key)
    
    def increment_velocity_counter(self, counter_type: str, identifier: str, window: str) -> int:
        """Incrementa contador de velocidade"""
        key = self.key_manager.generate_key('velocity_counters', counter_type, identifier, window)
        
        # Incrementa e define TTL se for nova chave
        client = self.cache.connection_manager.get_client()
        count = client.incr(key)
        
        if count == 1:  # Nova chave
            client.expire(key, self.ttls['velocity_counters'])
        
        return count
    
    def get_velocity_counter(self, counter_type: str, identifier: str, window: str) -> int:
        """Obtém contador de velocidade"""
        key = self.key_manager.generate_key('velocity_counters', counter_type, identifier, window)
        return self.cache.get(key, 0)
    
    def is_blacklisted(self, list_type: str, identifier: str) -> bool:
        """Verifica se item está na blacklist"""
        key = self.key_manager.generate_key('blacklist', list_type, identifier)
        return self.cache.exists(key)
    
    def add_to_blacklist(self, list_type: str, identifier: str, reason: str = None) -> bool:
        """Adiciona item à blacklist"""
        key = self.key_manager.generate_key('blacklist', list_type, identifier)
        data = {'added_at': datetime.now().isoformat(), 'reason': reason}
        return self.cache.set(key, data, self.ttls['blacklist'])
    
    def clear_fraud_cache(self) -> Dict[str, int]:
        """Limpa todo o cache de fraude"""
        patterns = [
            self.key_manager.pattern_key('transaction_analysis'),
            self.key_manager.pattern_key('user_profile'),
            self.key_manager.pattern_key('model_predictions'),
            self.key_manager.pattern_key('velocity_counters')
        ]
        
        results = {}
        for pattern in patterns:
            deleted = self.cache.clear_pattern(pattern)
            results[pattern] = deleted
        
        return results

def cache_result(cache_manager: FraudCacheManager, cache_type: str, ttl: int = None):
    """Decorator para cache automático de resultados de funções"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Gera chave baseada na função e argumentos
            key = cache_manager.key_manager.generate_key(
                f"{cache_type}_{func.__name__}",
                *args,
                **kwargs
            )
            
            # Tenta obter do cache
            cached_result = cache_manager.cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Executa função e cache resultado
            result = func(*args, **kwargs)
            
            if result is not None:
                ttl_value = ttl or cache_manager.ttls.get(cache_type, 3600)
                cache_manager.cache.set(key, result, ttl_value)
            
            return result
        
        return wrapper
    return decorator

# Instância global do sistema de cache
cache_config = CacheConfig()
redis_cache_system = RedisCacheSystem(cache_config)
fraud_cache_manager = FraudCacheManager(redis_cache_system)

# Teste do sistema
if __name__ == "__main__":
    print("🚀 Testando Sistema de Cache Redis Enterprise...")
    
    # Testa operações básicas
    test_key = "test:sankofa"
    test_data = {"message": "Hello Sankofa!", "timestamp": datetime.now().isoformat()}
    
    # Set
    success = redis_cache_system.set(test_key, test_data, 60)
    print(f"✅ Set: {success}")
    
    # Get
    retrieved_data = redis_cache_system.get(test_key)
    print(f"✅ Get: {retrieved_data}")
    
    # Testa cache de fraude
    transaction_id = "txn_123456"
    analysis_result = {
        "fraud_score": 0.85,
        "risk_level": "HIGH",
        "reasons": ["Unusual amount", "New merchant"],
        "timestamp": datetime.now().isoformat()
    }
    
    # Cache análise
    fraud_cache_manager.cache_transaction_analysis(transaction_id, analysis_result)
    print("✅ Análise de transação cacheada")
    
    # Recupera análise
    cached_analysis = fraud_cache_manager.get_transaction_analysis(transaction_id)
    print(f"✅ Análise recuperada: {cached_analysis['fraud_score']}")
    
    # Testa contador de velocidade
    count = fraud_cache_manager.increment_velocity_counter("card_usage", "1234567890", "1h")
    print(f"✅ Contador de velocidade: {count}")
    
    # Estatísticas
    stats = redis_cache_system.get_stats()
    print(f"✅ Hit rate: {stats['hit_rate']:.2%}")
    print(f"✅ Operações: {stats['operations']}")
    
    print("🚀 Teste do Sistema de Cache Redis Enterprise concluído!")
