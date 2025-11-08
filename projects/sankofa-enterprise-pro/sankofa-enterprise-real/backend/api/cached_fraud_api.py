#!/usr/bin/env python3
"""
API de Fraude com Cache Integrado para Sankofa Enterprise Pro
Integra sistema de cache Redis para alta performance na análise de fraude
"""

import os
import sys
import hashlib
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify, g
from functools import wraps
import json

# Adiciona o diretório pai ao path para importações
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache.redis_cache_system import RedisCacheSystem, CacheConfig, FraudCacheManager
from cache.distributed_fraud_cache import DistributedFraudCache
from security.enterprise_security_system import EnterpriseSecuritySystem

logger = logging.getLogger(__name__)

class CachedFraudAPI:
    """API de detecção de fraude com cache integrado para alta performance"""
    
    def __init__(self):
        self.app = Flask(__name__)
        
        # Inicializa sistemas
        self.cache_config = CacheConfig()
        self.redis_cache = RedisCacheSystem(self.cache_config)
        self.fraud_cache = FraudCacheManager(self.redis_cache)
        self.distributed_cache = DistributedFraudCache(self.redis_cache)
        self.security_system = EnterpriseSecuritySystem()
        
        # Métricas de performance
        self.performance_metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'avg_response_time': 0,
            'cache_hit_rate': 0
        }
        
        self._register_routes()
        logger.info("API de Fraude com Cache inicializada")
    
    def _update_metrics(self, cache_hit: bool, response_time: float):
        """Atualiza métricas de performance"""
        self.performance_metrics['total_requests'] += 1
        
        if cache_hit:
            self.performance_metrics['cache_hits'] += 1
        else:
            self.performance_metrics['cache_misses'] += 1
        
        # Calcula hit rate
        total = self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']
        if total > 0:
            self.performance_metrics['cache_hit_rate'] = self.performance_metrics['cache_hits'] / total
        
        # Atualiza tempo médio de resposta
        current_avg = self.performance_metrics['avg_response_time']
        total_requests = self.performance_metrics['total_requests']
        self.performance_metrics['avg_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
    
    def _generate_input_hash(self, data: Dict[str, Any]) -> str:
        """Gera hash determinístico dos dados de entrada"""
        # Remove campos que não afetam a análise (como timestamps)
        filtered_data = {k: v for k, v in data.items() 
                        if k not in ['timestamp', 'request_id', 'session_id']}
        
        # Ordena e serializa
        sorted_data = json.dumps(filtered_data, sort_keys=True, default=str)
        return hashlib.md5(sorted_data.encode()).hexdigest()
    
    def _register_routes(self):
        """Registra rotas da API com cache"""
        
        @self.app.route('/api/v1/fraud/analyze-cached', methods=['POST'])
        @self.security_system.require_auth()
        def analyze_transaction_cached():
            """Análise de transação com cache inteligente"""
            start_time = time.time()
            cache_hit = False
            
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Dados da transação são obrigatórios'}), 400
                
                # Gera hash dos dados de entrada
                input_hash = self._generate_input_hash(data)
                
                # Verifica cache primeiro
                cached_result = self.distributed_cache.get_ensemble_result(input_hash)
                
                if cached_result:
                    cache_hit = True
                    response_time = time.time() - start_time
                    self._update_metrics(cache_hit, response_time)
                    
                    return jsonify({
                        'success': True,
                        'message': 'Análise obtida do cache',
                        'data': cached_result,
                        'cached': True,
                        'response_time_ms': response_time * 1000
                    }), 200
                
                # Análise não está em cache - processa
                analysis_result = self._perform_fraud_analysis(data)
                
                # Cache o resultado
                self.distributed_cache.cache_ensemble_result(input_hash, analysis_result)
                
                response_time = time.time() - start_time
                self._update_metrics(cache_hit, response_time)
                
                return jsonify({
                    'success': True,
                    'message': 'Análise concluída e cacheada',
                    'data': analysis_result,
                    'cached': False,
                    'response_time_ms': response_time * 1000
                }), 200
                
            except Exception as e:
                logger.error(f"Erro na análise cacheada: {e}")
                response_time = time.time() - start_time
                self._update_metrics(False, response_time)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/fraud/user-profile-cached', methods=['GET'])
        @self.security_system.require_auth()
        def get_user_profile_cached():
            """Obtém perfil de usuário com cache"""
            start_time = time.time()
            cache_hit = False
            
            try:
                user_id = request.args.get('user_id')
                if not user_id:
                    return jsonify({'error': 'user_id é obrigatório'}), 400
                
                # Verifica cache
                cached_profile = self.distributed_cache.get_user_behavior(user_id)
                
                if cached_profile:
                    cache_hit = True
                    response_time = time.time() - start_time
                    self._update_metrics(cache_hit, response_time)
                    
                    return jsonify({
                        'success': True,
                        'data': cached_profile,
                        'cached': True,
                        'response_time_ms': response_time * 1000
                    }), 200
                
                # Gera perfil (simulado)
                profile = self._generate_user_profile(user_id)
                
                # Cache o perfil
                self.distributed_cache.cache_user_behavior(user_id, profile)
                
                response_time = time.time() - start_time
                self._update_metrics(cache_hit, response_time)
                
                return jsonify({
                    'success': True,
                    'data': profile,
                    'cached': False,
                    'response_time_ms': response_time * 1000
                }), 200
                
            except Exception as e:
                logger.error(f"Erro ao obter perfil: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/fraud/velocity-check', methods=['POST'])
        @self.security_system.require_auth()
        def velocity_check():
            """Verificação de velocidade com contadores em cache"""
            start_time = time.time()
            
            try:
                data = request.get_json()
                entity_type = data.get('entity_type')  # card, user, merchant, ip
                entity_id = data.get('entity_id')
                time_window = data.get('time_window', '1h')  # 1h, 24h, 7d
                
                if not entity_type or not entity_id:
                    return jsonify({'error': 'entity_type e entity_id são obrigatórios'}), 400
                
                # Incrementa contador
                count = self.distributed_cache.increment_velocity_counter(
                    entity_type, entity_id, time_window
                )
                
                # Define limites por tipo
                limits = {
                    'card': {'1h': 10, '24h': 50, '7d': 200},
                    'user': {'1h': 20, '24h': 100, '7d': 500},
                    'merchant': {'1h': 1000, '24h': 10000, '7d': 50000},
                    'ip': {'1h': 50, '24h': 200, '7d': 1000}
                }
                
                limit = limits.get(entity_type, {}).get(time_window, 100)
                is_exceeded = count > limit
                
                response_time = time.time() - start_time
                
                return jsonify({
                    'success': True,
                    'data': {
                        'entity_type': entity_type,
                        'entity_id': entity_id,
                        'time_window': time_window,
                        'current_count': count,
                        'limit': limit,
                        'is_exceeded': is_exceeded,
                        'risk_level': 'HIGH' if is_exceeded else 'NORMAL'
                    },
                    'response_time_ms': response_time * 1000
                }), 200
                
            except Exception as e:
                logger.error(f"Erro na verificação de velocidade: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/fraud/blacklist-check', methods=['POST'])
        @self.security_system.require_auth()
        def blacklist_check():
            """Verificação de blacklist com cache"""
            start_time = time.time()
            
            try:
                data = request.get_json()
                list_type = data.get('list_type')  # ip, card, email, device
                identifier = data.get('identifier')
                
                if not list_type or not identifier:
                    return jsonify({'error': 'list_type e identifier são obrigatórios'}), 400
                
                # Verifica blacklist
                is_blacklisted = self.distributed_cache.is_blacklisted(list_type, identifier)
                
                response_time = time.time() - start_time
                
                return jsonify({
                    'success': True,
                    'data': {
                        'list_type': list_type,
                        'identifier': identifier,
                        'is_blacklisted': is_blacklisted,
                        'risk_level': 'CRITICAL' if is_blacklisted else 'NORMAL'
                    },
                    'response_time_ms': response_time * 1000
                }), 200
                
            except Exception as e:
                logger.error(f"Erro na verificação de blacklist: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/fraud/cache-stats', methods=['GET'])
        @self.security_system.require_auth()
        def cache_statistics():
            """Estatísticas do sistema de cache"""
            try:
                redis_stats = self.redis_cache.get_stats()
                distributed_stats = self.distributed_cache.get_comprehensive_stats()
                
                return jsonify({
                    'success': True,
                    'data': {
                        'performance_metrics': self.performance_metrics,
                        'redis_cache': redis_stats,
                        'distributed_cache': distributed_stats
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"Erro ao obter estatísticas: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/fraud/cache-clear', methods=['POST'])
        @self.security_system.require_auth()
        @self.security_system.require_permission('system_config')
        def clear_cache():
            """Limpa cache de fraude"""
            try:
                data = request.get_json() or {}
                category = data.get('category')  # Opcional: limpar categoria específica
                
                if category:
                    results = self.distributed_cache.clear_fraud_cache(category)
                else:
                    results = self.distributed_cache.clear_fraud_cache()
                
                return jsonify({
                    'success': True,
                    'message': 'Cache limpo com sucesso',
                    'data': results
                }), 200
                
            except Exception as e:
                logger.error(f"Erro ao limpar cache: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/fraud/cache-warm-up', methods=['POST'])
        @self.security_system.require_auth()
        @self.security_system.require_permission('system_config')
        def warm_up_cache():
            """Aquece cache com dados frequentes"""
            try:
                # Dados de exemplo para warm-up
                warm_up_data = {
                    'user': {
                        'user_behavior': {
                            ('user_123',): {
                                'avg_transaction_amount': 150.00,
                                'preferred_merchants': ['grocery', 'gas'],
                                'typical_hours': [9, 12, 18]
                            },
                            ('user_456',): {
                                'avg_transaction_amount': 300.00,
                                'preferred_merchants': ['restaurant', 'shopping'],
                                'typical_hours': [10, 14, 19]
                            }
                        }
                    },
                    'geo': {
                        'geo_location': {
                            ('192.168.1.1',): {
                                'country': 'BR',
                                'city': 'São Paulo',
                                'latitude': -23.5505,
                                'longitude': -46.6333
                            }
                        }
                    }
                }
                
                self.distributed_cache.warm_up_cache(warm_up_data)
                
                return jsonify({
                    'success': True,
                    'message': 'Cache aquecido com sucesso'
                }), 200
                
            except Exception as e:
                logger.error(f"Erro no warm-up: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _perform_fraud_analysis(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simula análise de fraude (seria integrado com ML engine real)"""
        # Simula processamento complexo
        time.sleep(0.1)  # Simula latência de processamento
        
        amount = transaction_data.get('amount', 0)
        merchant = transaction_data.get('merchant', '')
        
        # Lógica simplificada de scoring
        fraud_score = 0.0
        
        if amount > 1000:
            fraud_score += 0.3
        if 'casino' in merchant.lower():
            fraud_score += 0.5
        if transaction_data.get('hour', 12) < 6 or transaction_data.get('hour', 12) > 23:
            fraud_score += 0.2
        
        risk_level = 'LOW'
        if fraud_score > 0.7:
            risk_level = 'HIGH'
        elif fraud_score > 0.4:
            risk_level = 'MEDIUM'
        
        return {
            'fraud_score': min(fraud_score, 1.0),
            'risk_level': risk_level,
            'reasons': ['High amount', 'Unusual merchant', 'Off-hours transaction'],
            'timestamp': datetime.now().isoformat(),
            'model_version': '1.0.0'
        }
    
    def _generate_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Simula geração de perfil de usuário"""
        # Simula processamento
        time.sleep(0.05)
        
        return {
            'user_id': user_id,
            'avg_transaction_amount': 250.00,
            'transaction_frequency': 15,  # por mês
            'preferred_merchants': ['grocery', 'gas', 'restaurant'],
            'typical_hours': [9, 12, 18, 20],
            'risk_score': 0.2,
            'account_age_days': 365,
            'last_updated': datetime.now().isoformat()
        }
    
    def run(self, host='0.0.0.0', port=8444, debug=False):
        """Executa a API"""
        logger.info(f"🚀 API de Fraude com Cache iniciada em http://{host}:{port}")
        logger.info("⚡ Sistema de cache Redis integrado")
        logger.info("📊 Métricas de performance ativadas")
        
        self.app.run(host=host, port=port, debug=debug, threaded=True)

# Instância global da API
cached_api = CachedFraudAPI()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='API de Fraude com Cache')
    parser.add_argument('--host', default='0.0.0.0', help='Host da API')
    parser.add_argument('--port', type=int, default=8444, help='Porta da API')
    parser.add_argument('--debug', action='store_true', help='Modo debug')
    
    args = parser.parse_args()
    
    cached_api.run(host=args.host, port=args.port, debug=args.debug)
