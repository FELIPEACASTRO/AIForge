#!/usr/bin/env python3
"""
Sistema de Load Balancing e Distribuição de Carga
Sankofa Enterprise Pro - Load Balancer
"""

import logging
import asyncio
import aiohttp
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import statistics
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Estratégias de load balancing"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    HASH_BASED = "hash_based"
    GEOGRAPHIC = "geographic"
    ADAPTIVE = "adaptive"

class ServerStatus(Enum):
    """Status dos servidores"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"

@dataclass
class ServerInstance:
    """Instância de servidor"""
    id: str
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 1000
    current_connections: int = 0
    status: ServerStatus = ServerStatus.HEALTHY
    region: str = "default"
    last_health_check: Optional[str] = None
    response_times: deque = None
    error_count: int = 0
    total_requests: int = 0
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = deque(maxlen=100)  # Últimos 100 tempos de resposta
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def connection_utilization(self) -> float:
        return self.current_connections / self.max_connections if self.max_connections > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        return self.error_count / self.total_requests if self.total_requests > 0 else 0.0

@dataclass
class LoadBalancerConfig:
    """Configuração do load balancer"""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    health_check_interval: int = 30  # segundos
    health_check_timeout: int = 5    # segundos
    max_retries: int = 3
    retry_delay: float = 0.5
    circuit_breaker_threshold: int = 5  # falhas consecutivas
    circuit_breaker_timeout: int = 60   # segundos
    sticky_sessions: bool = False
    session_timeout: int = 3600  # segundos

class CircuitBreaker:
    """Circuit breaker para proteção de servidores"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def record_success(self):
        """Registra sucesso"""
        self.failure_count = 0
        if self.state == "half-open":
            self.state = "closed"
            logger.info("🟢 Circuit breaker fechado - servidor recuperado")
    
    def record_failure(self):
        """Registra falha"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold and self.state == "closed":
            self.state = "open"
            logger.warning(f"🔴 Circuit breaker aberto - {self.failure_count} falhas consecutivas")
    
    def can_attempt(self) -> bool:
        """Verifica se pode tentar requisição"""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = "half-open"
                logger.info("🟡 Circuit breaker meio-aberto - testando servidor")
                return True
            return False
        
        # half-open state
        return True

class StickySessionManager:
    """Gerenciador de sessões sticky"""
    
    def __init__(self, timeout: int = 3600):
        self.timeout = timeout
        self.sessions = {}  # session_id -> (server_id, timestamp)
    
    def get_server_for_session(self, session_id: str) -> Optional[str]:
        """Obtém servidor para uma sessão"""
        if session_id in self.sessions:
            server_id, timestamp = self.sessions[session_id]
            if time.time() - timestamp < self.timeout:
                return server_id
            else:
                # Sessão expirada
                del self.sessions[session_id]
        return None
    
    def bind_session(self, session_id: str, server_id: str):
        """Vincula sessão a um servidor"""
        self.sessions[session_id] = (server_id, time.time())
    
    def cleanup_expired_sessions(self):
        """Remove sessões expiradas"""
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, (_, timestamp) in self.sessions.items()
            if current_time - timestamp >= self.timeout
        ]
        for session_id in expired_sessions:
            del self.sessions[session_id]

class FraudDetectionLoadBalancer:
    """Load balancer especializado para detecção de fraudes"""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.servers = {}  # server_id -> ServerInstance
        self.circuit_breakers = {}  # server_id -> CircuitBreaker
        self.session_manager = StickySessionManager(config.session_timeout)
        
        # Contadores para round robin
        self.round_robin_counter = 0
        
        # Métricas
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'requests_per_second': 0.0,
            'server_distribution': defaultdict(int)
        }
        
        # Health check
        self.health_check_task = None
        self.running = False
        
        logger.info("⚖️ Load Balancer inicializado")
    
    def add_server(self, server: ServerInstance):
        """Adiciona servidor ao pool"""
        self.servers[server.id] = server
        self.circuit_breakers[server.id] = CircuitBreaker(
            self.config.circuit_breaker_threshold,
            self.config.circuit_breaker_timeout
        )
        logger.info(f"➕ Servidor adicionado: {server.id} ({server.url})")
    
    def remove_server(self, server_id: str):
        """Remove servidor do pool"""
        if server_id in self.servers:
            del self.servers[server_id]
            del self.circuit_breakers[server_id]
            logger.info(f"➖ Servidor removido: {server_id}")
    
    def get_healthy_servers(self) -> List[ServerInstance]:
        """Retorna lista de servidores saudáveis"""
        healthy_servers = []
        for server in self.servers.values():
            if (server.status in [ServerStatus.HEALTHY, ServerStatus.DEGRADED] and
                self.circuit_breakers[server.id].can_attempt()):
                healthy_servers.append(server)
        return healthy_servers
    
    def select_server(self, request_data: Dict[str, Any] = None) -> Optional[ServerInstance]:
        """Seleciona servidor baseado na estratégia configurada"""
        healthy_servers = self.get_healthy_servers()
        
        if not healthy_servers:
            logger.error("❌ Nenhum servidor saudável disponível")
            return None
        
        # Verificar sticky session
        if self.config.sticky_sessions and request_data:
            session_id = request_data.get('session_id') or request_data.get('customer_id')
            if session_id:
                server_id = self.session_manager.get_server_for_session(session_id)
                if server_id and server_id in self.servers:
                    server = self.servers[server_id]
                    if server in healthy_servers:
                        return server
        
        # Selecionar baseado na estratégia
        if self.config.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_servers)
        
        elif self.config.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_servers)
        
        elif self.config.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_servers)
        
        elif self.config.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(healthy_servers)
        
        elif self.config.strategy == LoadBalancingStrategy.HASH_BASED:
            return self._hash_based_selection(healthy_servers, request_data)
        
        elif self.config.strategy == LoadBalancingStrategy.GEOGRAPHIC:
            return self._geographic_selection(healthy_servers, request_data)
        
        elif self.config.strategy == LoadBalancingStrategy.ADAPTIVE:
            return self._adaptive_selection(healthy_servers, request_data)
        
        else:
            # Fallback para round robin
            return self._round_robin_selection(healthy_servers)
    
    def _round_robin_selection(self, servers: List[ServerInstance]) -> ServerInstance:
        """Seleção round robin"""
        server = servers[self.round_robin_counter % len(servers)]
        self.round_robin_counter += 1
        return server
    
    def _weighted_round_robin_selection(self, servers: List[ServerInstance]) -> ServerInstance:
        """Seleção round robin ponderada"""
        # Criar lista expandida baseada nos pesos
        weighted_servers = []
        for server in servers:
            weight = max(1, int(server.weight * 10))  # Multiplicar por 10 para granularidade
            weighted_servers.extend([server] * weight)
        
        if weighted_servers:
            server = weighted_servers[self.round_robin_counter % len(weighted_servers)]
            self.round_robin_counter += 1
            return server
        
        return servers[0]
    
    def _least_connections_selection(self, servers: List[ServerInstance]) -> ServerInstance:
        """Seleção por menor número de conexões"""
        return min(servers, key=lambda s: s.current_connections)
    
    def _least_response_time_selection(self, servers: List[ServerInstance]) -> ServerInstance:
        """Seleção por menor tempo de resposta"""
        return min(servers, key=lambda s: s.avg_response_time)
    
    def _hash_based_selection(self, servers: List[ServerInstance], request_data: Dict[str, Any]) -> ServerInstance:
        """Seleção baseada em hash"""
        if not request_data:
            return servers[0]
        
        # Usar customer_id ou transaction_id para hash consistente
        hash_key = request_data.get('customer_id') or request_data.get('id') or 'default'
        hash_value = int(hashlib.md5(str(hash_key).encode()).hexdigest(), 16)
        
        return servers[hash_value % len(servers)]
    
    def _geographic_selection(self, servers: List[ServerInstance], request_data: Dict[str, Any]) -> ServerInstance:
        """Seleção baseada em localização geográfica"""
        if not request_data:
            return servers[0]
        
        # Tentar encontrar servidor na mesma região
        client_region = request_data.get('region', 'default')
        regional_servers = [s for s in servers if s.region == client_region]
        
        if regional_servers:
            return self._least_response_time_selection(regional_servers)
        
        return self._least_response_time_selection(servers)
    
    def _adaptive_selection(self, servers: List[ServerInstance], request_data: Dict[str, Any]) -> ServerInstance:
        """Seleção adaptativa baseada em múltiplos fatores"""
        # Calcular score para cada servidor
        server_scores = []
        
        for server in servers:
            # Fatores: utilização de conexões, tempo de resposta, taxa de erro
            connection_factor = 1.0 - server.connection_utilization
            response_time_factor = 1.0 / (1.0 + server.avg_response_time / 100.0)  # Normalizar
            error_factor = 1.0 - server.error_rate
            weight_factor = server.weight
            
            # Score composto
            score = (connection_factor * 0.3 + 
                    response_time_factor * 0.4 + 
                    error_factor * 0.2 + 
                    weight_factor * 0.1)
            
            server_scores.append((server, score))
        
        # Selecionar servidor com maior score
        return max(server_scores, key=lambda x: x[1])[0]
    
    async def forward_request(self, request_data: Dict[str, Any], endpoint: str = "/api/analyze") -> Dict[str, Any]:
        """Encaminha requisição para servidor selecionado"""
        start_time = time.time()
        selected_server = None
        
        try:
            # Selecionar servidor
            selected_server = self.select_server(request_data)
            if not selected_server:
                raise Exception("Nenhum servidor disponível")
            
            # Atualizar contadores
            selected_server.current_connections += 1
            selected_server.total_requests += 1
            self.metrics['total_requests'] += 1
            self.metrics['server_distribution'][selected_server.id] += 1
            
            # Configurar sticky session se necessário
            if self.config.sticky_sessions:
                session_id = request_data.get('session_id') or request_data.get('customer_id')
                if session_id:
                    self.session_manager.bind_session(session_id, selected_server.id)
            
            # Fazer requisição
            url = f"{selected_server.url}{endpoint}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(url, json=request_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Registrar sucesso
                        response_time = (time.time() - start_time) * 1000
                        selected_server.response_times.append(response_time)
                        self.circuit_breakers[selected_server.id].record_success()
                        self.metrics['successful_requests'] += 1
                        
                        return result
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
        
        except Exception as e:
            # Registrar falha
            if selected_server:
                selected_server.error_count += 1
                self.circuit_breakers[selected_server.id].record_failure()
            
            self.metrics['failed_requests'] += 1
            
            logger.error(f"❌ Erro ao encaminhar requisição: {e}")
            
            # Tentar retry se configurado
            if self.config.max_retries > 0:
                return await self._retry_request(request_data, endpoint, 1)
            
            raise
        
        finally:
            # Decrementar contador de conexões
            if selected_server:
                selected_server.current_connections = max(0, selected_server.current_connections - 1)
    
    async def _retry_request(self, request_data: Dict[str, Any], endpoint: str, attempt: int) -> Dict[str, Any]:
        """Tenta novamente a requisição"""
        if attempt > self.config.max_retries:
            raise Exception(f"Máximo de {self.config.max_retries} tentativas excedido")
        
        # Aguardar antes de tentar novamente
        await asyncio.sleep(self.config.retry_delay * attempt)
        
        try:
            return await self.forward_request(request_data, endpoint)
        except Exception as e:
            logger.warning(f"⚠️ Tentativa {attempt} falhou: {e}")
            return await self._retry_request(request_data, endpoint, attempt + 1)
    
    async def start_health_checks(self):
        """Inicia health checks periódicos"""
        self.running = True
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("🏥 Health checks iniciados")
    
    async def stop_health_checks(self):
        """Para health checks"""
        self.running = False
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Health checks parados")
    
    async def _health_check_loop(self):
        """Loop de health checks"""
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Erro no health check: {e}")
                await asyncio.sleep(10)
    
    async def _perform_health_checks(self):
        """Executa health checks em todos os servidores"""
        tasks = []
        for server in self.servers.values():
            task = asyncio.create_task(self._check_server_health(server))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Limpar sessões expiradas
        self.session_manager.cleanup_expired_sessions()
    
    async def _check_server_health(self, server: ServerInstance):
        """Verifica saúde de um servidor específico"""
        try:
            url = f"{server.url}/api/health"
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.health_check_timeout)
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        
                        # Atualizar status baseado na resposta
                        if health_data.get('status') == 'healthy':
                            server.status = ServerStatus.HEALTHY
                        elif health_data.get('status') == 'degraded':
                            server.status = ServerStatus.DEGRADED
                        else:
                            server.status = ServerStatus.UNHEALTHY
                    else:
                        server.status = ServerStatus.UNHEALTHY
                        
                    server.last_health_check = datetime.now().isoformat()
        
        except Exception as e:
            logger.warning(f"⚠️ Health check falhou para {server.id}: {e}")
            server.status = ServerStatus.UNHEALTHY
            server.last_health_check = datetime.now().isoformat()
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status do load balancer"""
        healthy_count = sum(1 for s in self.servers.values() if s.status == ServerStatus.HEALTHY)
        total_count = len(self.servers)
        
        return {
            'strategy': self.config.strategy.value,
            'servers': {
                'total': total_count,
                'healthy': healthy_count,
                'degraded': sum(1 for s in self.servers.values() if s.status == ServerStatus.DEGRADED),
                'unhealthy': sum(1 for s in self.servers.values() if s.status == ServerStatus.UNHEALTHY)
            },
            'metrics': self.metrics,
            'server_details': {
                server.id: {
                    'status': server.status.value,
                    'connections': server.current_connections,
                    'avg_response_time': server.avg_response_time,
                    'error_rate': server.error_rate,
                    'total_requests': server.total_requests
                }
                for server in self.servers.values()
            }
        }

# Função para criar load balancer configurado
def create_fraud_load_balancer(strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE) -> FraudDetectionLoadBalancer:
    """Cria load balancer para detecção de fraudes"""
    config = LoadBalancerConfig(
        strategy=strategy,
        health_check_interval=30,
        health_check_timeout=5,
        max_retries=2,
        retry_delay=0.5,
        circuit_breaker_threshold=3,
        circuit_breaker_timeout=60,
        sticky_sessions=True,
        session_timeout=1800  # 30 minutos
    )
    
    return FraudDetectionLoadBalancer(config)

# Exemplo de uso
async def main():
    """Exemplo de uso do load balancer"""
    lb = create_fraud_load_balancer()
    
    # Adicionar servidores
    servers = [
        ServerInstance("server1", "localhost", 8001, weight=1.0, region="us-east"),
        ServerInstance("server2", "localhost", 8002, weight=1.5, region="us-east"),
        ServerInstance("server3", "localhost", 8003, weight=1.0, region="us-west"),
    ]
    
    for server in servers:
        lb.add_server(server)
    
    # Iniciar health checks
    await lb.start_health_checks()
    
    try:
        # Simular requisições
        for i in range(10):
            transaction = {
                'id': f'TXN_{i:06d}',
                'customer_id': f'CUST_{i % 3}',  # 3 clientes diferentes
                'amount': random.uniform(10, 1000),
                'timestamp': datetime.now().isoformat()
            }
            
            try:
                result = await lb.forward_request(transaction)
                print(f"Transação {transaction['id']}: {result.get('decision', 'unknown')}")
            except Exception as e:
                print(f"Erro na transação {transaction['id']}: {e}")
        
        # Status do load balancer
        status = lb.get_status()
        print(f"\nStatus do Load Balancer:")
        print(f"Estratégia: {status['strategy']}")
        print(f"Servidores saudáveis: {status['servers']['healthy']}/{status['servers']['total']}")
        print(f"Requisições processadas: {status['metrics']['total_requests']}")
        
    finally:
        await lb.stop_health_checks()

if __name__ == '__main__':
    asyncio.run(main())

