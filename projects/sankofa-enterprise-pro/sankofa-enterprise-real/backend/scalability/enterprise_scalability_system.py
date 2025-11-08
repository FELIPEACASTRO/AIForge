"""
Sistema de Escalabilidade Enterprise para Sankofa Enterprise Pro
Implementa load balancing, circuit breakers, rate limiting avançado e auto-scaling
"""

import asyncio
import time
import random
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class ScalabilityConfig:
    # Load Balancer
    max_workers: int = 50
    health_check_interval: int = 30
    unhealthy_threshold: int = 3
    recovery_threshold: int = 2
    
    # Circuit Breaker
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    
    # Rate Limiting
    rate_limit_window: int = 60
    max_requests_per_window: int = 1000
    burst_limit: int = 100
    
    # Auto Scaling
    cpu_scale_up_threshold: float = 70.0
    cpu_scale_down_threshold: float = 30.0
    memory_scale_up_threshold: float = 80.0
    memory_scale_down_threshold: float = 40.0
    min_instances: int = 2
    max_instances: int = 20
    scale_cooldown: int = 300

@dataclass
class ServiceInstance:
    id: str
    host: str
    port: int
    weight: int = 1
    is_healthy: bool = True
    last_health_check: float = field(default_factory=time.time)
    failure_count: int = 0
    success_count: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))

class LoadBalancer:
    """Load Balancer com health checks e múltiplas estratégias"""
    
    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self.instances: List[ServiceInstance] = []
        self.current_index = 0
        self.lock = threading.Lock()
        self.health_check_thread = None
        self.running = False
        
    def add_instance(self, instance: ServiceInstance):
        """Adiciona instância ao pool"""
        with self.lock:
            self.instances.append(instance)
            logger.info(f"Instância adicionada: {instance.id} ({instance.host}:{instance.port})")
    
    def remove_instance(self, instance_id: str):
        """Remove instância do pool"""
        with self.lock:
            self.instances = [i for i in self.instances if i.id != instance_id]
            logger.info(f"Instância removida: {instance_id}")
    
    def get_healthy_instances(self) -> List[ServiceInstance]:
        """Retorna apenas instâncias saudáveis"""
        return [i for i in self.instances if i.is_healthy]
    
    def round_robin_select(self) -> Optional[ServiceInstance]:
        """Seleção round-robin"""
        healthy_instances = self.get_healthy_instances()
        if not healthy_instances:
            return None
            
        with self.lock:
            instance = healthy_instances[self.current_index % len(healthy_instances)]
            self.current_index += 1
            return instance
    
    def weighted_round_robin_select(self) -> Optional[ServiceInstance]:
        """Seleção weighted round-robin"""
        healthy_instances = self.get_healthy_instances()
        if not healthy_instances:
            return None
        
        # Cria lista ponderada
        weighted_instances = []
        for instance in healthy_instances:
            weighted_instances.extend([instance] * instance.weight)
        
        if not weighted_instances:
            return None
            
        with self.lock:
            instance = weighted_instances[self.current_index % len(weighted_instances)]
            self.current_index += 1
            return instance
    
    def least_connections_select(self) -> Optional[ServiceInstance]:
        """Seleção por menor número de conexões ativas"""
        healthy_instances = self.get_healthy_instances()
        if not healthy_instances:
            return None
        
        # Simula contagem de conexões baseada em response times
        return min(healthy_instances, key=lambda i: len(i.response_times))
    
    def least_response_time_select(self) -> Optional[ServiceInstance]:
        """Seleção por menor tempo de resposta"""
        healthy_instances = self.get_healthy_instances()
        if not healthy_instances:
            return None
        
        def avg_response_time(instance):
            if not instance.response_times:
                return 0
            return sum(instance.response_times) / len(instance.response_times)
        
        return min(healthy_instances, key=avg_response_time)
    
    def select_instance(self, strategy: str = "weighted_round_robin") -> Optional[ServiceInstance]:
        """Seleciona instância baseada na estratégia"""
        strategies = {
            "round_robin": self.round_robin_select,
            "weighted_round_robin": self.weighted_round_robin_select,
            "least_connections": self.least_connections_select,
            "least_response_time": self.least_response_time_select
        }
        
        return strategies.get(strategy, self.weighted_round_robin_select)()
    
    def health_check_instance(self, instance: ServiceInstance) -> bool:
        """Executa health check em uma instância"""
        try:
            # Simula health check HTTP
            start_time = time.time()
            
            # Simula latência de rede
            time.sleep(random.uniform(0.001, 0.01))
            
            # Simula falha ocasional
            if random.random() < 0.05:  # 5% chance de falha
                raise Exception("Health check failed")
            
            response_time = time.time() - start_time
            instance.response_times.append(response_time)
            instance.last_health_check = time.time()
            instance.success_count += 1
            
            if instance.failure_count > 0:
                instance.failure_count = max(0, instance.failure_count - 1)
            
            return True
            
        except Exception as e:
            instance.failure_count += 1
            logger.warning(f"Health check falhou para {instance.id}: {e}")
            return False
    
    def update_instance_health(self):
        """Atualiza status de saúde das instâncias"""
        for instance in self.instances:
            is_healthy = self.health_check_instance(instance)
            
            if is_healthy and instance.failure_count < self.config.unhealthy_threshold:
                instance.is_healthy = True
            elif instance.failure_count >= self.config.unhealthy_threshold:
                instance.is_healthy = False
                logger.warning(f"Instância {instance.id} marcada como não saudável")
            
            # Recovery check
            if not instance.is_healthy and instance.success_count >= self.config.recovery_threshold:
                instance.is_healthy = True
                instance.failure_count = 0
                logger.info(f"Instância {instance.id} recuperada")
    
    def start_health_checks(self):
        """Inicia thread de health checks"""
        self.running = True
        
        def health_check_loop():
            while self.running:
                try:
                    self.update_instance_health()
                    time.sleep(self.config.health_check_interval)
                except Exception as e:
                    logger.error(f"Erro no health check: {e}")
        
        self.health_check_thread = threading.Thread(target=health_check_loop, daemon=True)
        self.health_check_thread.start()
        logger.info("Health checks iniciados")
    
    def stop_health_checks(self):
        """Para thread de health checks"""
        self.running = False
        if self.health_check_thread:
            self.health_check_thread.join()
        logger.info("Health checks parados")

class CircuitBreaker:
    """Circuit Breaker para proteção contra falhas em cascata"""
    
    def __init__(self, config: ScalabilityConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        self.lock = threading.Lock()
        
        # Métricas
        self.total_calls = 0
        self.total_failures = 0
        self.state_changes = []
    
    def _should_attempt_reset(self) -> bool:
        """Verifica se deve tentar resetar o circuit breaker"""
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    def _record_success(self):
        """Registra sucesso"""
        with self.lock:
            self.failure_count = 0
            self.success_count += 1
            
            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.recovery_threshold:
                    self._change_state(CircuitState.CLOSED)
                    self.half_open_calls = 0
    
    def _record_failure(self):
        """Registra falha"""
        with self.lock:
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self._change_state(CircuitState.OPEN)
            elif self.state == CircuitState.HALF_OPEN:
                self._change_state(CircuitState.OPEN)
                self.half_open_calls = 0
    
    def _change_state(self, new_state: CircuitState):
        """Muda estado do circuit breaker"""
        old_state = self.state
        self.state = new_state
        self.state_changes.append({
            'timestamp': time.time(),
            'from': old_state.value,
            'to': new_state.value
        })
        logger.info(f"Circuit Breaker {self.name}: {old_state.value} -> {new_state.value}")
    
    def call(self, func: Callable, *args, **kwargs):
        """Executa função com proteção do circuit breaker"""
        with self.lock:
            self.total_calls += 1
            
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._change_state(CircuitState.HALF_OPEN)
                    self.half_open_calls = 0
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise Exception(f"Circuit breaker {self.name} half-open limit exceeded")
                self.half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise e
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do circuit breaker"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'total_calls': self.total_calls,
            'total_failures': self.total_failures,
            'failure_rate': self.total_failures / max(1, self.total_calls),
            'state_changes': len(self.state_changes),
            'last_failure_time': self.last_failure_time
        }

class RateLimiter:
    """Rate Limiter avançado com sliding window e burst protection"""
    
    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self.requests = defaultdict(deque)  # client_id -> timestamps
        self.burst_tokens = defaultdict(int)  # client_id -> tokens
        self.lock = threading.Lock()
        
        # Métricas
        self.total_requests = 0
        self.blocked_requests = 0
        self.clients = set()
    
    def _cleanup_old_requests(self, client_id: str):
        """Remove requisições antigas da janela"""
        current_time = time.time()
        window_start = current_time - self.config.rate_limit_window
        
        client_requests = self.requests[client_id]
        while client_requests and client_requests[0] < window_start:
            client_requests.popleft()
    
    def _refill_burst_tokens(self, client_id: str):
        """Reabastece tokens de burst"""
        current_time = time.time()
        
        # Simula refill baseado em tempo
        if not hasattr(self, '_last_refill'):
            self._last_refill = {}
        
        last_refill = self._last_refill.get(client_id, current_time)
        time_passed = current_time - last_refill
        
        # Adiciona tokens baseado no tempo passado
        tokens_to_add = int(time_passed * (self.config.burst_limit / 60))  # tokens por segundo
        self.burst_tokens[client_id] = min(
            self.config.burst_limit,
            self.burst_tokens[client_id] + tokens_to_add
        )
        
        self._last_refill[client_id] = current_time
    
    def is_allowed(self, client_id: str, tokens_requested: int = 1) -> bool:
        """Verifica se requisição é permitida"""
        with self.lock:
            self.total_requests += 1
            self.clients.add(client_id)
            
            current_time = time.time()
            
            # Cleanup e refill
            self._cleanup_old_requests(client_id)
            self._refill_burst_tokens(client_id)
            
            client_requests = self.requests[client_id]
            
            # Verifica limite da janela deslizante
            if len(client_requests) >= self.config.max_requests_per_window:
                self.blocked_requests += 1
                return False
            
            # Verifica burst limit
            if self.burst_tokens[client_id] < tokens_requested:
                self.blocked_requests += 1
                return False
            
            # Permite requisição
            client_requests.append(current_time)
            self.burst_tokens[client_id] -= tokens_requested
            
            return True
    
    def get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """Retorna estatísticas de um cliente"""
        with self.lock:
            self._cleanup_old_requests(client_id)
            self._refill_burst_tokens(client_id)
            
            return {
                'client_id': client_id,
                'requests_in_window': len(self.requests[client_id]),
                'burst_tokens': self.burst_tokens[client_id],
                'window_utilization': len(self.requests[client_id]) / self.config.max_requests_per_window,
                'burst_utilization': (self.config.burst_limit - self.burst_tokens[client_id]) / self.config.burst_limit
            }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas globais"""
        return {
            'total_requests': self.total_requests,
            'blocked_requests': self.blocked_requests,
            'block_rate': self.blocked_requests / max(1, self.total_requests),
            'active_clients': len(self.clients),
            'avg_requests_per_client': self.total_requests / max(1, len(self.clients))
        }

class AutoScaler:
    """Auto Scaler baseado em métricas de sistema"""
    
    def __init__(self, config: ScalabilityConfig, load_balancer: LoadBalancer):
        self.config = config
        self.load_balancer = load_balancer
        self.current_instances = config.min_instances
        self.last_scale_time = 0
        self.metrics_history = deque(maxlen=10)
        
        # Inicializa instâncias mínimas
        self._initialize_instances()
    
    def _initialize_instances(self):
        """Inicializa instâncias mínimas"""
        for i in range(self.config.min_instances):
            instance = ServiceInstance(
                id=f"auto-instance-{i}",
                host="localhost",
                port=5000 + i,
                weight=1
            )
            self.load_balancer.add_instance(instance)
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Coleta métricas do sistema"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'load_avg': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
            'timestamp': time.time()
        }
    
    def _should_scale_up(self, metrics: Dict[str, float]) -> bool:
        """Verifica se deve escalar para cima"""
        cpu_high = metrics['cpu_percent'] > self.config.cpu_scale_up_threshold
        memory_high = metrics['memory_percent'] > self.config.memory_scale_up_threshold
        
        return (cpu_high or memory_high) and self.current_instances < self.config.max_instances
    
    def _should_scale_down(self, metrics: Dict[str, float]) -> bool:
        """Verifica se deve escalar para baixo"""
        cpu_low = metrics['cpu_percent'] < self.config.cpu_scale_down_threshold
        memory_low = metrics['memory_percent'] < self.config.memory_scale_down_threshold
        
        return (cpu_low and memory_low) and self.current_instances > self.config.min_instances
    
    def _can_scale(self) -> bool:
        """Verifica se está no período de cooldown"""
        return (time.time() - self.last_scale_time) >= self.config.scale_cooldown
    
    def scale_up(self):
        """Adiciona nova instância"""
        if not self._can_scale():
            return False
        
        new_instance = ServiceInstance(
            id=f"auto-instance-{self.current_instances}",
            host="localhost",
            port=5000 + self.current_instances,
            weight=1
        )
        
        self.load_balancer.add_instance(new_instance)
        self.current_instances += 1
        self.last_scale_time = time.time()
        
        logger.info(f"Scaled UP: {self.current_instances} instances")
        return True
    
    def scale_down(self):
        """Remove instância"""
        if not self._can_scale():
            return False
        
        # Remove última instância
        instance_to_remove = f"auto-instance-{self.current_instances - 1}"
        self.load_balancer.remove_instance(instance_to_remove)
        self.current_instances -= 1
        self.last_scale_time = time.time()
        
        logger.info(f"Scaled DOWN: {self.current_instances} instances")
        return True
    
    def evaluate_scaling(self):
        """Avalia necessidade de scaling"""
        metrics = self._get_system_metrics()
        self.metrics_history.append(metrics)
        
        # Usa média das últimas métricas para decisão
        if len(self.metrics_history) >= 3:
            avg_cpu = sum(m['cpu_percent'] for m in list(self.metrics_history)[-3:]) / 3
            avg_memory = sum(m['memory_percent'] for m in list(self.metrics_history)[-3:]) / 3
            
            avg_metrics = {
                'cpu_percent': avg_cpu,
                'memory_percent': avg_memory
            }
            
            if self._should_scale_up(avg_metrics):
                return self.scale_up()
            elif self._should_scale_down(avg_metrics):
                return self.scale_down()
        
        return False
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de scaling"""
        current_metrics = self._get_system_metrics() if self.metrics_history else {}
        
        return {
            'current_instances': self.current_instances,
            'min_instances': self.config.min_instances,
            'max_instances': self.config.max_instances,
            'last_scale_time': self.last_scale_time,
            'cooldown_remaining': max(0, self.config.scale_cooldown - (time.time() - self.last_scale_time)),
            'current_metrics': current_metrics,
            'can_scale': self._can_scale()
        }

class EnterpriseScalabilitySystem:
    """Sistema completo de escalabilidade enterprise"""
    
    def __init__(self, config: ScalabilityConfig = None):
        self.config = config or ScalabilityConfig()
        
        # Componentes
        self.load_balancer = LoadBalancer(self.config)
        self.circuit_breakers = {}
        self.rate_limiter = RateLimiter(self.config)
        self.auto_scaler = AutoScaler(self.config, self.load_balancer)
        
        # Thread pool para processamento
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Métricas globais
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        logger.info("Sistema de Escalabilidade Enterprise inicializado")
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Obtém ou cria circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(self.config, name)
        return self.circuit_breakers[name]
    
    def process_request(self, client_id: str, request_data: Dict[str, Any], 
                       service_name: str = "default") -> Dict[str, Any]:
        """Processa requisição com todas as proteções"""
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Rate limiting
            if not self.rate_limiter.is_allowed(client_id):
                raise Exception("Rate limit exceeded")
            
            # Load balancing
            instance = self.load_balancer.select_instance()
            if not instance:
                raise Exception("No healthy instances available")
            
            # Circuit breaker
            circuit_breaker = self.get_circuit_breaker(service_name)
            
            def process_function():
                # Simula processamento da requisição
                processing_time = random.uniform(0.01, 0.1)
                time.sleep(processing_time)
                
                # Simula falha ocasional
                if random.random() < 0.02:  # 2% chance de falha
                    raise Exception("Service temporarily unavailable")
                
                return {
                    "status": "success",
                    "instance_id": instance.id,
                    "processing_time": processing_time,
                    "data": request_data
                }
            
            result = circuit_breaker.call(process_function)
            
            # Atualiza métricas da instância
            response_time = time.time() - start_time
            instance.response_times.append(response_time)
            instance.success_count += 1
            
            self.successful_requests += 1
            
            return {
                **result,
                "response_time": response_time,
                "client_id": client_id
            }
            
        except Exception as e:
            self.failed_requests += 1
            logger.warning(f"Request failed for client {client_id}: {e}")
            
            return {
                "status": "error",
                "error": str(e),
                "response_time": time.time() - start_time,
                "client_id": client_id
            }
    
    def start_monitoring(self):
        """Inicia monitoramento automático"""
        self.load_balancer.start_health_checks()
        
        def auto_scaling_loop():
            while True:
                try:
                    self.auto_scaler.evaluate_scaling()
                    time.sleep(60)  # Avalia a cada minuto
                except Exception as e:
                    logger.error(f"Erro no auto scaling: {e}")
        
        scaling_thread = threading.Thread(target=auto_scaling_loop, daemon=True)
        scaling_thread.start()
        
        logger.info("Monitoramento automático iniciado")
    
    def stop_monitoring(self):
        """Para monitoramento automático"""
        self.load_balancer.stop_health_checks()
        logger.info("Monitoramento automático parado")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas completas do sistema"""
        uptime = time.time() - self.start_time
        
        return {
            "uptime": uptime,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(1, self.total_requests),
            "requests_per_second": self.total_requests / max(1, uptime),
            "load_balancer": {
                "total_instances": len(self.load_balancer.instances),
                "healthy_instances": len(self.load_balancer.get_healthy_instances()),
                "instances": [
                    {
                        "id": i.id,
                        "host": f"{i.host}:{i.port}",
                        "healthy": i.is_healthy,
                        "weight": i.weight,
                        "success_count": i.success_count,
                        "failure_count": i.failure_count,
                        "avg_response_time": sum(i.response_times) / len(i.response_times) if i.response_times else 0
                    }
                    for i in self.load_balancer.instances
                ]
            },
            "circuit_breakers": {
                name: cb.get_stats() 
                for name, cb in self.circuit_breakers.items()
            },
            "rate_limiter": self.rate_limiter.get_global_stats(),
            "auto_scaler": self.auto_scaler.get_scaling_stats()
        }
    
    def shutdown(self):
        """Encerra sistema graciosamente"""
        self.stop_monitoring()
        self.executor.shutdown(wait=True)
        logger.info("Sistema de Escalabilidade Enterprise encerrado")

# Teste do sistema
if __name__ == "__main__":
    print("🚀 Testando Sistema de Escalabilidade Enterprise...")
    
    # Configuração
    config = ScalabilityConfig(
        max_workers=20,
        max_requests_per_window=100,
        failure_threshold=3,
        min_instances=2,
        max_instances=5
    )
    
    # Inicializa sistema
    scalability_system = EnterpriseScalabilitySystem(config)
    
    # Adiciona algumas instâncias manuais
    for i in range(3):
        instance = ServiceInstance(
            id=f"manual-instance-{i}",
            host="localhost",
            port=8000 + i,
            weight=1 + i  # Pesos diferentes
        )
        scalability_system.load_balancer.add_instance(instance)
    
    # Inicia monitoramento
    scalability_system.start_monitoring()
    
    try:
        # Simula carga de trabalho
        print("📊 Simulando carga de trabalho...")
        
        clients = [f"client_{i}" for i in range(10)]
        requests_per_client = 20
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for client in clients:
                for i in range(requests_per_client):
                    future = executor.submit(
                        scalability_system.process_request,
                        client,
                        {"request_id": f"{client}_req_{i}", "data": f"test_data_{i}"},
                        "fraud_detection"
                    )
                    futures.append(future)
            
            # Coleta resultados
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Erro na requisição: {e}")
        
        # Aguarda um pouco para métricas
        time.sleep(5)
        
        # Estatísticas finais
        stats = scalability_system.get_system_stats()
        
        print("\n✅ Teste concluído!")
        print(f"   Total de requisições: {stats['total_requests']}")
        print(f"   Taxa de sucesso: {stats['success_rate']:.2%}")
        print(f"   RPS médio: {stats['requests_per_second']:.2f}")
        print(f"   Instâncias saudáveis: {stats['load_balancer']['healthy_instances']}/{stats['load_balancer']['total_instances']}")
        print(f"   Circuit breakers: {len(stats['circuit_breakers'])}")
        print(f"   Clientes ativos: {stats['rate_limiter']['active_clients']}")
        print(f"   Instâncias auto-scaled: {stats['auto_scaler']['current_instances']}")
        
        # Testa rate limiting
        print("\n🔒 Testando rate limiting...")
        client_stats = scalability_system.rate_limiter.get_client_stats("client_0")
        print(f"   Cliente 0 - Utilização da janela: {client_stats['window_utilization']:.2%}")
        print(f"   Cliente 0 - Tokens de burst: {client_stats['burst_tokens']}")
        
        # Testa circuit breaker
        if "fraud_detection" in stats['circuit_breakers']:
            cb_stats = stats['circuit_breakers']['fraud_detection']
            print(f"\n⚡ Circuit Breaker 'fraud_detection':")
            print(f"   Estado: {cb_stats['state']}")
            print(f"   Taxa de falha: {cb_stats['failure_rate']:.2%}")
            print(f"   Total de chamadas: {cb_stats['total_calls']}")
        
    finally:
        # Encerra sistema
        scalability_system.shutdown()
    
    print("\n🚀 Teste do Sistema de Escalabilidade Enterprise concluído!")
