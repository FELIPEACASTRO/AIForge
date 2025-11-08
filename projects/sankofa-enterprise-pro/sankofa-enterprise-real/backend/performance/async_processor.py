#!/usr/bin/env python3
"""
Sistema de Processamento Assíncrono
Sankofa Enterprise Pro - Async Processor
Otimizado para alta vazão de transações
"""

import logging
import asyncio
import aiofiles
import aiokafka
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Coroutine
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from collections import defaultdict, deque
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)

class ProcessingPriority(Enum):
    """Prioridades de processamento"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class ProcessingStatus(Enum):
    """Status de processamento"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class AsyncTask:
    """Tarefa assíncrona"""
    id: str
    task_type: str
    data: Dict[str, Any]
    priority: ProcessingPriority
    created_at: str
    scheduled_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    callback_url: Optional[str] = None

@dataclass
class ProcessingMetrics:
    """Métricas de processamento"""
    tasks_processed: int = 0
    tasks_pending: int = 0
    tasks_failed: int = 0
    avg_processing_time: float = 0.0
    throughput_per_second: float = 0.0
    queue_depths: Dict[str, int] = None
    worker_utilization: float = 0.0
    
    def __post_init__(self):
        if self.queue_depths is None:
            self.queue_depths = {}

class AsyncTaskProcessor:
    """Processador de tarefas assíncronas"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Filas de prioridade
        self.queues = {
            ProcessingPriority.CRITICAL: asyncio.Queue(maxsize=1000),
            ProcessingPriority.HIGH: asyncio.Queue(maxsize=5000),
            ProcessingPriority.NORMAL: asyncio.Queue(maxsize=20000),
            ProcessingPriority.LOW: asyncio.Queue(maxsize=50000)
        }
        
        # Workers
        self.workers = []
        self.num_workers = config.get('num_workers', mp.cpu_count() * 2)
        self.running = False
        
        # Task handlers
        self.task_handlers = {}
        
        # Métricas
        self.metrics = ProcessingMetrics()
        self.processing_times = deque(maxlen=1000)
        
        # Armazenamento de tarefas
        self.storage_path = config.get('storage_path', '/tmp/async_tasks')
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Kafka para distribuição (opcional)
        self.kafka_enabled = config.get('kafka_enabled', False)
        self.kafka_producer = None
        self.kafka_consumer = None
        
        # Thread pool para tarefas CPU-intensivas
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.get('thread_pool_size', mp.cpu_count())
        )
        
        logger.info(f"🚀 Async Task Processor inicializado com {self.num_workers} workers")
    
    async def initialize(self):
        """Inicializa o processador"""
        # Inicializar Kafka se habilitado
        if self.kafka_enabled:
            await self._initialize_kafka()
        
        # Carregar tarefas pendentes
        await self._load_pending_tasks()
        
        # Iniciar workers
        self.running = True
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Iniciar worker de métricas
        metrics_worker = asyncio.create_task(self._metrics_worker())
        self.workers.append(metrics_worker)
        
        logger.info(f"✅ {self.num_workers} workers assíncronos iniciados")
    
    async def shutdown(self):
        """Para o processador"""
        self.running = False
        
        # Aguardar workers terminarem
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Fechar Kafka
        if self.kafka_producer:
            await self.kafka_producer.stop()
        if self.kafka_consumer:
            await self.kafka_consumer.stop()
        
        # Fechar thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("🛑 Async Task Processor parado")
    
    async def _initialize_kafka(self):
        """Inicializa conexões Kafka"""
        try:
            kafka_config = self.config.get('kafka', {})
            bootstrap_servers = kafka_config.get('bootstrap_servers', 'localhost:9092')
            
            # Producer
            self.kafka_producer = aiokafka.AIOKafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            await self.kafka_producer.start()
            
            # Consumer
            self.kafka_consumer = aiokafka.AIOKafkaConsumer(
                'fraud_tasks',
                bootstrap_servers=bootstrap_servers,
                group_id='fraud_processors',
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            await self.kafka_consumer.start()
            
            logger.info("✅ Kafka inicializado")
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao inicializar Kafka: {e}")
            self.kafka_enabled = False
    
    def register_handler(self, task_type: str, handler: Callable[[Dict[str, Any]], Coroutine]):
        """Registra handler para tipo de tarefa"""
        self.task_handlers[task_type] = handler
        logger.info(f"📝 Handler registrado para: {task_type}")
    
    async def submit_task(self, task_type: str, data: Dict[str, Any], 
                         priority: ProcessingPriority = ProcessingPriority.NORMAL,
                         callback_url: Optional[str] = None,
                         scheduled_at: Optional[str] = None) -> str:
        """Submete tarefa para processamento"""
        task_id = str(uuid.uuid4())
        
        task = AsyncTask(
            id=task_id,
            task_type=task_type,
            data=data,
            priority=priority,
            created_at=datetime.now().isoformat(),
            scheduled_at=scheduled_at,
            callback_url=callback_url
        )
        
        # Salvar tarefa
        await self._save_task(task)
        
        # Adicionar à fila apropriada
        if scheduled_at:
            # Tarefa agendada - será processada pelo scheduler
            await self._schedule_task(task)
        else:
            # Tarefa imediata
            await self.queues[priority].put(task)
            self.metrics.tasks_pending += 1
        
        # Enviar para Kafka se habilitado
        if self.kafka_enabled and self.kafka_producer:
            try:
                await self.kafka_producer.send('fraud_tasks', asdict(task))
            except Exception as e:
                logger.warning(f"⚠️ Erro ao enviar para Kafka: {e}")
        
        logger.debug(f"📤 Tarefa submetida: {task_id} ({task_type})")
        return task_id
    
    async def _worker(self, worker_id: str):
        """Worker para processar tarefas"""
        while self.running:
            try:
                task = await self._get_next_task()
                if task:
                    await self._process_task(task, worker_id)
                else:
                    # Sem tarefas, aguardar um pouco
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"❌ Erro no worker {worker_id}: {e}")
                await asyncio.sleep(1)
    
    async def _get_next_task(self) -> Optional[AsyncTask]:
        """Obtém próxima tarefa das filas (por prioridade)"""
        # Verificar filas por ordem de prioridade
        for priority in [ProcessingPriority.CRITICAL, ProcessingPriority.HIGH, 
                        ProcessingPriority.NORMAL, ProcessingPriority.LOW]:
            try:
                task = await asyncio.wait_for(self.queues[priority].get(), timeout=0.1)
                return task
            except asyncio.TimeoutError:
                continue
        
        return None
    
    async def _process_task(self, task: AsyncTask, worker_id: str):
        """Processa uma tarefa específica"""
        start_time = time.time()
        
        try:
            # Atualizar status
            task.status = ProcessingStatus.PROCESSING
            task.started_at = datetime.now().isoformat()
            await self._save_task(task)
            
            # Verificar se handler existe
            if task.task_type not in self.task_handlers:
                raise ValueError(f"Handler não encontrado para: {task.task_type}")
            
            # Executar handler
            handler = self.task_handlers[task.task_type]
            
            if asyncio.iscoroutinefunction(handler):
                # Handler assíncrono
                result = await handler(task.data)
            else:
                # Handler síncrono - executar em thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.thread_pool, handler, task.data)
            
            # Sucesso
            task.status = ProcessingStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            task.result = result
            
            # Atualizar métricas
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.metrics.tasks_processed += 1
            self.metrics.tasks_pending = max(0, self.metrics.tasks_pending - 1)
            
            logger.debug(f"✅ Tarefa processada: {task.id} em {processing_time:.3f}s por {worker_id}")
            
        except Exception as e:
            # Falha
            task.error = str(e)
            
            if task.retry_count < task.max_retries:
                # Tentar novamente
                task.retry_count += 1
                task.status = ProcessingStatus.RETRYING
                
                # Reagendar com delay exponencial
                delay = min(300, 2 ** task.retry_count)  # Max 5 minutos
                scheduled_time = datetime.now() + timedelta(seconds=delay)
                task.scheduled_at = scheduled_time.isoformat()
                
                await self._schedule_task(task)
                logger.warning(f"🔄 Reagendando tarefa {task.id} (tentativa {task.retry_count})")
                
            else:
                # Falha definitiva
                task.status = ProcessingStatus.FAILED
                task.completed_at = datetime.now().isoformat()
                self.metrics.tasks_failed += 1
                self.metrics.tasks_pending = max(0, self.metrics.tasks_pending - 1)
                
                logger.error(f"❌ Tarefa falhou definitivamente: {task.id} - {e}")
        
        finally:
            # Salvar estado final
            await self._save_task(task)
            
            # Callback se configurado
            if task.callback_url and task.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
                asyncio.create_task(self._send_callback(task))
    
    async def _schedule_task(self, task: AsyncTask):
        """Agenda tarefa para execução futura"""
        # Implementação simples - em produção usaria scheduler mais robusto
        if task.scheduled_at:
            scheduled_time = datetime.fromisoformat(task.scheduled_at)
            delay = (scheduled_time - datetime.now()).total_seconds()
            
            if delay > 0:
                asyncio.create_task(self._delayed_task_submission(task, delay))
            else:
                # Já passou do horário, processar imediatamente
                await self.queues[task.priority].put(task)
                self.metrics.tasks_pending += 1
    
    async def _delayed_task_submission(self, task: AsyncTask, delay: float):
        """Submete tarefa após delay"""
        await asyncio.sleep(delay)
        await self.queues[task.priority].put(task)
        self.metrics.tasks_pending += 1
    
    async def _save_task(self, task: AsyncTask):
        """Salva tarefa no armazenamento"""
        try:
            task_file = os.path.join(self.storage_path, f"{task.id}.json")
            async with aiofiles.open(task_file, 'w') as f:
                await f.write(json.dumps(asdict(task), indent=2, default=str))
        except Exception as e:
            logger.error(f"❌ Erro ao salvar tarefa {task.id}: {e}")
    
    async def _load_pending_tasks(self):
        """Carrega tarefas pendentes do armazenamento"""
        try:
            if not os.path.exists(self.storage_path):
                return
            
            loaded_count = 0
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    try:
                        task_file = os.path.join(self.storage_path, filename)
                        async with aiofiles.open(task_file, 'r') as f:
                            task_data = json.loads(await f.read())
                        
                        task = AsyncTask(**task_data)
                        
                        # Reprocessar tarefas pendentes ou em processamento
                        if task.status in [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]:
                            task.status = ProcessingStatus.PENDING
                            await self.queues[task.priority].put(task)
                            self.metrics.tasks_pending += 1
                            loaded_count += 1
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Erro ao carregar tarefa {filename}: {e}")
            
            if loaded_count > 0:
                logger.info(f"📂 {loaded_count} tarefas pendentes carregadas")
                
        except Exception as e:
            logger.error(f"❌ Erro ao carregar tarefas pendentes: {e}")
    
    async def _send_callback(self, task: AsyncTask):
        """Envia callback para URL configurada"""
        if not task.callback_url:
            return
        
        try:
            import aiohttp
            
            callback_data = {
                'task_id': task.id,
                'status': task.status.value,
                'result': task.result,
                'error': task.error,
                'completed_at': task.completed_at
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(task.callback_url, json=callback_data, timeout=30) as response:
                    if response.status == 200:
                        logger.debug(f"📞 Callback enviado para {task.id}")
                    else:
                        logger.warning(f"⚠️ Callback falhou para {task.id}: HTTP {response.status}")
                        
        except Exception as e:
            logger.error(f"❌ Erro ao enviar callback para {task.id}: {e}")
    
    async def _metrics_worker(self):
        """Worker para calcular métricas"""
        while self.running:
            try:
                await asyncio.sleep(10)  # Atualizar a cada 10 segundos
                
                # Calcular throughput
                if self.processing_times:
                    self.metrics.avg_processing_time = sum(self.processing_times) / len(self.processing_times)
                    self.metrics.throughput_per_second = len(self.processing_times) / 10.0  # Últimos 10 segundos
                
                # Calcular profundidade das filas
                self.metrics.queue_depths = {
                    priority.name: queue.qsize() 
                    for priority, queue in self.queues.items()
                }
                
                # Calcular utilização dos workers
                active_workers = sum(1 for w in self.workers if not w.done())
                self.metrics.worker_utilization = active_workers / len(self.workers) if self.workers else 0
                
                # Log periódico
                if self.metrics.tasks_processed % 1000 == 0 and self.metrics.tasks_processed > 0:
                    logger.info(
                        f"📊 Métricas: {self.metrics.throughput_per_second:.1f} tasks/s, "
                        f"Avg time: {self.metrics.avg_processing_time:.3f}s, "
                        f"Pending: {self.metrics.tasks_pending}, "
                        f"Worker util: {self.metrics.worker_utilization:.1%}"
                    )
                
            except Exception as e:
                logger.error(f"❌ Erro no worker de métricas: {e}")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtém status de uma tarefa"""
        try:
            task_file = os.path.join(self.storage_path, f"{task_id}.json")
            if os.path.exists(task_file):
                async with aiofiles.open(task_file, 'r') as f:
                    task_data = json.loads(await f.read())
                return task_data
            return None
        except Exception as e:
            logger.error(f"❌ Erro ao obter status da tarefa {task_id}: {e}")
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas atuais"""
        return asdict(self.metrics)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancela uma tarefa"""
        try:
            task_data = await self.get_task_status(task_id)
            if task_data and task_data['status'] == ProcessingStatus.PENDING.value:
                # Marcar como cancelada
                task_data['status'] = ProcessingStatus.FAILED.value
                task_data['error'] = 'Task cancelled by user'
                task_data['completed_at'] = datetime.now().isoformat()
                
                # Salvar
                task_file = os.path.join(self.storage_path, f"{task_id}.json")
                async with aiofiles.open(task_file, 'w') as f:
                    await f.write(json.dumps(task_data, indent=2, default=str))
                
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Erro ao cancelar tarefa {task_id}: {e}")
            return False

# Handlers específicos para detecção de fraude
class FraudDetectionHandlers:
    """Handlers específicos para detecção de fraude"""
    
    @staticmethod
    async def analyze_transaction(data: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa transação para fraude"""
        # Simular análise de fraude
        await asyncio.sleep(0.1)  # Simular processamento
        
        amount = data.get('amount', 0)
        fraud_score = min(amount / 10000, 0.9)  # Score baseado no valor
        
        return {
            'transaction_id': data.get('id'),
            'fraud_score': fraud_score,
            'decision': 'block' if fraud_score > 0.8 else 'approve',
            'confidence': 0.95,
            'processing_time_ms': 100
        }
    
    @staticmethod
    async def batch_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
        """Análise em lote de transações"""
        transactions = data.get('transactions', [])
        results = []
        
        for txn in transactions:
            result = await FraudDetectionHandlers.analyze_transaction(txn)
            results.append(result)
        
        return {
            'batch_id': data.get('batch_id'),
            'results': results,
            'processed_count': len(results)
        }
    
    @staticmethod
    def model_training(data: Dict[str, Any]) -> Dict[str, Any]:
        """Treinamento de modelo (CPU-intensivo)"""
        # Simular treinamento
        import time
        time.sleep(5)  # Simular processamento pesado
        
        return {
            'model_id': data.get('model_id'),
            'training_completed': True,
            'accuracy': 0.95,
            'training_time': 5.0
        }

# Função para criar processador configurado
def create_fraud_async_processor(config: Optional[Dict[str, Any]] = None) -> AsyncTaskProcessor:
    """Cria processador assíncrono para detecção de fraudes"""
    default_config = {
        'num_workers': mp.cpu_count() * 2,
        'thread_pool_size': mp.cpu_count(),
        'storage_path': '/home/ubuntu/sankofa-enterprise-real/data/async_tasks',
        'kafka_enabled': False,
        'kafka': {
            'bootstrap_servers': 'localhost:9092'
        }
    }
    
    if config:
        default_config.update(config)
    
    processor = AsyncTaskProcessor(default_config)
    
    # Registrar handlers
    processor.register_handler('analyze_transaction', FraudDetectionHandlers.analyze_transaction)
    processor.register_handler('batch_analysis', FraudDetectionHandlers.batch_analysis)
    processor.register_handler('model_training', FraudDetectionHandlers.model_training)
    
    return processor

# Exemplo de uso
async def main():
    """Exemplo de uso do processador assíncrono"""
    processor = create_fraud_async_processor()
    
    try:
        await processor.initialize()
        
        # Submeter tarefas
        tasks = []
        for i in range(100):
            transaction_data = {
                'id': f'TXN_{i:06d}',
                'amount': i * 100,
                'customer_id': f'CUST_{i % 10}'
            }
            
            task_id = await processor.submit_task(
                'analyze_transaction',
                transaction_data,
                priority=ProcessingPriority.HIGH if i % 10 == 0 else ProcessingPriority.NORMAL
            )
            tasks.append(task_id)
        
        # Aguardar processamento
        await asyncio.sleep(5)
        
        # Verificar status
        completed = 0
        for task_id in tasks[:10]:  # Verificar primeiras 10
            status = await processor.get_task_status(task_id)
            if status and status['status'] == 'completed':
                completed += 1
                print(f"Tarefa {task_id}: {status['result']['decision']}")
        
        print(f"\nTarefas completadas: {completed}/10")
        
        # Métricas
        metrics = processor.get_metrics()
        print(f"Throughput: {metrics['throughput_per_second']:.1f} tasks/s")
        print(f"Tempo médio: {metrics['avg_processing_time']:.3f}s")
        
    finally:
        await processor.shutdown()

if __name__ == '__main__':
    asyncio.run(main())

