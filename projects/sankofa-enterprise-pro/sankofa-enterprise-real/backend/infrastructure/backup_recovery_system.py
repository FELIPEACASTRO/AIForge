#!/usr/bin/env python3
"""
Sistema de Backup e Recovery
Sankofa Enterprise Pro - Backup & Recovery System
"""

import os
import json
import time
import shutil
import logging
import threading
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import tarfile
import gzip
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class BackupJob:
    """Job de backup"""
    job_id: str
    backup_type: str  # 'full', 'incremental', 'differential'
    source_paths: List[str]
    destination_path: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    size_bytes: int = 0
    checksum: str = ""
    error_message: str = ""

@dataclass
class RecoveryPoint:
    """Ponto de recuperação"""
    recovery_id: str
    backup_job_id: str
    timestamp: str
    backup_type: str
    size_bytes: int
    checksum: str
    retention_until: str
    metadata: Dict[str, Any]

class BackupRecoverySystem:
    """Sistema de Backup e Recovery Enterprise"""
    
    def __init__(self, backup_root: str = "/backup", retention_days: int = 30):
        self.backup_root = backup_root
        self.retention_days = retention_days
        
        # Configurações de backup
        self.backup_config = {
            'models': {
                'source': 'backend/models',
                'schedule': 'daily',
                'retention_days': 90,
                'compression': True
            },
            'data': {
                'source': 'backend/data',
                'schedule': 'hourly',
                'retention_days': 7,
                'compression': True
            },
            'configs': {
                'source': 'backend/configs',
                'schedule': 'daily',
                'retention_days': 30,
                'compression': False
            },
            'logs': {
                'source': 'logs',
                'schedule': 'daily',
                'retention_days': 90,
                'compression': True
            },
            'database': {
                'source': 'database',
                'schedule': 'every_6h',
                'retention_days': 30,
                'compression': True
            }
        }
        
        self.backup_jobs: List[BackupJob] = []
        self.recovery_points: List[RecoveryPoint] = []
        self.is_running = False
        self.backup_thread = None
        
        # Criar diretórios
        os.makedirs(self.backup_root, exist_ok=True)
        os.makedirs(os.path.join(self.backup_root, 'full'), exist_ok=True)
        os.makedirs(os.path.join(self.backup_root, 'incremental'), exist_ok=True)
        os.makedirs(os.path.join(self.backup_root, 'metadata'), exist_ok=True)
        
        logger.info("💾 Sistema de Backup e Recovery inicializado")
        logger.info(f"📁 Diretório de backup: {self.backup_root}")
        logger.info(f"🗓️ Retenção padrão: {self.retention_days} dias")
    
    def create_full_backup(self, backup_name: str = None) -> str:
        """Cria backup completo do sistema"""
        if not backup_name:
            backup_name = f"full_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        job_id = f"full_{int(time.time())}"
        
        # Definir caminhos de origem
        source_paths = []
        for component, config in self.backup_config.items():
            source_path = config['source']
            if os.path.exists(source_path):
                source_paths.append(source_path)
        
        destination_path = os.path.join(self.backup_root, 'full', f"{backup_name}.tar.gz")
        
        backup_job = BackupJob(
            job_id=job_id,
            backup_type='full',
            source_paths=source_paths,
            destination_path=destination_path,
            status='pending'
        )
        
        self.backup_jobs.append(backup_job)
        
        # Executar backup em thread separada
        backup_thread = threading.Thread(
            target=self._execute_backup_job,
            args=(backup_job,),
            daemon=True
        )
        backup_thread.start()
        
        logger.info(f"🚀 Backup completo iniciado: {job_id}")
        return job_id
    
    def create_incremental_backup(self, reference_backup_id: str = None) -> str:
        """Cria backup incremental"""
        job_id = f"inc_{int(time.time())}"
        
        # Encontrar último backup completo se não especificado
        if not reference_backup_id:
            full_backups = [job for job in self.backup_jobs 
                          if job.backup_type == 'full' and job.status == 'completed']
            if not full_backups:
                logger.error("❌ Nenhum backup completo encontrado para referência")
                return None
            
            reference_backup_id = full_backups[-1].job_id
        
        backup_name = f"incremental_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        destination_path = os.path.join(self.backup_root, 'incremental', f"{backup_name}.tar.gz")
        
        # Encontrar arquivos modificados desde o último backup
        reference_job = next((job for job in self.backup_jobs if job.job_id == reference_backup_id), None)
        if not reference_job:
            logger.error(f"❌ Backup de referência {reference_backup_id} não encontrado")
            return None
        
        reference_time = datetime.fromisoformat(reference_job.started_at)
        modified_files = self._find_modified_files_since(reference_time)
        
        backup_job = BackupJob(
            job_id=job_id,
            backup_type='incremental',
            source_paths=modified_files,
            destination_path=destination_path,
            status='pending'
        )
        
        self.backup_jobs.append(backup_job)
        
        # Executar backup
        backup_thread = threading.Thread(
            target=self._execute_backup_job,
            args=(backup_job,),
            daemon=True
        )
        backup_thread.start()
        
        logger.info(f"📈 Backup incremental iniciado: {job_id}")
        return job_id
    
    def _execute_backup_job(self, backup_job: BackupJob):
        """Executa um job de backup"""
        backup_job.status = 'running'
        backup_job.started_at = datetime.now().isoformat()
        
        try:
            logger.info(f"⚙️ Executando backup {backup_job.job_id}")
            
            # Criar arquivo tar comprimido
            with tarfile.open(backup_job.destination_path, 'w:gz') as tar:
                for source_path in backup_job.source_paths:
                    if os.path.exists(source_path):
                        tar.add(source_path, arcname=os.path.basename(source_path))
                        logger.debug(f"📁 Adicionado ao backup: {source_path}")
            
            # Calcular tamanho e checksum
            backup_job.size_bytes = os.path.getsize(backup_job.destination_path)
            backup_job.checksum = self._calculate_file_checksum(backup_job.destination_path)
            
            backup_job.status = 'completed'
            backup_job.completed_at = datetime.now().isoformat()
            
            # Criar ponto de recuperação
            self._create_recovery_point(backup_job)
            
            # Salvar metadata
            self._save_backup_metadata(backup_job)
            
            logger.info(f"✅ Backup {backup_job.job_id} concluído")
            logger.info(f"📊 Tamanho: {backup_job.size_bytes / 1024 / 1024:.1f} MB")
            logger.info(f"🔐 Checksum: {backup_job.checksum[:16]}...")
            
        except Exception as e:
            backup_job.status = 'failed'
            backup_job.error_message = str(e)
            backup_job.completed_at = datetime.now().isoformat()
            
            logger.error(f"❌ Falha no backup {backup_job.job_id}: {e}")
    
    def _find_modified_files_since(self, reference_time: datetime) -> List[str]:
        """Encontra arquivos modificados desde uma data de referência"""
        modified_files = []
        reference_timestamp = reference_time.timestamp()
        
        for component, config in self.backup_config.items():
            source_path = config['source']
            if os.path.exists(source_path):
                if os.path.isfile(source_path):
                    if os.path.getmtime(source_path) > reference_timestamp:
                        modified_files.append(source_path)
                else:
                    for root, dirs, files in os.walk(source_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if os.path.getmtime(file_path) > reference_timestamp:
                                modified_files.append(file_path)
        
        logger.info(f"📁 {len(modified_files)} arquivos modificados encontrados")
        return modified_files
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calcula checksum SHA-256 de um arquivo"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _create_recovery_point(self, backup_job: BackupJob):
        """Cria um ponto de recuperação"""
        recovery_id = f"recovery_{int(time.time())}"
        retention_until = (datetime.now() + timedelta(days=self.retention_days)).isoformat()
        
        recovery_point = RecoveryPoint(
            recovery_id=recovery_id,
            backup_job_id=backup_job.job_id,
            timestamp=backup_job.completed_at,
            backup_type=backup_job.backup_type,
            size_bytes=backup_job.size_bytes,
            checksum=backup_job.checksum,
            retention_until=retention_until,
            metadata={
                'source_paths': backup_job.source_paths,
                'destination_path': backup_job.destination_path,
                'system_info': {
                    'hostname': os.uname().nodename,
                    'platform': os.uname().sysname,
                    'python_version': os.sys.version.split()[0]
                }
            }
        )
        
        self.recovery_points.append(recovery_point)
        logger.info(f"🎯 Ponto de recuperação criado: {recovery_id}")
    
    def _save_backup_metadata(self, backup_job: BackupJob):
        """Salva metadata do backup"""
        metadata_file = os.path.join(
            self.backup_root, 'metadata', 
            f"{backup_job.job_id}_metadata.json"
        )
        
        with open(metadata_file, 'w') as f:
            json.dump(asdict(backup_job), f, indent=2)
    
    def restore_from_backup(self, recovery_id: str, restore_path: str = None) -> bool:
        """Restaura sistema a partir de um backup"""
        recovery_point = next((rp for rp in self.recovery_points if rp.recovery_id == recovery_id), None)
        if not recovery_point:
            logger.error(f"❌ Ponto de recuperação {recovery_id} não encontrado")
            return False
        
        backup_job = next((job for job in self.backup_jobs if job.job_id == recovery_point.backup_job_id), None)
        if not backup_job:
            logger.error(f"❌ Job de backup {recovery_point.backup_job_id} não encontrado")
            return False
        
        if not os.path.exists(backup_job.destination_path):
            logger.error(f"❌ Arquivo de backup não encontrado: {backup_job.destination_path}")
            return False
        
        # Verificar integridade
        current_checksum = self._calculate_file_checksum(backup_job.destination_path)
        if current_checksum != backup_job.checksum:
            logger.error("❌ Falha na verificação de integridade do backup")
            return False
        
        try:
            logger.info(f"🔄 Iniciando restauração de {recovery_id}")
            
            # Definir caminho de restauração
            if not restore_path:
                restore_path = f"restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            os.makedirs(restore_path, exist_ok=True)
            
            # Extrair backup
            with tarfile.open(backup_job.destination_path, 'r:gz') as tar:
                tar.extractall(path=restore_path)
            
            logger.info(f"✅ Restauração concluída em: {restore_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Falha na restauração: {e}")
            return False
    
    def cleanup_old_backups(self):
        """Remove backups antigos baseado na política de retenção"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        expired_recovery_points = [
            rp for rp in self.recovery_points
            if datetime.fromisoformat(rp.retention_until) < datetime.now()
        ]
        
        for recovery_point in expired_recovery_points:
            backup_job = next((job for job in self.backup_jobs 
                             if job.job_id == recovery_point.backup_job_id), None)
            
            if backup_job and os.path.exists(backup_job.destination_path):
                try:
                    os.remove(backup_job.destination_path)
                    logger.info(f"🗑️ Backup expirado removido: {backup_job.job_id}")
                except Exception as e:
                    logger.error(f"❌ Erro ao remover backup {backup_job.job_id}: {e}")
            
            # Remover da lista
            self.recovery_points.remove(recovery_point)
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Retorna status dos backups"""
        total_jobs = len(self.backup_jobs)
        completed_jobs = len([job for job in self.backup_jobs if job.status == 'completed'])
        failed_jobs = len([job for job in self.backup_jobs if job.status == 'failed'])
        running_jobs = len([job for job in self.backup_jobs if job.status == 'running'])
        
        total_size = sum(job.size_bytes for job in self.backup_jobs if job.status == 'completed')
        
        recent_backups = sorted(
            [job for job in self.backup_jobs if job.status == 'completed'],
            key=lambda x: x.completed_at,
            reverse=True
        )[:5]
        
        return {
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'running_jobs': running_jobs,
            'success_rate': (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0,
            'total_size_mb': total_size / 1024 / 1024,
            'recovery_points': len(self.recovery_points),
            'recent_backups': [
                {
                    'job_id': job.job_id,
                    'backup_type': job.backup_type,
                    'completed_at': job.completed_at,
                    'size_mb': job.size_bytes / 1024 / 1024
                }
                for job in recent_backups
            ],
            'last_updated': datetime.now().isoformat()
        }
    
    def start_scheduled_backups(self):
        """Inicia backups agendados"""
        if self.is_running:
            logger.warning("⚠️ Backups agendados já estão rodando")
            return
        
        self.is_running = True
        self.backup_thread = threading.Thread(target=self._scheduled_backup_loop, daemon=True)
        self.backup_thread.start()
        
        logger.info("⏰ Backups agendados iniciados")
    
    def stop_scheduled_backups(self):
        """Para backups agendados"""
        self.is_running = False
        if self.backup_thread:
            self.backup_thread.join(timeout=10)
        
        logger.info("🛑 Backups agendados parados")
    
    def _scheduled_backup_loop(self):
        """Loop de backups agendados"""
        last_full_backup = None
        last_incremental_backup = None
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Backup completo diário às 02:00
                if (current_time.hour == 2 and current_time.minute == 0 and
                    (not last_full_backup or 
                     (current_time - last_full_backup).days >= 1)):
                    
                    self.create_full_backup()
                    last_full_backup = current_time
                    logger.info("📅 Backup completo agendado executado")
                
                # Backup incremental a cada 6 horas
                if (current_time.hour % 6 == 0 and current_time.minute == 0 and
                    (not last_incremental_backup or 
                     (current_time - last_incremental_backup).seconds >= 6 * 3600)):
                    
                    self.create_incremental_backup()
                    last_incremental_backup = current_time
                    logger.info("📈 Backup incremental agendado executado")
                
                # Limpeza semanal
                if current_time.weekday() == 6 and current_time.hour == 3:  # Domingo às 03:00
                    self.cleanup_old_backups()
                    logger.info("🧹 Limpeza de backups antigos executada")
                
                time.sleep(60)  # Verificar a cada minuto
                
            except Exception as e:
                logger.error(f"❌ Erro no loop de backup agendado: {e}")
                time.sleep(300)  # Aguardar 5 minutos em caso de erro
    
    def test_backup_recovery(self) -> bool:
        """Testa o sistema de backup e recovery"""
        logger.info("🧪 Iniciando teste de backup e recovery")
        
        try:
            # Criar backup de teste
            test_job_id = self.create_full_backup("test_backup")
            
            # Aguardar conclusão
            timeout = 300  # 5 minutos
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                test_job = next((job for job in self.backup_jobs if job.job_id == test_job_id), None)
                if test_job and test_job.status == 'completed':
                    break
                elif test_job and test_job.status == 'failed':
                    logger.error("❌ Teste de backup falhou")
                    return False
                time.sleep(5)
            else:
                logger.error("❌ Timeout no teste de backup")
                return False
            
            # Encontrar ponto de recuperação
            recovery_point = next((rp for rp in self.recovery_points 
                                 if rp.backup_job_id == test_job_id), None)
            if not recovery_point:
                logger.error("❌ Ponto de recuperação não criado")
                return False
            
            # Testar restauração
            test_restore_path = f"/tmp/test_restore_{int(time.time())}"
            if not self.restore_from_backup(recovery_point.recovery_id, test_restore_path):
                logger.error("❌ Teste de restauração falhou")
                return False
            
            # Limpar arquivos de teste
            if os.path.exists(test_restore_path):
                shutil.rmtree(test_restore_path)
            
            logger.info("✅ Teste de backup e recovery concluído com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro no teste de backup e recovery: {e}")
            return False

# Instância global do sistema de backup
backup_system = BackupRecoverySystem()

if __name__ == "__main__":
    # Teste do sistema de backup
    system = BackupRecoverySystem()
    
    print("💾 Testando Sistema de Backup e Recovery")
    print("=" * 50)
    
    # Criar alguns arquivos de teste
    os.makedirs("test_data", exist_ok=True)
    with open("test_data/test_file.txt", "w") as f:
        f.write("Dados de teste para backup\n")
    
    # Testar backup completo
    job_id = system.create_full_backup("test_backup")
    print(f"🚀 Backup iniciado: {job_id}")
    
    # Aguardar conclusão
    time.sleep(5)
    
    # Verificar status
    status = system.get_backup_status()
    print(f"📊 Status: {status['completed_jobs']}/{status['total_jobs']} jobs concluídos")
    
    # Testar sistema completo
    if system.test_backup_recovery():
        print("✅ Sistema de Backup e Recovery testado com sucesso!")
    else:
        print("❌ Falha no teste do sistema")
    
    # Limpar arquivos de teste
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")
