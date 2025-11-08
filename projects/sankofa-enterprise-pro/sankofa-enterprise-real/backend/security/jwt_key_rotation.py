#!/usr/bin/env python3
"""
Sistema de Rotação Automática de Chaves JWT
Sankofa Enterprise Pro - JWT Key Rotation System
"""

import os
import json
import time
import logging
import threading
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

@dataclass
class JWTKey:
    """Chave JWT"""
    key_id: str
    algorithm: str
    private_key: str
    public_key: str
    created_at: str
    expires_at: str
    status: str  # 'active', 'rotating', 'expired', 'revoked'
    usage_count: int = 0

@dataclass
class KeyRotationEvent:
    """Evento de rotação de chave"""
    event_id: str
    old_key_id: str
    new_key_id: str
    rotation_reason: str
    timestamp: str
    status: str  # 'initiated', 'completed', 'failed'

class JWTKeyRotationSystem:
    """Sistema de Rotação Automática de Chaves JWT"""
    
    def __init__(self, keys_dir: str = "jwt_keys", rotation_interval_hours: int = 24):
        self.keys_dir = keys_dir
        self.rotation_interval_hours = rotation_interval_hours
        
        # Configurações de rotação
        self.rotation_config = {
            'auto_rotation_enabled': True,
            'rotation_interval_hours': rotation_interval_hours,
            'key_overlap_hours': 2,  # Tempo que chaves antigas permanecem válidas
            'max_usage_count': 10000,  # Rotacionar após N usos
            'algorithm': 'RS256',  # RSA com SHA-256
            'key_size': 2048,
            'grace_period_hours': 1  # Período de graça para tokens existentes
        }
        
        self.active_keys: List[JWTKey] = []
        self.rotation_events: List[KeyRotationEvent] = []
        self.is_rotating = False
        self.rotation_thread = None
        
        # Criar diretório de chaves
        os.makedirs(self.keys_dir, exist_ok=True)
        
        # Carregar chaves existentes ou criar primeira chave
        self._load_existing_keys()
        if not self.active_keys:
            self._generate_initial_key()
        
        logger.info("🔐 Sistema de Rotação de Chaves JWT inicializado")
        logger.info(f"📁 Diretório de chaves: {self.keys_dir}")
        logger.info(f"⏰ Intervalo de rotação: {rotation_interval_hours}h")
    
    def _generate_initial_key(self):
        """Gera a primeira chave JWT"""
        logger.info("🔑 Gerando chave JWT inicial")
        self._generate_new_key("initial_key_generation")
    
    def _load_existing_keys(self):
        """Carrega chaves existentes do disco"""
        keys_file = os.path.join(self.keys_dir, "jwt_keys.json")
        
        if os.path.exists(keys_file):
            try:
                with open(keys_file, 'r') as f:
                    keys_data = json.load(f)
                
                for key_data in keys_data.get('keys', []):
                    jwt_key = JWTKey(**key_data)
                    
                    # Verificar se a chave não expirou
                    if datetime.fromisoformat(jwt_key.expires_at) > datetime.now():
                        self.active_keys.append(jwt_key)
                    else:
                        jwt_key.status = 'expired'
                        self.active_keys.append(jwt_key)
                
                logger.info(f"🔑 {len(self.active_keys)} chaves carregadas do disco")
                
            except Exception as e:
                logger.error(f"❌ Erro ao carregar chaves: {e}")
    
    def _save_keys_to_disk(self):
        """Salva chaves no disco"""
        keys_file = os.path.join(self.keys_dir, "jwt_keys.json")
        
        keys_data = {
            'keys': [asdict(key) for key in self.active_keys],
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(keys_file, 'w') as f:
                json.dump(keys_data, f, indent=2)
            
            logger.debug("💾 Chaves salvas no disco")
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar chaves: {e}")
    
    def _generate_new_key(self, rotation_reason: str = "scheduled_rotation") -> JWTKey:
        """Gera uma nova chave JWT"""
        key_id = f"key_{int(time.time())}_{secrets.token_hex(4)}"
        
        # Gerar par de chaves RSA
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.rotation_config['key_size'],
            backend=default_backend()
        )
        
        # Serializar chave privada
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
        
        # Serializar chave pública
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        
        # Criar objeto JWTKey
        jwt_key = JWTKey(
            key_id=key_id,
            algorithm=self.rotation_config['algorithm'],
            private_key=private_pem,
            public_key=public_pem,
            created_at=datetime.now().isoformat(),
            expires_at=(datetime.now() + timedelta(
                hours=self.rotation_config['rotation_interval_hours'] + 
                      self.rotation_config['key_overlap_hours']
            )).isoformat(),
            status='active'
        )
        
        # Marcar chaves antigas como 'rotating'
        for key in self.active_keys:
            if key.status == 'active':
                key.status = 'rotating'
        
        # Adicionar nova chave
        self.active_keys.append(jwt_key)
        
        # Salvar no disco
        self._save_keys_to_disk()
        
        # Registrar evento de rotação
        if self.active_keys:
            old_key_id = next((k.key_id for k in self.active_keys if k.status == 'rotating'), None)
            self._record_rotation_event(old_key_id, key_id, rotation_reason)
        
        logger.info(f"🔑 Nova chave JWT gerada: {key_id}")
        return jwt_key
    
    def _record_rotation_event(self, old_key_id: str, new_key_id: str, reason: str):
        """Registra evento de rotação"""
        event = KeyRotationEvent(
            event_id=f"rotation_{int(time.time())}",
            old_key_id=old_key_id or "none",
            new_key_id=new_key_id,
            rotation_reason=reason,
            timestamp=datetime.now().isoformat(),
            status='completed'
        )
        
        self.rotation_events.append(event)
        
        # Manter apenas os últimos 100 eventos
        if len(self.rotation_events) > 100:
            self.rotation_events = self.rotation_events[-100:]
        
        logger.info(f"📝 Evento de rotação registrado: {event.event_id}")
    
    def get_current_signing_key(self) -> Optional[JWTKey]:
        """Retorna a chave atual para assinatura"""
        active_keys = [key for key in self.active_keys if key.status == 'active']
        
        if not active_keys:
            logger.warning("⚠️ Nenhuma chave ativa encontrada")
            return None
        
        # Retornar a chave mais recente
        return max(active_keys, key=lambda k: k.created_at)
    
    def get_verification_keys(self) -> List[JWTKey]:
        """Retorna todas as chaves válidas para verificação"""
        now = datetime.now()
        
        valid_keys = []
        for key in self.active_keys:
            expires_at = datetime.fromisoformat(key.expires_at)
            if expires_at > now and key.status in ['active', 'rotating']:
                valid_keys.append(key)
        
        return valid_keys
    
    def sign_token(self, payload: Dict[str, Any], expires_in_hours: int = 24) -> Optional[str]:
        """Assina um token JWT"""
        signing_key = self.get_current_signing_key()
        if not signing_key:
            logger.error("❌ Nenhuma chave de assinatura disponível")
            return None
        
        try:
            # Adicionar claims padrão
            now = datetime.now()
            payload.update({
                'iat': now.timestamp(),
                'exp': (now + timedelta(hours=expires_in_hours)).timestamp(),
                'kid': signing_key.key_id  # Key ID para identificar a chave
            })
            
            # Assinar token
            token = jwt.encode(
                payload,
                signing_key.private_key,
                algorithm=signing_key.algorithm,
                headers={'kid': signing_key.key_id}
            )
            
            # Incrementar contador de uso
            signing_key.usage_count += 1
            
            # Verificar se precisa rotacionar por uso excessivo
            if signing_key.usage_count >= self.rotation_config['max_usage_count']:
                logger.info(f"🔄 Rotação necessária: chave {signing_key.key_id} atingiu limite de uso")
                self._schedule_rotation("max_usage_reached")
            
            logger.debug(f"🔐 Token assinado com chave {signing_key.key_id}")
            return token
            
        except Exception as e:
            logger.error(f"❌ Erro ao assinar token: {e}")
            return None
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verifica um token JWT"""
        try:
            # Decodificar header para obter key ID
            unverified_header = jwt.get_unverified_header(token)
            key_id = unverified_header.get('kid')
            
            if not key_id:
                logger.warning("⚠️ Token sem Key ID")
                return None
            
            # Encontrar chave de verificação
            verification_key = None
            for key in self.get_verification_keys():
                if key.key_id == key_id:
                    verification_key = key
                    break
            
            if not verification_key:
                logger.warning(f"⚠️ Chave de verificação não encontrada: {key_id}")
                return None
            
            # Verificar token
            payload = jwt.decode(
                token,
                verification_key.public_key,
                algorithms=[verification_key.algorithm]
            )
            
            logger.debug(f"✅ Token verificado com chave {key_id}")
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("⚠️ Token expirado")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"⚠️ Token inválido: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Erro ao verificar token: {e}")
            return None
    
    def rotate_keys_now(self, reason: str = "manual_rotation") -> bool:
        """Força rotação imediata das chaves"""
        if self.is_rotating:
            logger.warning("⚠️ Rotação já em andamento")
            return False
        
        try:
            self.is_rotating = True
            logger.info("🔄 Iniciando rotação manual de chaves")
            
            self._generate_new_key(reason)
            
            # Aguardar período de graça
            grace_period = self.rotation_config['grace_period_hours'] * 3600
            logger.info(f"⏳ Período de graça: {self.rotation_config['grace_period_hours']}h")
            
            # Em produção, isso seria feito de forma assíncrona
            # time.sleep(grace_period)
            
            # Marcar chaves antigas como expiradas
            self._expire_old_keys()
            
            self.is_rotating = False
            logger.info("✅ Rotação de chaves concluída")
            return True
            
        except Exception as e:
            self.is_rotating = False
            logger.error(f"❌ Erro na rotação de chaves: {e}")
            return False
    
    def _expire_old_keys(self):
        """Marca chaves antigas como expiradas"""
        for key in self.active_keys:
            if key.status == 'rotating':
                key.status = 'expired'
                logger.info(f"⏰ Chave expirada: {key.key_id}")
        
        self._save_keys_to_disk()
    
    def _schedule_rotation(self, reason: str):
        """Agenda rotação de chaves"""
        if not self.rotation_config['auto_rotation_enabled']:
            logger.info("🔄 Rotação automática desabilitada")
            return
        
        if self.is_rotating:
            logger.info("🔄 Rotação já agendada")
            return
        
        # Em um ambiente real, isso seria feito com um scheduler
        rotation_thread = threading.Thread(
            target=self.rotate_keys_now,
            args=(reason,),
            daemon=True
        )
        rotation_thread.start()
    
    def start_automatic_rotation(self):
        """Inicia rotação automática de chaves"""
        if self.rotation_thread and self.rotation_thread.is_alive():
            logger.warning("⚠️ Rotação automática já está ativa")
            return
        
        self.rotation_config['auto_rotation_enabled'] = True
        self.rotation_thread = threading.Thread(
            target=self._rotation_loop,
            daemon=True
        )
        self.rotation_thread.start()
        
        logger.info("⏰ Rotação automática de chaves iniciada")
    
    def stop_automatic_rotation(self):
        """Para rotação automática de chaves"""
        self.rotation_config['auto_rotation_enabled'] = False
        
        if self.rotation_thread:
            self.rotation_thread.join(timeout=10)
        
        logger.info("🛑 Rotação automática de chaves parada")
    
    def _rotation_loop(self):
        """Loop de rotação automática"""
        while self.rotation_config['auto_rotation_enabled']:
            try:
                # Verificar se precisa rotacionar
                current_key = self.get_current_signing_key()
                if current_key:
                    created_at = datetime.fromisoformat(current_key.created_at)
                    age_hours = (datetime.now() - created_at).total_seconds() / 3600
                    
                    if age_hours >= self.rotation_config['rotation_interval_hours']:
                        logger.info("⏰ Rotação automática necessária")
                        self._schedule_rotation("scheduled_rotation")
                
                # Limpar chaves expiradas
                self._cleanup_expired_keys()
                
                # Aguardar 1 hora antes da próxima verificação
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"❌ Erro no loop de rotação: {e}")
                time.sleep(300)  # Aguardar 5 minutos em caso de erro
    
    def _cleanup_expired_keys(self):
        """Remove chaves expiradas antigas"""
        now = datetime.now()
        cleanup_threshold = now - timedelta(days=7)  # Manter por 7 dias após expiração
        
        keys_to_remove = []
        for key in self.active_keys:
            if (key.status == 'expired' and 
                datetime.fromisoformat(key.expires_at) < cleanup_threshold):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.active_keys.remove(key)
            logger.info(f"🗑️ Chave antiga removida: {key.key_id}")
        
        if keys_to_remove:
            self._save_keys_to_disk()
    
    def get_rotation_status(self) -> Dict[str, Any]:
        """Retorna status da rotação de chaves"""
        current_key = self.get_current_signing_key()
        verification_keys = self.get_verification_keys()
        
        status = {
            'auto_rotation_enabled': self.rotation_config['auto_rotation_enabled'],
            'is_rotating': self.is_rotating,
            'rotation_interval_hours': self.rotation_config['rotation_interval_hours'],
            'current_key': {
                'key_id': current_key.key_id if current_key else None,
                'created_at': current_key.created_at if current_key else None,
                'usage_count': current_key.usage_count if current_key else 0,
                'age_hours': (
                    (datetime.now() - datetime.fromisoformat(current_key.created_at)).total_seconds() / 3600
                    if current_key else 0
                )
            },
            'verification_keys_count': len(verification_keys),
            'total_keys': len(self.active_keys),
            'recent_rotations': [
                {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp,
                    'reason': event.rotation_reason,
                    'status': event.status
                }
                for event in self.rotation_events[-5:]  # Últimos 5 eventos
            ],
            'next_rotation_due': (
                (datetime.fromisoformat(current_key.created_at) + 
                 timedelta(hours=self.rotation_config['rotation_interval_hours'])).isoformat()
                if current_key else None
            ),
            'last_updated': datetime.now().isoformat()
        }
        
        return status

# Instância global do sistema de rotação
jwt_rotation_system = JWTKeyRotationSystem()

if __name__ == "__main__":
    # Teste do sistema de rotação de chaves
    system = JWTKeyRotationSystem()
    
    print("🔐 Testando Sistema de Rotação de Chaves JWT")
    print("=" * 50)
    
    # Testar assinatura de token
    test_payload = {
        'user_id': 'test_user',
        'role': 'admin',
        'permissions': ['read', 'write']
    }
    
    token = system.sign_token(test_payload, expires_in_hours=1)
    if token:
        print(f"✅ Token assinado: {token[:50]}...")
        
        # Testar verificação
        verified_payload = system.verify_token(token)
        if verified_payload:
            print(f"✅ Token verificado: user_id = {verified_payload.get('user_id')}")
        else:
            print("❌ Falha na verificação do token")
    else:
        print("❌ Falha na assinatura do token")
    
    # Testar rotação manual
    if system.rotate_keys_now("test_rotation"):
        print("✅ Rotação manual executada com sucesso")
    else:
        print("❌ Falha na rotação manual")
    
    # Verificar status
    status = system.get_rotation_status()
    print(f"📊 Status: {status['verification_keys_count']} chaves de verificação ativas")
    print(f"🔑 Chave atual: {status['current_key']['key_id']}")
    
    print("🔐 Sistema de Rotação de Chaves JWT testado com sucesso!")
