#!/usr/bin/env python3
"""
Sistema de Configuração Segura para Sankofa Enterprise Pro
Gerencia variáveis de ambiente, secrets e configurações sensíveis
"""

import os
import json
import base64
import secrets
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)

class SecureConfig:
    """Gerenciador de configurações seguras"""
    
    def __init__(self):
        self.config_file = '/home/ubuntu/sankofa-enterprise-real/backend/security/.env.encrypted'
        self.key_file = '/home/ubuntu/sankofa-enterprise-real/backend/security/.key'
        self._encryption_key = None
        self._config_cache = {}
        
        # Configurações padrão
        self.default_config = {
            # Banco de dados
            'DB_HOST': 'localhost',
            'DB_PORT': '5432',
            'DB_NAME': 'sankofa_enterprise',
            'DB_USER': 'sankofa_user',
            'DB_PASSWORD': self._generate_secure_password(),
            
            # JWT
            'JWT_SECRET_KEY': secrets.token_urlsafe(64),
            'JWT_EXPIRATION_HOURS': '8',
            'REFRESH_TOKEN_DAYS': '30',
            
            # Criptografia
            'ENCRYPTION_KEY': base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
            'SALT': base64.urlsafe_b64encode(secrets.token_bytes(16)).decode(),
            
            # API
            'API_HOST': '0.0.0.0',
            'API_PORT': '8443',
            'API_DEBUG': 'false',
            'CORS_ORIGINS': 'https://localhost:3000,https://sankofa.enterprise.com',
            
            # SSL
            'SSL_CERT_PATH': '/etc/ssl/certs/sankofa.crt',
            'SSL_KEY_PATH': '/etc/ssl/private/sankofa.key',
            
            # Redis
            'REDIS_HOST': 'localhost',
            'REDIS_PORT': '6379',
            'REDIS_PASSWORD': self._generate_secure_password(),
            'REDIS_DB': '0',
            
            # DataDog
            'DATADOG_API_KEY': '',
            'DATADOG_APP_KEY': '',
            'DATADOG_SERVICE_NAME': 'sankofa-enterprise-pro',
            
            # AWS
            'AWS_REGION': 'us-east-1',
            'AWS_ACCESS_KEY_ID': '',
            'AWS_SECRET_ACCESS_KEY': '',
            
            # Segurança
            'RATE_LIMIT_REQUESTS': '100',
            'RATE_LIMIT_WINDOW': '60',
            'MAX_LOGIN_ATTEMPTS': '3',
            'LOCKOUT_DURATION_MINUTES': '15',
            
            # Logs
            'LOG_LEVEL': 'INFO',
            'LOG_FILE': '/var/log/sankofa/app.log',
            'SECURITY_LOG_FILE': '/var/log/sankofa/security.log',
            
            # Performance
            'WORKER_PROCESSES': '4',
            'WORKER_THREADS': '8',
            'MAX_CONNECTIONS': '1000',
            
            # ML Models
            'MODEL_PATH': '/home/ubuntu/sankofa-enterprise-real/models',
            'MODEL_CACHE_SIZE': '1000',
            'AUTO_RETRAIN_ENABLED': 'true',
            'RETRAIN_INTERVAL_HOURS': '24'
        }
        
        self._init_encryption()
        self._load_or_create_config()
    
    def _generate_secure_password(self, length: int = 32) -> str:
        """Gera senha segura"""
        return secrets.token_urlsafe(length)
    
    def _init_encryption(self):
        """Inicializa sistema de criptografia"""
        try:
            # Tenta carregar chave existente
            if os.path.exists(self.key_file):
                with open(self.key_file, 'rb') as f:
                    self._encryption_key = f.read()
            else:
                # Gera nova chave
                self._encryption_key = Fernet.generate_key()
                
                # Salva chave com permissões seguras
                os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
                with open(self.key_file, 'wb') as f:
                    f.write(self._encryption_key)
                
                os.chmod(self.key_file, 0o600)
                logger.info("Nova chave de criptografia gerada")
        
        except Exception as e:
            logger.error(f"Erro ao inicializar criptografia: {e}")
            raise
    
    def _encrypt_data(self, data: str) -> bytes:
        """Criptografa dados"""
        cipher = Fernet(self._encryption_key)
        return cipher.encrypt(data.encode())
    
    def _decrypt_data(self, encrypted_data: bytes) -> str:
        """Descriptografa dados"""
        cipher = Fernet(self._encryption_key)
        return cipher.decrypt(encrypted_data).decode()
    
    def _load_or_create_config(self):
        """Carrega configuração existente ou cria nova"""
        try:
            if os.path.exists(self.config_file):
                # Carrega configuração criptografada
                with open(self.config_file, 'rb') as f:
                    encrypted_config = f.read()
                
                decrypted_config = self._decrypt_data(encrypted_config)
                self._config_cache = json.loads(decrypted_config)
                logger.info("Configuração carregada do arquivo criptografado")
            else:
                # Cria nova configuração
                self._config_cache = self.default_config.copy()
                self._save_config()
                logger.info("Nova configuração criada com valores padrão")
        
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            # Fallback para configuração padrão
            self._config_cache = self.default_config.copy()
    
    def _save_config(self):
        """Salva configuração criptografada"""
        try:
            config_json = json.dumps(self._config_cache, indent=2)
            encrypted_config = self._encrypt_data(config_json)
            
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'wb') as f:
                f.write(encrypted_config)
            
            os.chmod(self.config_file, 0o600)
            logger.info("Configuração salva com criptografia")
        
        except Exception as e:
            logger.error(f"Erro ao salvar configuração: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtém valor de configuração"""
        # Primeiro verifica variáveis de ambiente
        env_value = os.environ.get(key)
        if env_value is not None:
            return env_value
        
        # Depois verifica cache de configuração
        return self._config_cache.get(key, default)
    
    def set(self, key: str, value: Any):
        """Define valor de configuração"""
        self._config_cache[key] = str(value)
        self._save_config()
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Obtém valor booleano"""
        value = self.get(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Obtém valor inteiro"""
        try:
            return int(self.get(key, default))
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Obtém valor float"""
        try:
            return float(self.get(key, default))
        except (ValueError, TypeError):
            return default
    
    def get_list(self, key: str, separator: str = ',', default: list = None) -> list:
        """Obtém lista de valores"""
        if default is None:
            default = []
        
        value = self.get(key)
        if not value:
            return default
        
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    def get_database_url(self) -> str:
        """Constrói URL de conexão com banco de dados"""
        host = self.get('DB_HOST')
        port = self.get('DB_PORT')
        name = self.get('DB_NAME')
        user = self.get('DB_USER')
        password = self.get('DB_PASSWORD')
        
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"
    
    def get_redis_url(self) -> str:
        """Constrói URL de conexão com Redis"""
        host = self.get('REDIS_HOST')
        port = self.get('REDIS_PORT')
        password = self.get('REDIS_PASSWORD')
        db = self.get('REDIS_DB')
        
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        else:
            return f"redis://{host}:{port}/{db}"
    
    def get_cors_origins(self) -> list:
        """Obtém lista de origens CORS permitidas"""
        return self.get_list('CORS_ORIGINS')
    
    def validate_config(self) -> Dict[str, Any]:
        """Valida configuração atual"""
        issues = []
        warnings = []
        
        # Verifica configurações críticas
        critical_keys = [
            'JWT_SECRET_KEY', 'ENCRYPTION_KEY', 'DB_PASSWORD', 'REDIS_PASSWORD'
        ]
        
        for key in critical_keys:
            value = self.get(key)
            if not value or len(value) < 16:
                issues.append(f"{key} não está definido ou é muito curto")
        
        # Verifica configurações de segurança
        if self.get_int('MAX_LOGIN_ATTEMPTS') < 3:
            warnings.append("MAX_LOGIN_ATTEMPTS muito baixo (recomendado: >= 3)")
        
        if self.get_int('RATE_LIMIT_REQUESTS') > 1000:
            warnings.append("RATE_LIMIT_REQUESTS muito alto (pode impactar performance)")
        
        # Verifica SSL
        ssl_cert = self.get('SSL_CERT_PATH')
        ssl_key = self.get('SSL_KEY_PATH')
        
        if ssl_cert and not os.path.exists(ssl_cert):
            issues.append(f"Certificado SSL não encontrado: {ssl_cert}")
        
        if ssl_key and not os.path.exists(ssl_key):
            issues.append(f"Chave SSL não encontrada: {ssl_key}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def export_config(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Exporta configuração (opcionalmente sem secrets)"""
        config = self._config_cache.copy()
        
        if not include_secrets:
            # Remove valores sensíveis
            sensitive_keys = [
                'JWT_SECRET_KEY', 'ENCRYPTION_KEY', 'DB_PASSWORD',
                'REDIS_PASSWORD', 'DATADOG_API_KEY', 'DATADOG_APP_KEY',
                'AWS_SECRET_ACCESS_KEY'
            ]
            
            for key in sensitive_keys:
                if key in config:
                    config[key] = '***HIDDEN***'
        
        return config
    
    def reset_to_defaults(self):
        """Reseta configuração para valores padrão"""
        self._config_cache = self.default_config.copy()
        self._save_config()
        logger.warning("Configuração resetada para valores padrão")
    
    def rotate_secrets(self):
        """Rotaciona secrets de segurança"""
        logger.info("Iniciando rotação de secrets...")
        
        # Gera novos secrets
        new_secrets = {
            'JWT_SECRET_KEY': secrets.token_urlsafe(64),
            'ENCRYPTION_KEY': base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
            'DB_PASSWORD': self._generate_secure_password(),
            'REDIS_PASSWORD': self._generate_secure_password()
        }
        
        # Atualiza configuração
        for key, value in new_secrets.items():
            self.set(key, value)
        
        logger.info("Rotação de secrets concluída")
        return new_secrets

# Instância global de configuração
config = SecureConfig()

# Funções de conveniência
def get_config(key: str, default: Any = None) -> Any:
    """Função de conveniência para obter configuração"""
    return config.get(key, default)

def get_database_url() -> str:
    """Função de conveniência para URL do banco"""
    return config.get_database_url()

def get_redis_url() -> str:
    """Função de conveniência para URL do Redis"""
    return config.get_redis_url()

# Teste da configuração
if __name__ == "__main__":
    print("🔧 Testando Sistema de Configuração Segura...")
    
    # Valida configuração
    validation = config.validate_config()
    print(f"✅ Configuração válida: {validation['valid']}")
    
    if validation['issues']:
        print("❌ Issues encontrados:")
        for issue in validation['issues']:
            print(f"   - {issue}")
    
    if validation['warnings']:
        print("⚠️ Warnings:")
        for warning in validation['warnings']:
            print(f"   - {warning}")
    
    # Testa algumas configurações
    print(f"🔑 JWT Secret: {config.get('JWT_SECRET_KEY')[:20]}...")
    print(f"🗄️ Database URL: {config.get_database_url()}")
    print(f"📦 Redis URL: {config.get_redis_url()}")
    print(f"🌐 CORS Origins: {config.get_cors_origins()}")
    
    print("🔧 Teste do Sistema de Configuração Segura concluído!")
