#!/usr/bin/env python3
"""
Sistema de Segurança Enterprise para Sankofa Enterprise Pro
Implementa OAuth2/JWT, HTTPS, criptografia AES-256, RBAC e auditoria completa
"""

import os
import jwt
import bcrypt
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import hashlib
import logging
from functools import wraps
from flask import request, jsonify, current_app
import sqlite3
import json

# Configuração de logging seguro
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler('/var/log/sankofa/security.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnterpriseSecuritySystem:
    """Sistema de segurança enterprise completo para ambiente bancário"""
    
    def __init__(self):
        self.jwt_secret = os.environ.get('SANKOFA_JWT_SECRET')
        if not self.jwt_secret:
            logger.warning('SANKOFA_JWT_SECRET não definido, gerando um segredo temporário. Não use em produção.')
            self.jwt_secret = secrets.token_urlsafe(64)
        else:
            logger.info('Chave JWT carregada da variável de ambiente SANKOFA_JWT_SECRET.')
        self.encryption_key = self._generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.db_path = '/home/ubuntu/sankofa-enterprise-real/backend/security/security.db'
        self._init_security_database()
        self._init_default_roles()
        
        # Configurações de segurança
        self.jwt_expiration_hours = 8
        self.refresh_token_days = 30
        self.max_login_attempts = 3
        self.lockout_duration_minutes = 15
        
        logger.info("Sistema de Segurança Enterprise inicializado")
    
    def _generate_jwt_secret(self) -> str:
        """Gera chave secreta segura para JWT"""
        return secrets.token_urlsafe(64)
    
    def _generate_encryption_key(self) -> bytes:
        """Gera chave de criptografia AES-256"""
        password = secrets.token_bytes(32)
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def _init_security_database(self):
        """Inicializa banco de dados de segurança"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabela de usuários
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role_id INTEGER NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP NULL,
                    last_login TIMESTAMP NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (role_id) REFERENCES roles (id)
                )
            ''')
            
            # Tabela de roles (RBAC)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS roles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    permissions TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tabela de sessões ativas
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS active_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    refresh_token TEXT UNIQUE NOT NULL,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Tabela de auditoria
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    resource TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    success BOOLEAN NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            conn.commit()
    
    def _init_default_roles(self):
        """Inicializa roles padrão do sistema"""
        default_roles = [
            {
                'name': 'admin',
                'description': 'Administrador do sistema',
                'permissions': json.dumps([
                    'read_all', 'write_all', 'delete_all', 'manage_users',
                    'view_audit', 'system_config', 'fraud_analysis', 'share_fraud_data'
                ])
            },
            {
                'name': 'analyst',
                'description': 'Analista de fraude',
                'permissions': json.dumps([
                    'read_transactions', 'fraud_analysis', 'view_reports',
                    'mark_fraud', 'view_dashboard'
                ])
            },
            {
                'name': 'operator',
                'description': 'Operador do sistema',
                'permissions': json.dumps([
                    'read_transactions', 'view_dashboard', 'basic_reports'
                ])
            },
            {
                'name': 'auditor',
                'description': 'Auditor de compliance',
                'permissions': json.dumps([
                    'view_audit', 'read_all', 'compliance_reports'
                ])
            }
        ]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for role in default_roles:
                cursor.execute('''
                    INSERT OR IGNORE INTO roles (name, description, permissions)
                    VALUES (?, ?, ?)
                ''', (role['name'], role['description'], role['permissions']))
            
            conn.commit()
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Criptografa dados sensíveis com AES-256"""
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Erro na criptografia: {e}")
            raise
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Descriptografa dados sensíveis"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Erro na descriptografia: {e}")
            raise
    
    def hash_password(self, password: str) -> tuple:
        """Gera hash seguro da senha com salt"""
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8'), salt.decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verifica senha contra hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def create_user(self, username: str, email: str, password: str, role_name: str) -> Dict[str, Any]:
        """Cria novo usuário no sistema"""
        try:
            password_hash, salt = self.hash_password(password)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Busca role_id
                cursor.execute('SELECT id FROM roles WHERE name = ?', (role_name,))
                role_result = cursor.fetchone()
                if not role_result:
                    raise ValueError(f"Role '{role_name}' não encontrada")
                
                role_id = role_result[0]
                
                # Cria usuário
                cursor.execute('''
                    INSERT INTO users (username, email, password_hash, salt, role_id)
                    VALUES (?, ?, ?, ?, ?)
                ''', (username, email, password_hash, salt, role_id))
                
                user_id = cursor.lastrowid
                conn.commit()
                
                self._log_audit(user_id, 'user_created', 'users', 
                              f'Usuário {username} criado com role {role_name}')
                
                logger.info(f"Usuário {username} criado com sucesso")
                return {'user_id': user_id, 'username': username, 'role': role_name}
                
        except Exception as e:
            logger.error(f"Erro ao criar usuário: {e}")
            raise
    
    def authenticate_user(self, username: str, password: str, ip_address: str, user_agent: str) -> Dict[str, Any]:
        """Autentica usuário e gera tokens JWT"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Busca usuário
                cursor.execute('''
                    SELECT u.id, u.username, u.email, u.password_hash, u.is_active,
                           u.failed_login_attempts, u.locked_until, r.name as role_name,
                           r.permissions
                    FROM users u
                    JOIN roles r ON u.role_id = r.id
                    WHERE u.username = ? OR u.email = ?
                ''', (username, username))
                
                user = cursor.fetchone()
                if not user:
                    self._log_audit(None, 'login_failed', 'authentication', 
                                  f'Usuário não encontrado: {username}', ip_address, user_agent, False)
                    raise ValueError("Credenciais inválidas")
                
                user_id, username, email, password_hash, is_active, failed_attempts, locked_until, role_name, permissions = user
                
                # Verifica se conta está ativa
                if not is_active:
                    self._log_audit(user_id, 'login_failed', 'authentication', 
                                  'Conta inativa', ip_address, user_agent, False)
                    raise ValueError("Conta inativa")
                
                # Verifica se conta está bloqueada
                if locked_until and datetime.fromisoformat(locked_until) > datetime.now():
                    self._log_audit(user_id, 'login_failed', 'authentication', 
                                  'Conta bloqueada', ip_address, user_agent, False)
                    raise ValueError("Conta temporariamente bloqueada")
                
                # Verifica senha
                if not self.verify_password(password, password_hash):
                    # Incrementa tentativas falhadas
                    failed_attempts += 1
                    locked_until = None
                    
                    if failed_attempts >= self.max_login_attempts:
                        locked_until = (datetime.now() + timedelta(minutes=self.lockout_duration_minutes)).isoformat()
                    
                    cursor.execute('''
                        UPDATE users 
                        SET failed_login_attempts = ?, locked_until = ?
                        WHERE id = ?
                    ''', (failed_attempts, locked_until, user_id))
                    conn.commit()
                    
                    self._log_audit(user_id, 'login_failed', 'authentication', 
                                  f'Senha incorreta. Tentativas: {failed_attempts}', ip_address, user_agent, False)
                    raise ValueError("Credenciais inválidas")
                
                # Login bem-sucedido - reset tentativas
                cursor.execute('''
                    UPDATE users 
                    SET failed_login_attempts = 0, locked_until = NULL, last_login = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (user_id,))
                
                # Gera tokens
                access_token = self._generate_access_token(user_id, username, role_name, permissions)
                refresh_token = self._generate_refresh_token()
                
                # Salva sessão
                expires_at = datetime.now() + timedelta(hours=self.jwt_expiration_hours)
                cursor.execute('''
                    INSERT INTO active_sessions (user_id, session_token, refresh_token, ip_address, user_agent, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (user_id, access_token, refresh_token, ip_address, user_agent, expires_at))
                
                conn.commit()
                
                self._log_audit(user_id, 'login_success', 'authentication', 
                              'Login bem-sucedido', ip_address, user_agent, True)
                
                logger.info(f"Login bem-sucedido para usuário {username}")
                
                return {
                    'access_token': access_token,
                    'refresh_token': refresh_token,
                    'token_type': 'Bearer',
                    'expires_in': self.jwt_expiration_hours * 3600,
                    'user': {
                        'id': user_id,
                        'username': username,
                        'email': email,
                        'role': role_name,
                        'permissions': json.loads(permissions)
                    }
                }
                
        except Exception as e:
            logger.error(f"Erro na autenticação: {e}")
            raise
    
    def _generate_access_token(self, user_id: int, username: str, role: str, permissions: str) -> str:
        """Gera token JWT de acesso"""
        payload = {
            'user_id': user_id,
            'username': username,
            'role': role,
            'permissions': json.loads(permissions),
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=self.jwt_expiration_hours),
            'type': 'access'
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def _generate_refresh_token(self) -> str:
        """Gera token de refresh"""
        return secrets.token_urlsafe(64)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verifica e decodifica token JWT"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Verifica se é token de acesso
            if payload.get('type') != 'access':
                raise ValueError("Token inválido")
            
            # Verifica se sessão ainda está ativa
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id FROM active_sessions 
                    WHERE session_token = ? AND expires_at > CURRENT_TIMESTAMP
                ''', (token,))
                
                if not cursor.fetchone():
                    raise ValueError("Sessão expirada ou inválida")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise ValueError("Token expirado")
        except jwt.InvalidTokenError:
            raise ValueError("Token inválido")

    def check_permission(self, user_id: int, required_permission: str) -> bool:
        """Verifica se o usuário tem a permissão necessária"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT r.permissions
                    FROM users u
                    JOIN roles r ON u.role_id = r.id
                    WHERE u.id = ?
                ''', (user_id,))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                permissions = json.loads(result[0])
                return required_permission in permissions
        except Exception as e:
            logger.error(f"Erro ao verificar permissão: {e}")
            return False

    def _log_audit(self, user_id: Optional[int], action: str, resource: str, details: str, 
                   ip_address: str = None, user_agent: str = None, success: bool = True):
        """Registra evento na trilha de auditoria"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO audit_log (user_id, action, resource, details, ip_address, user_agent, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, action, resource, details, ip_address, user_agent, success))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao registrar auditoria: {e}")

    def require_auth(self):
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    return jsonify({'error': 'Token de autenticação ausente ou mal formatado'}), 401
                
                token = auth_header.split(' ')[1]
                
                try:
                    payload = self.verify_token(token)
                    # Adiciona o payload do usuário ao contexto da requisição (g)
                    from flask import g
                    g.user = payload
                except ValueError as e:
                    return jsonify({'error': str(e)}), 401
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator

    def require_permission(self, required_permission: str):
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                from flask import g
                user = g.get('user')
                if not user or not self.check_permission(user['user_id'], required_permission):
                    return jsonify({'error': 'Permissão insuficiente'}), 403
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator

