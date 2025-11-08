#!/usr/bin/env python3
"""
Middleware de Segurança para Sankofa Enterprise Pro
Implementa proteções contra ataques comuns: CSRF, XSS, SQL Injection, Rate Limiting
"""

import time
import hashlib
import secrets
import re
from typing import Dict, List, Optional, Any
from functools import wraps
from flask import request, jsonify, session, g
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
import ipaddress

logger = logging.getLogger(__name__)

class SecurityMiddleware:
    """Middleware de segurança avançado para proteção da API"""
    
    def __init__(self):
        # Rate limiting - armazena tentativas por IP
        self.rate_limit_storage = defaultdict(deque)
        self.blocked_ips = {}
        
        # CSRF tokens
        self.csrf_tokens = {}
        
        # Configurações de segurança
        self.rate_limit_requests = 100  # requests por minuto
        self.rate_limit_window = 60     # janela em segundos
        self.block_duration = 300       # bloqueio por 5 minutos
        
        # Padrões de ataques conhecidos
        self.sql_injection_patterns = [
            r"(\bunion\b.*\bselect\b)",
            r"(\bselect\b.*\bfrom\b)",
            r"(\binsert\b.*\binto\b)",
            r"(\bdelete\b.*\bfrom\b)",
            r"(\bdrop\b.*\btable\b)",
            r"(\bupdate\b.*\bset\b)",
            r"('.*or.*'.*=.*')",
            r"(--|\#|\/\*)",
            r"(\bexec\b|\bexecute\b)",
            r"(\bsp_\w+)",
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
            r"<link[^>]*>",
            r"<meta[^>]*>",
        ]
        
        # IPs confiáveis (whitelist)
        self.trusted_ips = [
            '127.0.0.1',
            '::1',
            '10.0.0.0/8',
            '172.16.0.0/12',
            '192.168.0.0/16'
        ]
        
        logger.info("Middleware de Segurança inicializado")
    
    def is_trusted_ip(self, ip_address: str) -> bool:
        """Verifica se o IP está na lista de confiáveis"""
        try:
            ip = ipaddress.ip_address(ip_address)
            for trusted in self.trusted_ips:
                if '/' in trusted:
                    if ip in ipaddress.ip_network(trusted):
                        return True
                else:
                    if str(ip) == trusted:
                        return True
            return False
        except:
            return False
    
    def rate_limit_check(self, ip_address: str) -> bool:
        """Verifica rate limiting por IP"""
        current_time = time.time()
        
        # Verifica se IP está bloqueado
        if ip_address in self.blocked_ips:
            if current_time < self.blocked_ips[ip_address]:
                return False
            else:
                # Remove bloqueio expirado
                del self.blocked_ips[ip_address]
        
        # IPs confiáveis têm limite mais alto
        if self.is_trusted_ip(ip_address):
            limit = self.rate_limit_requests * 5
        else:
            limit = self.rate_limit_requests
        
        # Limpa requisições antigas
        requests = self.rate_limit_storage[ip_address]
        while requests and requests[0] < current_time - self.rate_limit_window:
            requests.popleft()
        
        # Verifica limite
        if len(requests) >= limit:
            # Bloqueia IP
            self.blocked_ips[ip_address] = current_time + self.block_duration
            logger.warning(f"IP {ip_address} bloqueado por excesso de requisições")
            return False
        
        # Adiciona requisição atual
        requests.append(current_time)
        return True
    
    def detect_sql_injection(self, data: Any) -> bool:
        """Detecta tentativas de SQL Injection"""
        if not data:
            return False
        
        # Converte para string se necessário
        if isinstance(data, dict):
            text = str(data)
        elif isinstance(data, list):
            text = ' '.join(str(item) for item in data)
        else:
            text = str(data)
        
        text = text.lower()
        
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"SQL Injection detectado: {pattern}")
                return True
        
        return False
    
    def detect_xss(self, data: Any) -> bool:
        """Detecta tentativas de XSS"""
        if not data:
            return False
        
        # Converte para string se necessário
        if isinstance(data, dict):
            text = str(data)
        elif isinstance(data, list):
            text = ' '.join(str(item) for item in data)
        else:
            text = str(data)
        
        for pattern in self.xss_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"XSS detectado: {pattern}")
                return True
        
        return False
    
    def sanitize_input(self, data: Any) -> Any:
        """Sanitiza entrada de dados"""
        if isinstance(data, str):
            # Remove caracteres perigosos
            data = re.sub(r'[<>"\']', '', data)
            # Limita tamanho
            if len(data) > 10000:
                data = data[:10000]
        elif isinstance(data, dict):
            return {k: self.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        
        return data
    
    def generate_csrf_token(self, user_id: str) -> str:
        """Gera token CSRF para usuário"""
        token = secrets.token_urlsafe(32)
        self.csrf_tokens[user_id] = {
            'token': token,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(hours=1)
        }
        return token
    
    def validate_csrf_token(self, user_id: str, token: str) -> bool:
        """Valida token CSRF"""
        if user_id not in self.csrf_tokens:
            return False
        
        stored_token = self.csrf_tokens[user_id]
        
        # Verifica expiração
        if datetime.now() > stored_token['expires_at']:
            del self.csrf_tokens[user_id]
            return False
        
        # Verifica token
        return stored_token['token'] == token
    
    def validate_request_headers(self, headers: Dict[str, str]) -> bool:
        """Valida headers da requisição"""
        # Verifica Content-Type para POST/PUT
        if request.method in ['POST', 'PUT', 'PATCH']:
            content_type = headers.get('Content-Type', '')
            if not content_type.startswith('application/json'):
                logger.warning(f"Content-Type inválido: {content_type}")
                return False
        
        # Verifica User-Agent
        user_agent = headers.get('User-Agent', '')
        if not user_agent or len(user_agent) < 10:
            logger.warning("User-Agent suspeito ou ausente")
            return False
        
        # Detecta bots maliciosos
        malicious_agents = [
            'sqlmap', 'nikto', 'nmap', 'masscan', 'zap',
            'burp', 'w3af', 'havij', 'pangolin'
        ]
        
        for agent in malicious_agents:
            if agent.lower() in user_agent.lower():
                logger.warning(f"User-Agent malicioso detectado: {user_agent}")
                return False
        
        return True
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Registra evento de segurança"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'ip_address': request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr),
            'user_agent': request.headers.get('User-Agent'),
            'path': request.path,
            'method': request.method,
            'details': details
        }
        
        logger.warning(f"SECURITY EVENT: {event_type} - {details}")
        
        # Em produção, enviar para SIEM/DataDog
        # self.send_to_siem(log_entry)
    
    def security_check_middleware(self):
        """Middleware principal de verificação de segurança"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
                
                # 1. Rate Limiting
                if not self.rate_limit_check(ip_address):
                    self.log_security_event('rate_limit_exceeded', {
                        'ip': ip_address,
                        'requests_per_minute': len(self.rate_limit_storage[ip_address])
                    })
                    return jsonify({'error': 'Rate limit exceeded'}), 429
                
                # 2. Validação de headers
                if not self.validate_request_headers(request.headers):
                    self.log_security_event('invalid_headers', {
                        'headers': dict(request.headers)
                    })
                    return jsonify({'error': 'Invalid request headers'}), 400
                
                # 3. Verificação de payload
                if request.is_json:
                    try:
                        data = request.get_json()
                        
                        # Detecta SQL Injection
                        if self.detect_sql_injection(data):
                            self.log_security_event('sql_injection_attempt', {
                                'payload': str(data)[:500]
                            })
                            return jsonify({'error': 'Malicious payload detected'}), 400
                        
                        # Detecta XSS
                        if self.detect_xss(data):
                            self.log_security_event('xss_attempt', {
                                'payload': str(data)[:500]
                            })
                            return jsonify({'error': 'Malicious payload detected'}), 400
                        
                        # Sanitiza dados
                        sanitized_data = self.sanitize_input(data)
                        request._cached_json = sanitized_data
                        
                    except Exception as e:
                        logger.error(f"Erro ao processar JSON: {e}")
                        return jsonify({'error': 'Invalid JSON payload'}), 400
                
                # 4. Verificação de tamanho da requisição
                if request.content_length and request.content_length > 10 * 1024 * 1024:  # 10MB
                    self.log_security_event('oversized_request', {
                        'content_length': request.content_length
                    })
                    return jsonify({'error': 'Request too large'}), 413
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def csrf_protection(self):
        """Decorator para proteção CSRF"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
                    # Verifica se usuário está autenticado
                    if not hasattr(g, 'current_user') or not g.current_user:
                        return jsonify({'error': 'Authentication required'}), 401
                    
                    # Verifica token CSRF
                    csrf_token = request.headers.get('X-CSRF-Token')
                    if not csrf_token:
                        return jsonify({'error': 'CSRF token required'}), 400
                    
                    user_id = str(g.current_user.get('user_id'))
                    if not self.validate_csrf_token(user_id, csrf_token):
                        self.log_security_event('csrf_token_invalid', {
                            'user_id': user_id,
                            'token': csrf_token[:10] + '...'
                        })
                        return jsonify({'error': 'Invalid CSRF token'}), 403
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de segurança"""
        current_time = time.time()
        
        # Conta IPs ativos
        active_ips = 0
        total_requests = 0
        
        for ip, requests in self.rate_limit_storage.items():
            # Remove requisições antigas
            while requests and requests[0] < current_time - self.rate_limit_window:
                requests.popleft()
            
            if requests:
                active_ips += 1
                total_requests += len(requests)
        
        return {
            'active_ips': active_ips,
            'blocked_ips': len(self.blocked_ips),
            'total_requests_last_minute': total_requests,
            'csrf_tokens_active': len(self.csrf_tokens),
            'rate_limit_window': self.rate_limit_window,
            'rate_limit_per_ip': self.rate_limit_requests
        }

# Instância global do middleware
security_middleware = SecurityMiddleware()

# Decorators exportados
require_security_check = security_middleware.security_check_middleware
require_csrf_protection = security_middleware.csrf_protection
