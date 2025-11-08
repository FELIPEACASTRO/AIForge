#!/usr/bin/env python3
"""
API Principal Segura do Sankofa Enterprise Pro
Integra sistema de segurança enterprise com autenticação JWT, HTTPS e RBAC
"""

import os
import sys
import ssl
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import traceback

# Adiciona o diretório pai ao path para importações
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security.enterprise_security_system import EnterpriseSecuritySystem
from ml_engine.continuous_learning_system import ContinuousLearningSystem
from performance.high_performance_system import HighPerformanceSystem
from scalability.enterprise_scalability_system import EnterpriseScalabilitySystem

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/sankofa/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SecureMainAPI:
    """API Principal com Segurança Enterprise Integrada"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.wsgi_app = ProxyFix(self.app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
        
        # Configurações de segurança
        self.app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'sankofa-enterprise-pro-2025')
        self.app.config['JSON_SORT_KEYS'] = False
        
        # CORS configurado de forma segura
        CORS(self.app, 
             origins=['https://localhost:3000', 'https://sankofa.enterprise.com'],
             methods=['GET', 'POST', 'PUT', 'DELETE'],
             allow_headers=['Content-Type', 'Authorization'],
             supports_credentials=True)
        
        # Inicializa sistemas
        self.security_system = EnterpriseSecuritySystem()
        self.ml_system = ContinuousLearningSystem()
        self.performance_system = HighPerformanceSystem()
        self.scalability_system = EnterpriseScalabilitySystem()
        
        # Registra rotas
        self._register_routes()
        self._register_error_handlers()
        
        logger.info("API Principal Segura inicializada")
    
    def _register_routes(self):
        """Registra todas as rotas da API"""
        
        # Rotas de autenticação
        @self.app.route('/api/v1/auth/login', methods=['POST'])
        def login():
            """Endpoint de login com autenticação JWT"""
            try:
                data = request.get_json()
                if not data or not data.get('username') or not data.get('password'):
                    return jsonify({'error': 'Username e password são obrigatórios'}), 400
                
                # Captura informações da requisição
                ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
                user_agent = request.headers.get('User-Agent', 'Unknown')
                
                # Autentica usuário
                auth_result = self.security_system.authenticate_user(
                    username=data['username'],
                    password=data['password'],
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                
                return jsonify({
                    'success': True,
                    'message': 'Login realizado com sucesso',
                    'data': auth_result
                }), 200
                
            except Exception as e:
                logger.error(f"Erro no login: {e}")
                return jsonify({'error': str(e)}), 401
        
        @self.app.route('/api/v1/auth/logout', methods=['POST'])
        @self.security_system.require_auth()
        def logout():
            """Endpoint de logout"""
            try:
                token = request.headers.get('Authorization', '').replace('Bearer ', '')
                self.security_system.logout_user(token)
                
                return jsonify({
                    'success': True,
                    'message': 'Logout realizado com sucesso'
                }), 200
                
            except Exception as e:
                logger.error(f"Erro no logout: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/auth/refresh', methods=['POST'])
        def refresh_token():
            """Endpoint para renovar token"""
            try:
                data = request.get_json()
                if not data or not data.get('refresh_token'):
                    return jsonify({'error': 'Refresh token é obrigatório'}), 400
                
                new_tokens = self.security_system.refresh_access_token(data['refresh_token'])
                
                return jsonify({
                    'success': True,
                    'message': 'Token renovado com sucesso',
                    'data': new_tokens
                }), 200
                
            except Exception as e:
                logger.error(f"Erro na renovação do token: {e}")
                return jsonify({'error': str(e)}), 401
        
        # Rotas de análise de fraude
        @self.app.route('/api/v1/fraud/analyze', methods=['POST'])
        @self.security_system.require_auth()
        @self.security_system.require_permission('fraud_analysis')
        def analyze_transaction():
            """Endpoint para análise de transação"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Dados da transação são obrigatórios'}), 400
                
                # Valida dados obrigatórios
                required_fields = ['amount', 'merchant', 'card_number']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Campo {field} é obrigatório'}), 400
                
                # Criptografa dados sensíveis
                if 'card_number' in data:
                    data['card_number'] = self.security_system.encrypt_sensitive_data(data['card_number'])
                
                # Processa análise com sistema de alta performance
                result = self.performance_system.analyze_transaction_fast(data)
                
                # Log de auditoria
                self.security_system._log_audit(
                    user_id=g.current_user['user_id'],
                    action='fraud_analysis',
                    resource='transaction',
                    details=f"Transação analisada - Score: {result.get('fraud_score', 'N/A')}",
                    ip_address=request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr),
                    user_agent=request.headers.get('User-Agent'),
                    success=True
                )
                
                return jsonify({
                    'success': True,
                    'message': 'Análise concluída',
                    'data': result
                }), 200
                
            except Exception as e:
                logger.error(f"Erro na análise de fraude: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/fraud/batch-analyze', methods=['POST'])
        @self.security_system.require_auth()
        @self.security_system.require_permission('fraud_analysis')
        def batch_analyze():
            """Endpoint para análise em lote"""
            try:
                data = request.get_json()
                if not data or not data.get('transactions'):
                    return jsonify({'error': 'Lista de transações é obrigatória'}), 400
                
                transactions = data['transactions']
                if len(transactions) > 1000:
                    return jsonify({'error': 'Máximo de 1000 transações por lote'}), 400
                
                # Processa em lote com sistema de escalabilidade
                results = self.scalability_system.process_batch_transactions(transactions)
                
                # Log de auditoria
                self.security_system._log_audit(
                    user_id=g.current_user['user_id'],
                    action='batch_fraud_analysis',
                    resource='transactions',
                    details=f"Lote de {len(transactions)} transações analisadas",
                    ip_address=request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr),
                    user_agent=request.headers.get('User-Agent'),
                    success=True
                )
                
                return jsonify({
                    'success': True,
                    'message': f'Análise em lote concluída - {len(transactions)} transações',
                    'data': results
                }), 200
                
            except Exception as e:
                logger.error(f"Erro na análise em lote: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Rotas de machine learning
        @self.app.route('/api/v1/ml/retrain', methods=['POST'])
        @self.security_system.require_auth()
        @self.security_system.require_permission('system_config')
        def retrain_models():
            """Endpoint para retreino de modelos"""
            try:
                # Inicia retreino assíncrono
                job_id = self.ml_system.start_async_retraining()
                
                # Log de auditoria
                self.security_system._log_audit(
                    user_id=g.current_user['user_id'],
                    action='model_retrain',
                    resource='ml_models',
                    details=f"Retreino iniciado - Job ID: {job_id}",
                    ip_address=request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr),
                    user_agent=request.headers.get('User-Agent'),
                    success=True
                )
                
                return jsonify({
                    'success': True,
                    'message': 'Retreino iniciado',
                    'data': {'job_id': job_id}
                }), 200
                
            except Exception as e:
                logger.error(f"Erro no retreino: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/ml/status', methods=['GET'])
        @self.security_system.require_auth()
        @self.security_system.require_permission('read_all')
        def ml_status():
            """Endpoint para status do sistema ML"""
            try:
                status = self.ml_system.get_system_status()
                
                return jsonify({
                    'success': True,
                    'message': 'Status obtido',
                    'data': status
                }), 200
                
            except Exception as e:
                logger.error(f"Erro ao obter status ML: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Rotas de monitoramento
        @self.app.route('/api/v1/health', methods=['GET'])
        def health_check():
            """Endpoint de health check"""
            try:
                health_status = {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'services': {
                        'security': self.security_system.health_check(),
                        'ml_engine': self.ml_system.health_check(),
                        'performance': self.performance_system.health_check(),
                        'scalability': self.scalability_system.health_check()
                    }
                }
                
                return jsonify(health_status), 200
                
            except Exception as e:
                logger.error(f"Erro no health check: {e}")
                return jsonify({'status': 'unhealthy', 'error': str(e)}), 500
        
        @self.app.route('/api/v1/metrics', methods=['GET'])
        @self.security_system.require_auth()
        @self.security_system.require_permission('view_dashboard')
        def get_metrics():
            """Endpoint para métricas do sistema"""
            try:
                metrics = {
                    'performance': self.performance_system.get_metrics(),
                    'ml_models': self.ml_system.get_model_metrics(),
                    'security': self.security_system.get_security_metrics()
                }
                
                return jsonify({
                    'success': True,
                    'message': 'Métricas obtidas',
                    'data': metrics
                }), 200
                
            except Exception as e:
                logger.error(f"Erro ao obter métricas: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Rotas administrativas
        @self.app.route('/api/v1/admin/users', methods=['POST'])
        @self.security_system.require_auth()
        @self.security_system.require_permission('manage_users')
        def create_user():
            """Endpoint para criar usuário"""
            try:
                data = request.get_json()
                required_fields = ['username', 'email', 'password', 'role']
                
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Campo {field} é obrigatório'}), 400
                
                user = self.security_system.create_user(
                    username=data['username'],
                    email=data['email'],
                    password=data['password'],
                    role_name=data['role']
                )
                
                return jsonify({
                    'success': True,
                    'message': 'Usuário criado com sucesso',
                    'data': user
                }), 201
                
            except Exception as e:
                logger.error(f"Erro ao criar usuário: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/admin/audit', methods=['GET'])
        @self.security_system.require_auth()
        @self.security_system.require_permission('view_audit')
        def get_audit_logs():
            """Endpoint para logs de auditoria"""
            try:
                limit = request.args.get('limit', 100, type=int)
                offset = request.args.get('offset', 0, type=int)
                
                logs = self.security_system.get_audit_logs(limit=limit, offset=offset)
                
                return jsonify({
                    'success': True,
                    'message': 'Logs de auditoria obtidos',
                    'data': logs
                }), 200
                
            except Exception as e:
                logger.error(f"Erro ao obter logs de auditoria: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _register_error_handlers(self):
        """Registra handlers de erro personalizados"""
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Endpoint não encontrado'}), 404
        
        @self.app.errorhandler(405)
        def method_not_allowed(error):
            return jsonify({'error': 'Método não permitido'}), 405
        
        @self.app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Erro interno: {error}")
            return jsonify({'error': 'Erro interno do servidor'}), 500
        
        @self.app.before_request
        def before_request():
            """Middleware executado antes de cada requisição"""
            # Log da requisição
            logger.info(f"{request.method} {request.path} - IP: {request.remote_addr}")
            
            # Rate limiting básico (será expandido com Redis)
            # TODO: Implementar rate limiting com Redis
            pass
        
        @self.app.after_request
        def after_request(response):
            """Middleware executado após cada requisição"""
            # Headers de segurança
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            
            return response
    
    def create_ssl_context(self):
        """Cria contexto SSL para HTTPS"""
        context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        
        # Certificados (em produção, usar certificados válidos)
        cert_path = '/etc/ssl/certs/sankofa.crt'
        key_path = '/etc/ssl/private/sankofa.key'
        
        if os.path.exists(cert_path) and os.path.exists(key_path):
            context.load_cert_chain(cert_path, key_path)
        else:
            # Gera certificado auto-assinado para desenvolvimento
            logger.warning("Usando certificado auto-assinado para desenvolvimento")
            context.load_cert_chain('cert.pem', 'key.pem')
        
        return context
    
    def run(self, host='0.0.0.0', port=8443, debug=False):
        """Executa a API com HTTPS"""
        try:
            # Cria usuário admin padrão se não existir
            try:
                self.security_system.create_user(
                    username="admin",
                    email="admin@sankofa.com",
                    password="SankoFa2025!@#",
                    role_name="admin"
                )
                logger.info("Usuário admin padrão criado")
            except:
                logger.info("Usuário admin já existe")
            
            # Configura SSL
            ssl_context = self.create_ssl_context()
            
            logger.info(f"🚀 Sankofa Enterprise Pro API iniciada em https://{host}:{port}")
            logger.info("🔒 Segurança Enterprise ativada")
            logger.info("📊 Sistemas ML e Performance integrados")
            
            self.app.run(
                host=host,
                port=port,
                debug=debug,
                ssl_context=ssl_context,
                threaded=True
            )
            
        except Exception as e:
            logger.error(f"Erro ao iniciar API: {e}")
            raise

# Instância global da API
api = SecureMainAPI()

if __name__ == "__main__":
    # Configurações de produção
    import argparse
    
    parser = argparse.ArgumentParser(description='Sankofa Enterprise Pro API')
    parser.add_argument('--host', default='0.0.0.0', help='Host da API')
    parser.add_argument('--port', type=int, default=8443, help='Porta da API')
    parser.add_argument('--debug', action='store_true', help='Modo debug')
    
    args = parser.parse_args()
    
    api.run(host=args.host, port=args.port, debug=args.debug)
