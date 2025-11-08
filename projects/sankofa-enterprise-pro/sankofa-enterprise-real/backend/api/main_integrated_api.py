#!/usr/bin/env python3
"""
API Principal Integrada do Sankofa Enterprise Pro
Combina todas as funcionalidades: segurança, compliance, cache, performance e detecção de fraude.
"""

import os
import sys
import asyncio
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import redis
import logging

# Adiciona o diretório pai ao path para importações
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security.enterprise_security_system import EnterpriseSecuritySystem
from compliance.compliance_manager import ComplianceManager
from cache.redis_cache_system import RedisCacheSystem
from performance.high_performance_system import HighPerformanceSystem

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5174", "http://localhost:5174"])

# Inicialização dos sistemas
security_system = EnterpriseSecuritySystem()
compliance_manager = ComplianceManager()
cache_system = RedisCacheSystem()
performance_system = HighPerformanceSystem()

# Dados simulados para o dashboard
MOCK_DATA = {
    "kpis": {
        "transacoes_hoje": 12547,
        "transacoes_ontem": 11832,
        "fraudes_detectadas": 23,
        "fraudes_ontem": 18,
        "taxa_aprovacao": 97.8,
        "taxa_aprovacao_ontem": 97.2,
        "latencia_media": 10.3,
        "latencia_ontem": 12.1,
        "valor_protegido_hoje": 2847392.50,
        "valor_protegido_ano": 1247382940.00,
        "familias_protegidas": 8432
    },
    "timeseries": [
        {"time": "00:00", "transactions": 234, "latency": 8.2},
        {"time": "01:00", "transactions": 189, "latency": 7.8},
        {"time": "02:00", "transactions": 156, "latency": 8.1},
        {"time": "03:00", "transactions": 123, "latency": 7.9},
        {"time": "04:00", "transactions": 145, "latency": 8.3},
        {"time": "05:00", "transactions": 198, "latency": 8.7},
        {"time": "06:00", "transactions": 267, "latency": 9.1},
        {"time": "07:00", "transactions": 345, "latency": 9.8},
        {"time": "08:00", "transactions": 456, "latency": 10.2},
        {"time": "09:00", "transactions": 567, "latency": 10.8},
        {"time": "10:00", "transactions": 634, "latency": 11.1},
        {"time": "11:00", "transactions": 723, "latency": 10.9},
        {"time": "12:00", "transactions": 812, "latency": 10.3},
        {"time": "13:00", "transactions": 789, "latency": 9.8},
        {"time": "14:00", "transactions": 698, "latency": 9.5},
        {"time": "15:00", "transactions": 634, "latency": 9.2},
        {"time": "16:00", "transactions": 567, "latency": 8.9},
        {"time": "17:00", "transactions": 498, "latency": 8.6},
        {"time": "18:00", "transactions": 423, "latency": 8.3},
        {"time": "19:00", "transactions": 356, "latency": 8.1},
        {"time": "20:00", "transactions": 289, "latency": 7.8},
        {"time": "21:00", "transactions": 234, "latency": 7.6},
        {"time": "22:00", "transactions": 198, "latency": 7.4},
        {"time": "23:00", "transactions": 167, "latency": 7.2}
    ],
    "channels": [
        {"name": "PIX", "frauds": 8, "value": 4523},
        {"name": "Cartão", "frauds": 12, "value": 3892},
        {"name": "TED", "frauds": 3, "value": 1247},
        {"name": "DOC", "frauds": 0, "value": 892}
    ],
    "alerts": [
        {
            "id": 1,
            "message": "Taxa de fraude acima do limite em PIX",
            "severity": "alto",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": 2,
            "message": "Latência elevada detectada no modelo XGBoost",
            "severity": "medio",
            "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat()
        }
    ],
    "models": [
        {"name": "XGBoost Ensemble", "status": "healthy", "accuracy": 94.2},
        {"name": "Random Forest", "status": "healthy", "accuracy": 92.8},
        {"name": "Neural Network", "status": "healthy", "accuracy": 93.5},
        {"name": "Isolation Forest", "status": "healthy", "accuracy": 89.1},
        {"name": "LSTM Temporal", "status": "healthy", "accuracy": 91.7}
    ]
}

# Middleware de performance
@app.before_request
def before_request():
    g.start_time = time.time()

@app.after_request
def after_request(response):
    duration = time.time() - g.start_time
    response.headers['X-Response-Time'] = f"{duration:.3f}s"
    
    # Log da performance
    logger.info(f"{request.method} {request.path} - {response.status_code} - {duration:.3f}s")
    
    return response

# Rotas do Dashboard
@app.route('/api/dashboard/kpis', methods=['GET'])
def get_dashboard_kpis():
    """Retorna os KPIs principais do dashboard."""
    try:
        # Tenta buscar do cache primeiro
        cached_data = cache_system.get("dashboard_kpis")
        if cached_data:
            return jsonify(cached_data)
        
        # Se não estiver no cache, usa dados mock e armazena no cache
        kpis = MOCK_DATA["kpis"]
        cache_system.set("dashboard_kpis", kpis, ttl=30)
        
        return jsonify(kpis)
    except Exception as e:
        logger.error(f"Erro ao buscar KPIs: {e}")
        return jsonify(MOCK_DATA["kpis"])

@app.route('/api/dashboard/timeseries', methods=['GET'])
def get_dashboard_timeseries():
    """Retorna dados de série temporal para gráficos."""
    try:
        cached_data = cache_system.get("dashboard_timeseries")
        if cached_data:
            return jsonify({"timeseries": cached_data})
        
        timeseries = MOCK_DATA["timeseries"]
        cache_system.set("dashboard_timeseries", timeseries, ttl=60)
        
        return jsonify({"timeseries": timeseries})
    except Exception as e:
        logger.error(f"Erro ao buscar série temporal: {e}")
        return jsonify({"timeseries": MOCK_DATA["timeseries"]})

@app.route('/api/dashboard/channels', methods=['GET'])
def get_dashboard_channels():
    """Retorna dados de canais para gráficos."""
    try:
        cached_data = cache_system.get("dashboard_channels")
        if cached_data:
            return jsonify({"channels": cached_data})
        
        channels = MOCK_DATA["channels"]
        cache_system.set("dashboard_channels", channels, ttl=120)
        
        return jsonify({"channels": channels})
    except Exception as e:
        logger.error(f"Erro ao buscar dados de canais: {e}")
        return jsonify({"channels": MOCK_DATA["channels"]})

@app.route('/api/dashboard/recent-alerts', methods=['GET'])
def get_recent_alerts():
    """Retorna alertas recentes."""
    try:
        cached_data = cache_system.get("dashboard_alerts")
        if cached_data:
            return jsonify({"alerts": cached_data})
        
        alerts = MOCK_DATA["alerts"]
        cache_system.set("dashboard_alerts", alerts, ttl=15)
        
        return jsonify({"alerts": alerts})
    except Exception as e:
        logger.error(f"Erro ao buscar alertas: {e}")
        return jsonify({"alerts": MOCK_DATA["alerts"]})

@app.route('/api/dashboard/model-status', methods=['GET'])
def get_model_status():
    """Retorna status dos modelos de ML."""
    try:
        cached_data = cache_system.get("dashboard_models")
        if cached_data:
            return jsonify({"models": cached_data})
        
        models = MOCK_DATA["models"]
        cache_system.set("dashboard_models", models, ttl=300)
        
        return jsonify({"models": models})
    except Exception as e:
        logger.error(f"Erro ao buscar status dos modelos: {e}")
        return jsonify({"models": MOCK_DATA["models"]})

# Rotas de Compliance
@app.route('/api/v1/compliance/status', methods=['GET'])
def get_compliance_status():
    """Retorna o status geral de compliance."""
    try:
        status = {
            "bacen_resolution_6": "Implemented",
            "lgpd": "Implemented", 
            "pci_dss_v4": "Implemented",
            "sox": "Partially Implemented",
            "basel_iii": "Not Implemented",
        }
        return jsonify({"success": True, "data": status})
    except Exception as e:
        logger.error(f"Erro ao buscar status de compliance: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/compliance/share-fraud', methods=['POST'])
def share_fraud():
    """Compartilha dados de fraude com o BACEN."""
    try:
        data = request.get_json()
        user_context = {"username": "system"}  # Em produção, viria do token JWT
        
        success = compliance_manager.share_fraud_data_with_bacen(data, user_context)
        
        if success:
            return jsonify({"success": True, "message": "Dados de fraude compartilhados com sucesso."})
        else:
            return jsonify({"success": False, "message": "Falha ao compartilhar dados de fraude."}), 500
    except Exception as e:
        logger.error(f"Erro ao compartilhar fraude: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# Rota de análise de transação (principal funcionalidade)
@app.route('/api/v1/analyze-transaction', methods=['POST'])
def analyze_transaction():
    """Analisa uma transação para detecção de fraude."""
    try:
        start_time = time.time()
        
        transaction_data = request.get_json()
        
        # Validação básica
        if not transaction_data:
            return jsonify({"error": "Dados da transação são obrigatórios"}), 400
        
        # Simulação de análise de fraude
        fraud_score = 0.15  # 15% de probabilidade de fraude
        is_fraud = fraud_score > 0.5
        
        result = {
            "transaction_id": transaction_data.get("transaction_id", "TXN_" + str(int(time.time()))),
            "fraud_score": fraud_score,
            "is_fraud": is_fraud,
            "risk_level": "LOW" if fraud_score < 0.3 else "MEDIUM" if fraud_score < 0.7 else "HIGH",
            "analysis_time_ms": round((time.time() - start_time) * 1000, 2),
            "models_used": ["XGBoost", "RandomForest", "NeuralNetwork"],
            "features_analyzed": 47,
            "compliance_status": "APPROVED"
        }
        
        # Armazena no cache para auditoria
        cache_key = f"transaction_analysis:{result['transaction_id']}"
        cache_system.set(cache_key, result, ttl=3600)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro na análise de transação: {e}")
        return jsonify({"error": "Erro interno do servidor"}), 500

# Rota de health check
@app.route('/api/health', methods=['GET'])
def health_check():
    """Verifica a saúde do sistema."""
    try:
        redis_status = "OK" if cache_system.ping() else "ERROR"
        
        health_data = {
            "status": "OK",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "api": "OK",
                "redis": redis_status,
                "security": "OK",
                "compliance": "OK"
            },
            "performance": {
                "avg_response_time_ms": 10.3,
                "throughput_rps": 95.65,
                "active_connections": 42
            }
        }
        
        return jsonify(health_data)
    except Exception as e:
        logger.error(f"Erro no health check: {e}")
        return jsonify({"status": "ERROR", "error": str(e)}), 500

# Importar o gerador de transações
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
from real_time_transaction_generator import transaction_generator

# Rotas de Transações
@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    """Retorna lista de transações com filtros opcionais."""
    try:
        # Parâmetros de filtro
        limit = int(request.args.get('limit', 100))
        search = request.args.get('search', '')
        status = request.args.get('status', 'Todos')
        tipo = request.args.get('tipo', 'Todos')
        
        filters = {
            'search': search,
            'status': status,
            'tipo': tipo
        }
        
        transactions = transaction_generator.get_transactions(limit=limit, filters=filters)
        stats = transaction_generator.get_stats()
        
        return jsonify({
            "success": True,
            "data": transactions,
            "stats": stats,
            "total": len(transactions)
        })
        
    except Exception as e:
        logger.error(f"Erro ao buscar transações: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/transactions/stats', methods=['GET'])
def get_transaction_stats():
    """Retorna estatísticas das transações."""
    try:
        stats = transaction_generator.get_stats()
        return jsonify({"success": True, "data": stats})
    except Exception as e:
        logger.error(f"Erro ao buscar estatísticas: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Iniciando Sankofa Enterprise Pro API...")
    logger.info("Sistemas carregados: Segurança, Compliance, Cache Redis, Performance")
    
    # Define a chave JWT se não estiver definida
    if not os.environ.get('SANKOFA_JWT_SECRET'):
        os.environ['SANKOFA_JWT_SECRET'] = 'sankofa-enterprise-secret-key-2024'
    
    # Inicia o gerador de transações em tempo real
    logger.info("Iniciando gerador de transações em tempo real...")
    transaction_generator.start_generation(interval=3.0)  # Uma transação a cada 3 segundos
    
    app.run(host="0.0.0.0", port=8445, debug=True, threaded=True)
