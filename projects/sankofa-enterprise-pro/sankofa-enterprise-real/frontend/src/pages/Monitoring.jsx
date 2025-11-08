import React, { useState, useEffect } from 'react';

const Monitoring = () => {
  const [systemHealth, setSystemHealth] = useState({
    overall_status: 'healthy',
    cpu_usage: 45.2,
    memory_usage: 62.8,
    disk_usage: 34.1,
    network_latency: 12.5,
    active_connections: 156,
    uptime: '15d 8h 23m',
    active_models: 5,
    transactions_per_second: 127,
    avg_response_time: 0.15,
    fraud_detection_rate: 94.2,
    false_positive_rate: 2.1,
    processed_today: 15420
  });

  const [alerts, setAlerts] = useState([
    {
      id: 1,
      type: 'warning',
      message: 'Uso de CPU acima de 80% nos últimos 5 minutos',
      timestamp: new Date().toISOString(),
      severity: 'medium'
    },
    {
      id: 2,
      type: 'info',
      message: 'Retreinamento automático concluído com sucesso',
      timestamp: new Date(Date.now() - 300000).toISOString(),
      severity: 'low'
    }
  ]);

  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        setSystemHealth(prev => ({
          ...prev,
          cpu_usage: Math.max(20, Math.min(90, prev.cpu_usage + (Math.random() - 0.5) * 10)),
          memory_usage: Math.max(30, Math.min(95, prev.memory_usage + (Math.random() - 0.5) * 5)),
          network_latency: Math.max(5, Math.min(50, prev.network_latency + (Math.random() - 0.5) * 5)),
          active_connections: Math.max(100, Math.min(300, prev.active_connections + Math.floor((Math.random() - 0.5) * 20))),
          transactions_per_second: Math.max(50, Math.min(200, prev.transactions_per_second + Math.floor((Math.random() - 0.5) * 20))),
          processed_today: prev.processed_today + Math.floor(Math.random() * 10)
        }));
      }, 3000);

      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getStatusColor = (value, type = 'usage') => {
    if (type === 'usage') {
      if (value >= 85) return 'text-red-600 bg-red-100';
      if (value >= 70) return 'text-yellow-600 bg-yellow-100';
      return 'text-green-600 bg-green-100';
    } else if (type === 'health') {
      if (value >= 95) return 'text-green-600 bg-green-100';
      if (value >= 85) return 'text-yellow-600 bg-yellow-100';
      return 'text-red-600 bg-red-100';
    }
  };

  const getAlertColor = (type) => {
    switch (type) {
      case 'error': return 'border-red-500 bg-red-50';
      case 'warning': return 'border-yellow-500 bg-yellow-50';
      case 'info': return 'border-blue-500 bg-blue-50';
      default: return 'border-gray-500 bg-gray-50';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 flex items-center">
            <span className="mr-3">🖥️</span>
            Monitoramento do Sistema
          </h1>
          <p className="text-gray-600 mt-1">Saúde dos modelos de IA e performance em tempo real</p>
        </div>
        <div className="flex space-x-3">
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-4 py-2 rounded-lg font-medium ${
              autoRefresh 
                ? 'bg-green-600 text-white' 
                : 'bg-gray-200 text-gray-700'
            }`}
          >
            Auto-refresh {autoRefresh ? 'ON' : 'OFF'}
          </button>
          <button className="px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700">
            🔄 Atualizar
          </button>
        </div>
      </div>

      {/* Status Geral */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Status Geral</p>
              <p className="text-2xl font-bold text-green-600">Saudável</p>
            </div>
            <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
              <span className="text-green-600 text-xl">✅</span>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Modelos Ativos</p>
              <p className="text-2xl font-bold text-blue-600">{systemHealth.active_models}</p>
            </div>
            <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
              <span className="text-blue-600 text-xl">⚡</span>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Transações/seg</p>
              <p className="text-2xl font-bold text-purple-600">{systemHealth.transactions_per_second}</p>
            </div>
            <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center">
              <span className="text-purple-600 text-xl">📊</span>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Tempo Resposta</p>
              <p className="text-2xl font-bold text-orange-600">{systemHealth.avg_response_time}s</p>
            </div>
            <div className="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center">
              <span className="text-orange-600 text-xl">⏱️</span>
            </div>
          </div>
        </div>
      </div>

      {/* Métricas de Performance */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Taxa Detecção</p>
              <p className="text-2xl font-bold text-green-600">{systemHealth.fraud_detection_rate}%</p>
            </div>
            <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
              <span className="text-green-600 text-xl">🛡️</span>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Falsos Positivos</p>
              <p className="text-2xl font-bold text-yellow-600">{systemHealth.false_positive_rate}%</p>
            </div>
            <div className="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center">
              <span className="text-yellow-600 text-xl">⚠️</span>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Processadas Hoje</p>
              <p className="text-2xl font-bold text-blue-600">{systemHealth.processed_today.toLocaleString()}</p>
            </div>
            <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
              <span className="text-blue-600 text-xl">📈</span>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Uptime</p>
              <p className="text-2xl font-bold text-green-600">{systemHealth.uptime}</p>
            </div>
            <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
              <span className="text-green-600 text-xl">⏰</span>
            </div>
          </div>
        </div>
      </div>

      {/* Recursos do Sistema */}
      <div className="bg-white rounded-lg shadow-md">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">Recursos do Sistema</h2>
          <p className="text-gray-600 mt-1">Monitoramento em tempo real dos recursos</p>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="w-16 h-16 mx-auto bg-blue-100 rounded-full flex items-center justify-center mb-3">
                <span className="text-blue-600 text-2xl">💻</span>
              </div>
              <p className="text-sm font-medium text-gray-600">CPU</p>
              <p className="text-2xl font-bold text-gray-900">{systemHealth.cpu_usage.toFixed(1)}%</p>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(systemHealth.cpu_usage)}`}>
                {systemHealth.cpu_usage >= 85 ? 'Alto' : systemHealth.cpu_usage >= 70 ? 'Médio' : 'Normal'}
              </span>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 mx-auto bg-green-100 rounded-full flex items-center justify-center mb-3">
                <span className="text-green-600 text-2xl">💾</span>
              </div>
              <p className="text-sm font-medium text-gray-600">Memória</p>
              <p className="text-2xl font-bold text-gray-900">{systemHealth.memory_usage.toFixed(1)}%</p>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(systemHealth.memory_usage)}`}>
                {systemHealth.memory_usage >= 85 ? 'Alto' : systemHealth.memory_usage >= 70 ? 'Médio' : 'Normal'}
              </span>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 mx-auto bg-yellow-100 rounded-full flex items-center justify-center mb-3">
                <span className="text-yellow-600 text-2xl">💿</span>
              </div>
              <p className="text-sm font-medium text-gray-600">Disco</p>
              <p className="text-2xl font-bold text-gray-900">{systemHealth.disk_usage.toFixed(1)}%</p>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(systemHealth.disk_usage)}`}>
                {systemHealth.disk_usage >= 85 ? 'Alto' : systemHealth.disk_usage >= 70 ? 'Médio' : 'Normal'}
              </span>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 mx-auto bg-purple-100 rounded-full flex items-center justify-center mb-3">
                <span className="text-purple-600 text-2xl">🌐</span>
              </div>
              <p className="text-sm font-medium text-gray-600">Latência</p>
              <p className="text-2xl font-bold text-gray-900">{systemHealth.network_latency.toFixed(1)}ms</p>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(100 - systemHealth.network_latency, 'health')}`}>
                {systemHealth.network_latency <= 20 ? 'Excelente' : systemHealth.network_latency <= 50 ? 'Bom' : 'Lento'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Alertas e Conexões */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-md">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900">Alertas Recentes</h2>
          </div>
          <div className="p-6">
            {alerts.length === 0 ? (
              <div className="text-center py-8">
                <span className="text-6xl">✅</span>
                <p className="text-gray-500 mt-2">Nenhum alerta ativo</p>
              </div>
            ) : (
              <div className="space-y-3">
                {alerts.map((alert) => (
                  <div key={alert.id} className={`p-4 rounded-lg border-l-4 ${getAlertColor(alert.type)}`}>
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="font-medium text-gray-900">{alert.message}</p>
                        <p className="text-sm text-gray-500 mt-1">
                          {new Date(alert.timestamp).toLocaleString('pt-BR')}
                        </p>
                      </div>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        alert.severity === 'high' ? 'bg-red-100 text-red-800' :
                        alert.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-blue-100 text-blue-800'
                      }`}>
                        {alert.severity === 'high' ? 'Alta' : alert.severity === 'medium' ? 'Média' : 'Baixa'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900">Informações do Sistema</h2>
          </div>
          <div className="p-6 space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Conexões Ativas</span>
              <span className="font-medium text-gray-900">{systemHealth.active_connections}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Uptime do Sistema</span>
              <span className="font-medium text-gray-900">{systemHealth.uptime}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Última Atualização</span>
              <span className="font-medium text-gray-900">{new Date().toLocaleTimeString('pt-BR')}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Status dos Modelos</span>
              <span className="font-medium text-green-600">Todos Online</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Modo de Operação</span>
              <span className="font-medium text-blue-600">Produção</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Auto-refresh</span>
              <span className={`font-medium ${autoRefresh ? 'text-green-600' : 'text-gray-600'}`}>
                {autoRefresh ? 'Ativo' : 'Inativo'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};



export default Monitoring;

