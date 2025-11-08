import { useState, useEffect } from 'react';
import { 
  Bell, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Filter,
  Search,
  Settings,
  Eye,
  EyeOff,
  Trash2,
  MoreVertical,
  Zap,
  Shield,
  TrendingUp,
  Users,
  Server,
  Activity,
  RefreshCw,
  X
} from 'lucide-react';
import { Button } from '@/components/ui/Button.jsx';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';
import { Badge } from '@/components/ui/Badge.jsx';
import { Input } from '@/components/ui/Input.jsx';

export function Alerts() {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState('all');
  const [severityFilter, setSeverityFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [stats, setStats] = useState({
    total: 0,
    new: 0,
    investigating: 0,
    resolved: 0,
    critical: 0
  });

  useEffect(() => {
    loadAlerts();
    // Auto-refresh a cada 30 segundos
    const interval = setInterval(loadAlerts, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadAlerts = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/alerts');
      const data = await response.json();
      
      if (data.alerts) {
        setAlerts(data.alerts);
        calculateStats(data.alerts);
      }
    } catch (error) {
      console.error('Erro ao carregar alertas:', error);
    } finally {
      setLoading(false);
    }
  };

  const calculateStats = (alertsData) => {
    const total = alertsData.length;
    const newAlerts = alertsData.filter(alert => alert.status === 'novo').length;
    const investigating = alertsData.filter(alert => alert.status === 'investigando').length;
    const resolved = alertsData.filter(alert => alert.status === 'resolvido').length;
    const critical = alertsData.filter(alert => alert.severidade === 'critico').length;

    setStats({
      total,
      new: newAlerts,
      investigating,
      resolved,
      critical
    });
  };

  const updateAlertStatus = async (alertId, newStatus) => {
    try {
      const response = await fetch(`/api/alerts/${alertId}/status`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ status: newStatus }),
      });

      if (response.ok) {
        // Atualizar o estado local
        setAlerts(prevAlerts => 
          prevAlerts.map(alert => 
            alert.id === alertId ? { ...alert, status: newStatus } : alert
          )
        );
        
        // Recalcular estatísticas
        const updatedAlerts = alerts.map(alert => 
          alert.id === alertId ? { ...alert, status: newStatus } : alert
        );
        calculateStats(updatedAlerts);
      }
    } catch (error) {
      console.error('Erro ao atualizar status do alerta:', error);
    }
  };

  // Filtrar alertas
  const filteredAlerts = alerts.filter(alert => {
    const matchesSearch = alert.titulo?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         alert.id?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         alert.descricao?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType = typeFilter === 'all' || alert.tipo === typeFilter;
    const matchesSeverity = severityFilter === 'all' || alert.severidade === severityFilter;
    const matchesStatus = statusFilter === 'all' || alert.status === statusFilter;
    
    return matchesSearch && matchesType && matchesSeverity && matchesStatus;
  });

  const getSeverityBadge = (severity) => {
    const variants = {
      critico: 'destructive',
      alto: 'destructive',
      medio: 'warning',
      baixo: 'secondary'
    };
    
    const labels = {
      critico: 'Crítico',
      alto: 'Alto',
      medio: 'Médio',
      baixo: 'Baixo'
    };
    
    return <Badge variant={variants[severity] || 'secondary'}>{labels[severity] || severity}</Badge>;
  };

  const getStatusBadge = (status) => {
    const variants = {
      novo: 'destructive',
      investigando: 'warning',
      resolvido: 'success',
      ignorado: 'secondary'
    };
    
    const labels = {
      novo: 'Novo',
      investigando: 'Investigando',
      resolvido: 'Resolvido',
      ignorado: 'Ignorado'
    };
    
    return <Badge variant={variants[status] || 'secondary'}>{labels[status] || status}</Badge>;
  };

  const getTypeIcon = (type) => {
    const icons = {
      fraud_detected: Shield,
      system_error: AlertTriangle,
      performance_issue: Activity,
      security_alert: Shield,
      model_drift: TrendingUp,
      threshold_exceeded: Zap
    };
    
    const IconComponent = icons[type] || Bell;
    return <IconComponent className="h-4 w-4" />;
  };

  const getTypeLabel = (type) => {
    const labels = {
      fraud_detected: 'Fraude Detectada',
      system_error: 'Erro do Sistema',
      performance_issue: 'Performance',
      security_alert: 'Segurança',
      model_drift: 'Drift do Modelo',
      threshold_exceeded: 'Limite Excedido'
    };
    
    return labels[type] || type;
  };

  const formatCurrency = (value) => {
    if (!value) return null;
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency: 'BRL'
    }).format(value);
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString('pt-BR');
  };

  const getTimeAgo = (dateString) => {
    const now = new Date();
    const alertTime = new Date(dateString);
    const diffMs = now - alertTime;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'Agora';
    if (diffMins < 60) return `${diffMins}m atrás`;
    if (diffHours < 24) return `${diffHours}h atrás`;
    return `${diffDays}d atrás`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center space-x-2">
          <RefreshCw className="h-6 w-6 animate-spin" />
          <span>Carregando alertas...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-h1 flex items-center space-x-2">
            <Bell className="h-8 w-8" />
            <span>Central de Alertas</span>
          </h1>
          <p className="text-[var(--color-text-secondary)] mt-1">
            Monitoramento em tempo real de eventos críticos
          </p>
        </div>
        <div className="flex space-x-2">
          <Button variant="secondary" onClick={loadAlerts}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Atualizar
          </Button>
          <Button variant="secondary">
            <Settings className="h-4 w-4 mr-2" />
            Configurações
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-5">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Bell className="h-4 w-4 text-blue-600" />
              </div>
              <div>
                <p className="text-sm text-[var(--color-text-secondary)]">Total</p>
                <p className="text-2xl font-bold">{stats.total}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <div className="p-2 bg-red-100 rounded-lg">
                <AlertTriangle className="h-4 w-4 text-red-600" />
              </div>
              <div>
                <p className="text-sm text-[var(--color-text-secondary)]">Novos</p>
                <p className="text-2xl font-bold">{stats.new}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <div className="p-2 bg-yellow-100 rounded-lg">
                <Clock className="h-4 w-4 text-yellow-600" />
              </div>
              <div>
                <p className="text-sm text-[var(--color-text-secondary)]">Investigando</p>
                <p className="text-2xl font-bold">{stats.investigating}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <div className="p-2 bg-green-100 rounded-lg">
                <CheckCircle className="h-4 w-4 text-green-600" />
              </div>
              <div>
                <p className="text-sm text-[var(--color-text-secondary)]">Resolvidos</p>
                <p className="text-2xl font-bold">{stats.resolved}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <div className="p-2 bg-red-100 rounded-lg">
                <Zap className="h-4 w-4 text-red-600" />
              </div>
              <div>
                <p className="text-sm text-[var(--color-text-secondary)]">Críticos</p>
                <p className="text-2xl font-bold">{stats.critical}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Lista de Alertas */}
        <div className="lg:col-span-2 space-y-4">
          {/* Filtros */}
          <Card>
            <CardContent className="p-4">
              <div className="flex flex-wrap gap-4">
                <div className="flex-1 min-w-[200px]">
                  <Input
                    placeholder="Buscar alertas..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full"
                  />
                </div>
                <select
                  value={typeFilter}
                  onChange={(e) => setTypeFilter(e.target.value)}
                  className="px-3 py-2 border border-[var(--color-border)] rounded-md bg-[var(--color-surface)]"
                >
                  <option value="all">Todos os Tipos</option>
                  <option value="fraud_detected">Fraude Detectada</option>
                  <option value="system_error">Erro do Sistema</option>
                  <option value="performance_issue">Performance</option>
                  <option value="security_alert">Segurança</option>
                  <option value="model_drift">Drift do Modelo</option>
                  <option value="threshold_exceeded">Limite Excedido</option>
                </select>
                <select
                  value={severityFilter}
                  onChange={(e) => setSeverityFilter(e.target.value)}
                  className="px-3 py-2 border border-[var(--color-border)] rounded-md bg-[var(--color-surface)]"
                >
                  <option value="all">Todas as Severidades</option>
                  <option value="critico">Crítico</option>
                  <option value="alto">Alto</option>
                  <option value="medio">Médio</option>
                  <option value="baixo">Baixo</option>
                </select>
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="px-3 py-2 border border-[var(--color-border)] rounded-md bg-[var(--color-surface)]"
                >
                  <option value="all">Todos os Status</option>
                  <option value="novo">Novo</option>
                  <option value="investigando">Investigando</option>
                  <option value="resolvido">Resolvido</option>
                  <option value="ignorado">Ignorado</option>
                </select>
              </div>
            </CardContent>
          </Card>

          {/* Lista */}
          <div className="space-y-3">
            {filteredAlerts.length === 0 ? (
              <Card>
                <CardContent className="p-8 text-center">
                  <Bell className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-[var(--color-text-secondary)]">
                    Nenhum alerta encontrado
                  </p>
                </CardContent>
              </Card>
            ) : (
              filteredAlerts.map((alert) => (
                <Card 
                  key={alert.id}
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedAlert?.id === alert.id ? 'ring-2 ring-[var(--color-brand)]' : ''
                  } ${alert.status === 'novo' ? 'border-l-4 border-l-red-500' : ''}`}
                  onClick={() => setSelectedAlert(alert)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-3 flex-1">
                        <div className="p-2 bg-gray-100 rounded-lg">
                          {getTypeIcon(alert.tipo)}
                        </div>
                        
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-1">
                            <h3 className="font-semibold text-[var(--color-text-primary)]">
                              {alert.titulo}
                            </h3>
                            {getSeverityBadge(alert.severidade)}
                            {getStatusBadge(alert.status)}
                          </div>
                          
                          <p className="text-sm text-[var(--color-text-secondary)] mb-2">
                            {alert.descricao}
                          </p>
                          
                          <div className="flex items-center space-x-4 text-xs text-[var(--color-text-secondary)]">
                            <span>{getTypeLabel(alert.tipo)}</span>
                            <span>•</span>
                            <span>{alert.id}</span>
                            <span>•</span>
                            <span>{getTimeAgo(alert.timestamp)}</span>
                            {alert.valor_envolvido && (
                              <>
                                <span>•</span>
                                <span className="font-semibold text-red-600">
                                  {formatCurrency(alert.valor_envolvido)}
                                </span>
                              </>
                            )}
                          </div>
                          
                          {alert.tags && (
                            <div className="flex flex-wrap gap-1 mt-2">
                              {alert.tags.map((tag, index) => (
                                <Badge key={index} variant="outline" size="sm">
                                  {tag}
                                </Badge>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        {alert.status === 'novo' && (
                          <Button 
                            size="sm" 
                            variant="secondary"
                            onClick={(e) => {
                              e.stopPropagation();
                              updateAlertStatus(alert.id, 'investigando');
                            }}
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                        )}
                        {alert.status === 'investigando' && (
                          <Button 
                            size="sm" 
                            variant="secondary"
                            onClick={(e) => {
                              e.stopPropagation();
                              updateAlertStatus(alert.id, 'resolvido');
                            }}
                          >
                            <CheckCircle className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </div>

        {/* Detalhes do Alerta */}
        <div className="space-y-4">
          {selectedAlert ? (
            <>
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>Detalhes do Alerta</span>
                    <Button 
                      variant="ghost" 
                      size="sm"
                      onClick={() => setSelectedAlert(null)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">Informações Gerais</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">ID:</span>
                        <span className="font-mono">{selectedAlert.id}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Tipo:</span>
                        <span>{getTypeLabel(selectedAlert.tipo)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Severidade:</span>
                        {getSeverityBadge(selectedAlert.severidade)}
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Status:</span>
                        {getStatusBadge(selectedAlert.status)}
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Criado em:</span>
                        <span>{formatDate(selectedAlert.timestamp)}</span>
                      </div>
                      {selectedAlert.investigador && (
                        <div className="flex justify-between">
                          <span className="text-[var(--color-text-secondary)]">Investigador:</span>
                          <span>{selectedAlert.investigador}</span>
                        </div>
                      )}
                      {selectedAlert.valor_envolvido && (
                        <div className="flex justify-between">
                          <span className="text-[var(--color-text-secondary)]">Valor:</span>
                          <span className="font-semibold text-red-600">
                            {formatCurrency(selectedAlert.valor_envolvido)}
                          </span>
                        </div>
                      )}
                      {selectedAlert.transacao_id && (
                        <div className="flex justify-between">
                          <span className="text-[var(--color-text-secondary)]">Transação:</span>
                          <span className="font-mono">{selectedAlert.transacao_id}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {selectedAlert.acao_recomendada && (
                    <div>
                      <h4 className="font-medium mb-2">Ação Recomendada</h4>
                      <p className="text-sm bg-yellow-50 p-3 rounded-lg border border-yellow-200">
                        {selectedAlert.acao_recomendada}
                      </p>
                    </div>
                  )}
                  
                  <div>
                    <h4 className="font-medium mb-2">Ações</h4>
                    <div className="space-y-2">
                      {selectedAlert.status === 'novo' && (
                        <Button 
                          variant="outline" 
                          size="sm" 
                          className="w-full justify-start"
                          onClick={() => updateAlertStatus(selectedAlert.id, 'investigando')}
                        >
                          <Eye className="h-4 w-4 mr-2" />
                          Iniciar Investigação
                        </Button>
                      )}
                      {selectedAlert.status === 'investigando' && (
                        <Button 
                          variant="outline" 
                          size="sm" 
                          className="w-full justify-start"
                          onClick={() => updateAlertStatus(selectedAlert.id, 'resolvido')}
                        >
                          <CheckCircle className="h-4 w-4 mr-2" />
                          Marcar como Resolvido
                        </Button>
                      )}
                      <Button variant="outline" size="sm" className="w-full justify-start">
                        <Users className="h-4 w-4 mr-2" />
                        Atribuir Investigador
                      </Button>
                      <Button variant="outline" size="sm" className="w-full justify-start">
                        <EyeOff className="h-4 w-4 mr-2" />
                        Ignorar Alerta
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card>
              <CardContent className="p-8 text-center">
                <Bell className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                <p className="text-[var(--color-text-secondary)]">
                  Selecione um alerta para ver os detalhes
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

