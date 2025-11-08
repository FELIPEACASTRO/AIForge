import { useState, useEffect } from 'react';
import { 
  Shield, 
  CreditCard, 
  AlertTriangle, 
  Clock,
  TrendingUp,
  Activity,
  Users,
  DollarSign,
  RefreshCw
} from 'lucide-react';
import { KPICard } from '@/components/charts/KPICard';
import { SimpleLineChart, SimpleAreaChart, SimpleBarChart, SimplePieChart } from '@/components/charts/SimpleChart';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';
import { Badge, TransactionStatusBadge } from '@/components/ui/Badge.jsx';
import { Button } from '@/components/ui/Button.jsx';

export function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [kpis, setKpis] = useState({});
  const [timeSeriesData, setTimeSeriesData] = useState([]);
  const [channelData, setChannelData] = useState([]);
  const [recentAlerts, setRecentAlerts] = useState([]);
  const [modelStatus, setModelStatus] = useState([]);
  const [lastUpdate, setLastUpdate] = useState(null);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      
      // Buscar todos os dados em paralelo
      const [kpisRes, timeseriesRes, channelsRes, alertsRes, modelsRes] = await Promise.all([
        fetch('/api/dashboard/kpis'),
        fetch('/api/dashboard/timeseries'),
        fetch('/api/dashboard/channels'),
        fetch('/api/dashboard/recent-alerts'),
        fetch('/api/dashboard/model-status')
      ]);

      const [kpisData, timeseriesData, channelsData, alertsData, modelsData] = await Promise.all([
        kpisRes.json(),
        timeseriesRes.json(),
        channelsRes.json(),
        alertsRes.json(),
        modelsRes.json()
      ]);

      setKpis(kpisData);
      setTimeSeriesData(timeseriesData.timeseries || []);
      setChannelData(channelsData.channels || []);
      setRecentAlerts(alertsData.alerts || []);
      setModelStatus(modelsData.models || []);
      setLastUpdate(new Date());
      
    } catch (error) {
      console.error('Erro ao buscar dados do dashboard:', error);
      // Em caso de erro, manter os dados anteriores ou mostrar estado de erro
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    
    // Atualizar dados a cada 30 segundos para tempo real
    const interval = setInterval(fetchDashboardData, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const formatCurrency = (value) => {
    if (!value) return 'R$ 0';
    if (value >= 1000000000) {
      return `R$ ${(value / 1000000000).toFixed(1)}B`;
    } else if (value >= 1000000) {
      return `R$ ${(value / 1000000).toFixed(1)}M`;
    } else if (value >= 1000) {
      return `R$ ${(value / 1000).toFixed(1)}K`;
    }
    return `R$ ${value.toLocaleString('pt-BR')}`;
  };

  const formatNumber = (value) => {
    if (!value) return '0';
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`;
    } else if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K`;
    }
    return value.toLocaleString('pt-BR');
  };

  if (loading && !lastUpdate) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-h1">Dashboard Executivo</h1>
        </div>
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-32 bg-[var(--neutral-100)] animate-pulse rounded-lg" />
          ))}
        </div>
        <div className="flex items-center justify-center h-32">
          <div className="flex items-center space-x-2">
            <RefreshCw className="h-6 w-6 animate-spin" />
            <span>Carregando dados do dashboard...</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-h1">Dashboard Executivo</h1>
          <p className="text-[var(--color-text-secondary)] mt-1">
            Visão geral do sistema de detecção de fraudes
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant="success">Sistema Online</Badge>
          <Badge variant="info">{modelStatus.length} Algoritmos Ativos</Badge>
          {lastUpdate && (
            <span className="text-sm text-[var(--color-text-secondary)]">
              Atualizado: {lastUpdate.toLocaleTimeString('pt-BR')}
            </span>
          )}
          <Button variant="secondary" size="sm" onClick={fetchDashboardData} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Atualizar
          </Button>
        </div>
      </div>

      {/* KPIs */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <KPICard
          title="Transações Hoje"
          value={kpis.transacoes_hoje || 0}
          previousValue={kpis.transacoes_ontem || 0}
          format="number"
          icon={CreditCard}
        />
        <KPICard
          title="Fraudes Detectadas"
          value={kpis.fraudes_detectadas || 0}
          previousValue={kpis.fraudes_ontem || 0}
          format="number"
          icon={Shield}
        />
        <KPICard
          title="Taxa de Aprovação"
          value={kpis.taxa_aprovacao || 0}
          previousValue={kpis.taxa_aprovacao_ontem || 0}
          format="percentage"
          icon={TrendingUp}
        />
        <KPICard
          title="Latência Média"
          value={kpis.latencia_media || 0}
          previousValue={kpis.latencia_ontem || 0}
          format="decimal"
          icon={Clock}
        />
      </div>

      {/* Charts Row 1 */}
      <div className="grid gap-6 lg:grid-cols-2">
        <SimpleLineChart
          title="Transações por Hora"
          data={timeSeriesData}
          dataKey="transactions"
          xAxisKey="time"
          formatter={(value) => `${value.toLocaleString('pt-BR')} transações`}
        />
        <SimpleAreaChart
          title="Latência do Sistema"
          data={timeSeriesData}
          dataKey="latency"
          xAxisKey="time"
          color="var(--accent-amber-400)"
          formatter={(value) => `${value}ms`}
        />
      </div>

      {/* Charts Row 2 */}
      <div className="grid gap-6 lg:grid-cols-3">
        <SimpleBarChart
          title="Fraudes por Canal"
          data={channelData}
          dataKey="frauds"
          xAxisKey="name"
          color="var(--error-500)"
          formatter={(value) => `${value} fraudes`}
        />
        
        <SimplePieChart
          title="Distribuição por Canal"
          data={channelData}
          dataKey="value"
          nameKey="name"
        />

        {/* Recent Alerts */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5" />
              <span>Alertas Recentes</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {recentAlerts.length === 0 ? (
                <div className="text-center py-4">
                  <p className="text-[var(--color-text-secondary)]">Nenhum alerta recente</p>
                </div>
              ) : (
                recentAlerts.map((alert) => (
                  <div key={alert.id} className="flex items-start space-x-3 p-3 rounded-lg bg-[var(--neutral-50)]">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-[var(--color-text-primary)]">
                        {alert.message || alert.titulo}
                      </p>
                      <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                        {alert.time || new Date(alert.timestamp).toLocaleString('pt-BR')}
                      </p>
                    </div>
                    <Badge variant={alert.severity === 'critico' ? 'destructive' : 
                                   alert.severity === 'alto' ? 'destructive' :
                                   alert.severity === 'medio' ? 'warning' : 'secondary'} size="sm">
                      {alert.severity === 'critico' ? 'Crítico' : 
                       alert.severity === 'alto' ? 'Alto' :
                       alert.severity === 'medio' ? 'Médio' : 'Baixo'}
                    </Badge>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* System Status */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="h-5 w-5" />
              <span>Status dos Modelos</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {modelStatus.length === 0 ? (
                <div className="text-center py-4">
                  <p className="text-[var(--color-text-secondary)]">Carregando status dos modelos...</p>
                </div>
              ) : (
                modelStatus.map((model) => (
                  <div key={model.name} className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium">{model.name}</p>
                      <p className="text-xs text-[var(--color-text-secondary)]">
                        Precisão: {model.accuracy}%
                      </p>
                    </div>
                    <Badge variant={model.status === 'healthy' ? 'success' : 'warning'}>
                      {model.status === 'healthy' ? 'Saudável' : 'Atenção'}
                    </Badge>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <DollarSign className="h-5 w-5" />
              <span>Valor Protegido</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="text-center">
                <div className="text-h1 font-bold text-[var(--success-500)]">
                  {formatCurrency(kpis.valor_protegido_hoje || 0)}
                </div>
                <p className="text-sm text-[var(--color-text-secondary)]">
                  Valor protegido hoje
                </p>
              </div>
              
              <div className="grid grid-cols-2 gap-4 pt-4 border-t border-[var(--color-border)]">
                <div className="text-center">
                  <div className="text-h3 font-semibold">
                    {formatCurrency(kpis.valor_protegido_ano || 0)}
                  </div>
                  <p className="text-xs text-[var(--color-text-secondary)]">Este ano</p>
                </div>
                <div className="text-center">
                  <div className="text-h3 font-semibold">
                    {formatNumber(kpis.familias_protegidas || 0)}
                  </div>
                  <p className="text-xs text-[var(--color-text-secondary)]">Famílias protegidas</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

