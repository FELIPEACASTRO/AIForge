import { useState, useEffect } from 'react';
import { 
  Shield, 
  Search, 
  Filter, 
  Eye, 
  AlertTriangle,
  CheckCircle,
  Clock,
  User,
  CreditCard,
  MapPin,
  TrendingUp,
  FileText,
  Download,
  Share2,
  Zap,
  RefreshCw
} from 'lucide-react';
import { Button } from '@/components/ui/Button.jsx';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';
import { Badge } from '@/components/ui/Badge.jsx';
import { Input } from '@/components/ui/Input.jsx';
import { SimpleLineChart } from '@/components/charts/SimpleChart';

export function Investigation() {
  const [investigations, setInvestigations] = useState([]);
  const [selectedInvestigation, setSelectedInvestigation] = useState(null);
  const [selectedTransactions, setSelectedTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadingTransactions, setLoadingTransactions] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [priorityFilter, setPriorityFilter] = useState('all');
  const [stats, setStats] = useState({
    active: 0,
    investigating: 0,
    resolved: 0,
    resolutionRate: 0
  });

  useEffect(() => {
    loadInvestigations();
  }, []);

  const loadInvestigations = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/investigations');
      const data = await response.json();
      
      if (data.investigations) {
        setInvestigations(data.investigations);
        calculateStats(data.investigations);
      }
    } catch (error) {
      console.error('Erro ao carregar investigações:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadInvestigationTransactions = async (investigationId) => {
    try {
      setLoadingTransactions(true);
      const response = await fetch(`/api/investigations/${investigationId}/transactions`);
      const data = await response.json();
      
      if (data.transactions) {
        setSelectedTransactions(data.transactions);
      }
    } catch (error) {
      console.error('Erro ao carregar transações da investigação:', error);
    } finally {
      setLoadingTransactions(false);
    }
  };

  const calculateStats = (investigationsData) => {
    const active = investigationsData.filter(inv => inv.status === 'ativo').length;
    const investigating = investigationsData.filter(inv => inv.status === 'investigando').length;
    const resolved = investigationsData.filter(inv => inv.status === 'resolvido').length;
    const total = investigationsData.length;
    const resolutionRate = total > 0 ? Math.round((resolved / total) * 100) : 0;

    setStats({
      active,
      investigating,
      resolved,
      resolutionRate
    });
  };

  const handleInvestigationSelect = (investigation) => {
    setSelectedInvestigation(investigation);
    loadInvestigationTransactions(investigation.id);
  };

  // Filtrar investigações
  const filteredInvestigations = investigations.filter(inv => {
    const matchesSearch = inv.titulo?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         inv.id?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'all' || inv.status === statusFilter;
    const matchesPriority = priorityFilter === 'all' || inv.prioridade === priorityFilter;
    
    return matchesSearch && matchesStatus && matchesPriority;
  });

  const getStatusBadge = (status) => {
    const variants = {
      ativo: 'destructive',
      investigando: 'warning',
      resolvido: 'success',
      fechado: 'secondary'
    };
    
    const labels = {
      ativo: 'Ativo',
      investigando: 'Investigando',
      resolvido: 'Resolvido',
      fechado: 'Fechado'
    };
    
    return <Badge variant={variants[status] || 'secondary'}>{labels[status] || status}</Badge>;
  };

  const getPriorityBadge = (priority) => {
    const variants = {
      alta: 'destructive',
      media: 'warning',
      baixa: 'secondary'
    };
    
    const labels = {
      alta: 'Alta',
      media: 'Média',
      baixa: 'Baixa'
    };
    
    return <Badge variant={variants[priority] || 'secondary'}>{labels[priority] || priority}</Badge>;
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency: 'BRL'
    }).format(value);
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString('pt-BR');
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center space-x-2">
          <RefreshCw className="h-6 w-6 animate-spin" />
          <span>Carregando investigações...</span>
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
            <Shield className="h-8 w-8" />
            <span>Central de Investigação</span>
          </h1>
          <p className="text-[var(--color-text-secondary)] mt-1">
            Análise detalhada de fraudes e casos suspeitos
          </p>
        </div>
        <div className="flex space-x-2">
          <Button variant="secondary" onClick={loadInvestigations}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Atualizar
          </Button>
          <Button className="flex items-center space-x-2">
            <AlertTriangle className="h-4 w-4" />
            <span>Nova Investigação</span>
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <div className="p-2 bg-red-100 rounded-lg">
                <AlertTriangle className="h-4 w-4 text-red-600" />
              </div>
              <div>
                <p className="text-sm text-[var(--color-text-secondary)]">Casos Ativos</p>
                <p className="text-2xl font-bold">{stats.active}</p>
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
                <p className="text-sm text-[var(--color-text-secondary)]">Em Investigação</p>
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
              <div className="p-2 bg-blue-100 rounded-lg">
                <TrendingUp className="h-4 w-4 text-blue-600" />
              </div>
              <div>
                <p className="text-sm text-[var(--color-text-secondary)]">Taxa de Resolução</p>
                <p className="text-2xl font-bold">{stats.resolutionRate}%</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Lista de Investigações */}
        <div className="lg:col-span-2 space-y-4">
          {/* Filtros */}
          <Card>
            <CardContent className="p-4">
              <div className="flex flex-wrap gap-4">
                <div className="flex-1 min-w-[200px]">
                  <Input
                    placeholder="Buscar investigações..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full"
                  />
                </div>
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="px-3 py-2 border border-[var(--color-border)] rounded-md bg-[var(--color-surface)]"
                >
                  <option value="all">Todos os Status</option>
                  <option value="ativo">Ativo</option>
                  <option value="investigando">Investigando</option>
                  <option value="resolvido">Resolvido</option>
                  <option value="fechado">Fechado</option>
                </select>
                <select
                  value={priorityFilter}
                  onChange={(e) => setPriorityFilter(e.target.value)}
                  className="px-3 py-2 border border-[var(--color-border)] rounded-md bg-[var(--color-surface)]"
                >
                  <option value="all">Todas as Prioridades</option>
                  <option value="alta">Alta</option>
                  <option value="media">Média</option>
                  <option value="baixa">Baixa</option>
                </select>
              </div>
            </CardContent>
          </Card>

          {/* Lista */}
          <div className="space-y-4">
            {filteredInvestigations.length === 0 ? (
              <Card>
                <CardContent className="p-8 text-center">
                  <Shield className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-[var(--color-text-secondary)]">
                    Nenhuma investigação encontrada
                  </p>
                </CardContent>
              </Card>
            ) : (
              filteredInvestigations.map((investigation) => (
                <Card 
                  key={investigation.id}
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedInvestigation?.id === investigation.id ? 'ring-2 ring-[var(--color-brand)]' : ''
                  }`}
                  onClick={() => handleInvestigationSelect(investigation)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <h3 className="font-semibold text-[var(--color-text-primary)]">
                            {investigation.titulo}
                          </h3>
                          {getStatusBadge(investigation.status)}
                          {getPriorityBadge(investigation.prioridade)}
                        </div>
                        
                        <p className="text-sm text-[var(--color-text-secondary)] mb-3">
                          {investigation.descricao}
                        </p>
                        
                        {investigation.tags && (
                          <div className="flex flex-wrap gap-1 mb-3">
                            {investigation.tags.map((tag, index) => (
                              <Badge key={index} variant="outline" size="sm">
                                {tag}
                              </Badge>
                            ))}
                          </div>
                        )}
                        
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                          <div>
                            <p className="text-[var(--color-text-secondary)]">ID</p>
                            <p className="font-mono">{investigation.id}</p>
                          </div>
                          <div>
                            <p className="text-[var(--color-text-secondary)]">Transações</p>
                            <p className="font-semibold">{investigation.transacoes_envolvidas || 0}</p>
                          </div>
                          <div>
                            <p className="text-[var(--color-text-secondary)]">Valor Total</p>
                            <p className="font-semibold">{formatCurrency(investigation.valor_total || 0)}</p>
                          </div>
                          <div>
                            <p className="text-[var(--color-text-secondary)]">Risk Score</p>
                            <p className="font-semibold text-red-600">
                              {((investigation.risk_score_medio || 0) * 100).toFixed(1)}%
                            </p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="ml-4 text-right">
                        <p className="text-xs text-[var(--color-text-secondary)]">
                          {formatDate(investigation.data_criacao)}
                        </p>
                        <p className="text-sm font-medium mt-1">
                          {investigation.investigador}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </div>

        {/* Detalhes da Investigação */}
        <div className="space-y-4">
          {selectedInvestigation ? (
            <>
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>Detalhes da Investigação</span>
                    <div className="flex space-x-2">
                      <Button variant="secondary" size="sm">
                        <Download className="h-4 w-4" />
                      </Button>
                      <Button variant="secondary" size="sm">
                        <Share2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">Informações Gerais</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Tipo:</span>
                        <span className="font-medium">{selectedInvestigation.tipo}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Criado em:</span>
                        <span className="font-medium">{formatDate(selectedInvestigation.data_criacao)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Atualizado em:</span>
                        <span className="font-medium">{formatDate(selectedInvestigation.data_atualizacao)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Investigador:</span>
                        <span className="font-medium">{selectedInvestigation.investigador}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">Ações Disponíveis</h4>
                    <div className="space-y-2">
                      <Button variant="outline" size="sm" className="w-full justify-start">
                        <Eye className="h-4 w-4 mr-2" />
                        Ver Transações Relacionadas
                      </Button>
                      <Button variant="outline" size="sm" className="w-full justify-start">
                        <User className="h-4 w-4 mr-2" />
                        Analisar Perfil do Cliente
                      </Button>
                      <Button variant="outline" size="sm" className="w-full justify-start">
                        <MapPin className="h-4 w-4 mr-2" />
                        Análise Geográfica
                      </Button>
                      <Button variant="outline" size="sm" className="w-full justify-start">
                        <FileText className="h-4 w-4 mr-2" />
                        Gerar Relatório
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>Transações Relacionadas</span>
                    {loadingTransactions && <RefreshCw className="h-4 w-4 animate-spin" />}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {loadingTransactions ? (
                    <div className="text-center py-4">
                      <p className="text-[var(--color-text-secondary)]">Carregando transações...</p>
                    </div>
                  ) : selectedTransactions.length === 0 ? (
                    <div className="text-center py-4">
                      <p className="text-[var(--color-text-secondary)]">Nenhuma transação encontrada</p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {selectedTransactions.map((tx) => (
                        <div key={tx.id} className="p-3 border border-[var(--color-border)] rounded-lg">
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-mono text-sm">{tx.id}</span>
                            <Badge variant={tx.status === 'flagged' ? 'destructive' : 'warning'}>
                              {tx.status === 'flagged' ? 'Sinalizada' : 'Em Análise'}
                            </Badge>
                          </div>
                          <div className="text-sm space-y-1">
                            <p><strong>{formatCurrency(tx.valor || tx.amount || 0)}</strong> - {tx.tipo || tx.type}</p>
                            <p>{tx.cliente || tx.customer} ({tx.cpf})</p>
                            <p className="text-[var(--color-text-secondary)]">{tx.localizacao || tx.location}</p>
                            <p className="text-red-600">Risk: {((tx.risk_score || tx.riskScore || 0) * 100).toFixed(1)}%</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </>
          ) : (
            <Card>
              <CardContent className="p-8 text-center">
                <Eye className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                <p className="text-[var(--color-text-secondary)]">
                  Selecione uma investigação para ver os detalhes
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

