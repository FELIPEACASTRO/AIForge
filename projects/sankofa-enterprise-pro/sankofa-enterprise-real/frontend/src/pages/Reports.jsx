import { useState, useEffect } from 'react';
import { 
  FileText, 
  Download, 
  Calendar, 
  Filter,
  BarChart3,
  PieChart,
  TrendingUp,
  DollarSign,
  Shield,
  Users,
  Clock,
  Target,
  Eye,
  Share2,
  Plus,
  RefreshCw,
  Search
} from 'lucide-react';
import { Button } from '@/components/ui/Button.jsx';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';
import { Badge } from '@/components/ui/Badge.jsx';
import { Input } from '@/components/ui/Input.jsx';
import { SimpleLineChart } from '@/components/charts/SimpleChart';

export function Reports() {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [selectedReport, setSelectedReport] = useState(null);
  const [generatingReport, setGeneratingReport] = useState(false);

  useEffect(() => {
    loadReports();
  }, []);

  const loadReports = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/reports');
      const data = await response.json();
      
      if (data.reports) {
        setReports(data.reports);
      }
    } catch (error) {
      console.error('Erro ao carregar relatórios:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateReport = async (templateId, params = {}) => {
    try {
      setGeneratingReport(true);
      const response = await fetch('/api/reports/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          template: templateId,
          parameters: params
        }),
      });

      if (response.ok) {
        const data = await response.json();
        // Recarregar lista de relatórios
        await loadReports();
        return data;
      }
    } catch (error) {
      console.error('Erro ao gerar relatório:', error);
    } finally {
      setGeneratingReport(false);
    }
  };

  const downloadReport = async (reportId) => {
    try {
      const response = await fetch(`/api/reports/${reportId}/download`);
      if (response.ok) {
        const data = await response.json();
        // Abrir URL de download em nova aba
        window.open(data.download_url, '_blank');
      }
    } catch (error) {
      console.error('Erro ao baixar relatório:', error);
    }
  };

  // Filtrar relatórios
  const filteredReports = reports.filter(report => {
    const matchesSearch = report.titulo?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         report.id?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         report.descricao?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType = typeFilter === 'all' || report.tipo === typeFilter;
    const matchesStatus = statusFilter === 'all' || report.status === statusFilter;
    
    return matchesSearch && matchesType && matchesStatus;
  });

  const getStatusBadge = (status) => {
    const variants = {
      concluido: 'success',
      gerando: 'warning',
      erro: 'destructive',
      pendente: 'secondary'
    };
    
    const labels = {
      concluido: 'Concluído',
      gerando: 'Gerando',
      erro: 'Erro',
      pendente: 'Pendente'
    };
    
    return <Badge variant={variants[status] || 'secondary'}>{labels[status] || status}</Badge>;
  };

  const getTypeIcon = (type) => {
    const icons = {
      fraud_analysis: Shield,
      performance: BarChart3,
      trend_analysis: TrendingUp,
      financial: DollarSign,
      user_activity: Users,
      system_health: Target
    };
    
    const IconComponent = icons[type] || FileText;
    return <IconComponent className="h-4 w-4" />;
  };

  const getTypeLabel = (type) => {
    const labels = {
      fraud_analysis: 'Análise de Fraudes',
      performance: 'Performance',
      trend_analysis: 'Análise de Tendências',
      financial: 'Financeiro',
      user_activity: 'Atividade de Usuários',
      system_health: 'Saúde do Sistema'
    };
    
    return labels[type] || type;
  };

  const formatCurrency = (value) => {
    if (!value) return 'N/A';
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency: 'BRL'
    }).format(value);
  };

  const formatNumber = (value) => {
    if (!value) return 'N/A';
    return new Intl.NumberFormat('pt-BR').format(value);
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString('pt-BR');
  };

  const reportTemplates = [
    {
      id: 'fraud_monthly',
      title: 'Relatório Mensal de Fraudes',
      description: 'Análise completa das fraudes detectadas no período',
      icon: Shield,
      estimatedTime: '5-10 min'
    },
    {
      id: 'performance_quarterly',
      title: 'Performance Trimestral',
      description: 'Avaliação da performance dos algoritmos de IA',
      icon: BarChart3,
      estimatedTime: '3-5 min'
    },
    {
      id: 'trend_analysis',
      title: 'Análise de Tendências',
      description: 'Identificação de padrões e tendências emergentes',
      icon: TrendingUp,
      estimatedTime: '7-12 min'
    },
    {
      id: 'financial_impact',
      title: 'Impacto Financeiro',
      description: 'Análise do impacto financeiro das fraudes detectadas',
      icon: DollarSign,
      estimatedTime: '4-8 min'
    }
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center space-x-2">
          <RefreshCw className="h-6 w-6 animate-spin" />
          <span>Carregando relatórios...</span>
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
            <FileText className="h-8 w-8" />
            <span>Central de Relatórios</span>
          </h1>
          <p className="text-[var(--color-text-secondary)] mt-1">
            Geração e análise de relatórios detalhados
          </p>
        </div>
        <div className="flex space-x-2">
          <Button variant="secondary" onClick={loadReports}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Atualizar
          </Button>
          <Button>
            <Plus className="h-4 w-4 mr-2" />
            Novo Relatório
          </Button>
        </div>
      </div>

      {/* Templates de Relatórios */}
      <Card>
        <CardHeader>
          <CardTitle>Templates Disponíveis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {reportTemplates.map((template) => {
              const IconComponent = template.icon;
              return (
                <Card key={template.id} className="cursor-pointer hover:shadow-md transition-shadow">
                  <CardContent className="p-4">
                    <div className="flex items-start space-x-3">
                      <div className="p-2 bg-blue-100 rounded-lg">
                        <IconComponent className="h-5 w-5 text-blue-600" />
                      </div>
                      <div className="flex-1">
                        <h3 className="font-semibold text-sm mb-1">{template.title}</h3>
                        <p className="text-xs text-[var(--color-text-secondary)] mb-2">
                          {template.description}
                        </p>
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-[var(--color-text-secondary)]">
                            {template.estimatedTime}
                          </span>
                          <Button 
                            size="sm" 
                            variant="outline"
                            onClick={() => generateReport(template.id)}
                            disabled={generatingReport}
                          >
                            {generatingReport ? (
                              <RefreshCw className="h-3 w-3 animate-spin" />
                            ) : (
                              <Plus className="h-3 w-3" />
                            )}
                          </Button>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Lista de Relatórios */}
        <div className="lg:col-span-2 space-y-4">
          {/* Filtros */}
          <Card>
            <CardContent className="p-4">
              <div className="flex flex-wrap gap-4">
                <div className="flex-1 min-w-[200px]">
                  <Input
                    placeholder="Buscar relatórios..."
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
                  <option value="fraud_analysis">Análise de Fraudes</option>
                  <option value="performance">Performance</option>
                  <option value="trend_analysis">Análise de Tendências</option>
                  <option value="financial">Financeiro</option>
                  <option value="user_activity">Atividade de Usuários</option>
                  <option value="system_health">Saúde do Sistema</option>
                </select>
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="px-3 py-2 border border-[var(--color-border)] rounded-md bg-[var(--color-surface)]"
                >
                  <option value="all">Todos os Status</option>
                  <option value="concluido">Concluído</option>
                  <option value="gerando">Gerando</option>
                  <option value="erro">Erro</option>
                  <option value="pendente">Pendente</option>
                </select>
              </div>
            </CardContent>
          </Card>

          {/* Lista */}
          <div className="space-y-4">
            {filteredReports.length === 0 ? (
              <Card>
                <CardContent className="p-8 text-center">
                  <FileText className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-[var(--color-text-secondary)]">
                    Nenhum relatório encontrado
                  </p>
                </CardContent>
              </Card>
            ) : (
              filteredReports.map((report) => (
                <Card 
                  key={report.id}
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedReport?.id === report.id ? 'ring-2 ring-[var(--color-brand)]' : ''
                  }`}
                  onClick={() => setSelectedReport(report)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-3 flex-1">
                        <div className="p-2 bg-gray-100 rounded-lg">
                          {getTypeIcon(report.tipo)}
                        </div>
                        
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-1">
                            <h3 className="font-semibold text-[var(--color-text-primary)]">
                              {report.titulo}
                            </h3>
                            {getStatusBadge(report.status)}
                          </div>
                          
                          <p className="text-sm text-[var(--color-text-secondary)] mb-2">
                            {report.descricao}
                          </p>
                          
                          <div className="flex items-center space-x-4 text-xs text-[var(--color-text-secondary)]">
                            <span>{getTypeLabel(report.tipo)}</span>
                            <span>•</span>
                            <span>{report.id}</span>
                            <span>•</span>
                            <span>{formatDate(report.data_criacao)}</span>
                            <span>•</span>
                            <span>Por: {report.gerado_por}</span>
                          </div>
                          
                          {report.tags && (
                            <div className="flex flex-wrap gap-1 mt-2">
                              {report.tags.map((tag, index) => (
                                <Badge key={index} variant="outline" size="sm">
                                  {tag}
                                </Badge>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        {report.status === 'concluido' && (
                          <Button 
                            size="sm" 
                            variant="secondary"
                            onClick={(e) => {
                              e.stopPropagation();
                              downloadReport(report.id);
                            }}
                          >
                            <Download className="h-4 w-4" />
                          </Button>
                        )}
                        <Button 
                          size="sm" 
                          variant="secondary"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedReport(report);
                          }}
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </div>

        {/* Detalhes do Relatório */}
        <div className="space-y-4">
          {selectedReport ? (
            <>
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>Detalhes do Relatório</span>
                    <div className="flex space-x-2">
                      {selectedReport.status === 'concluido' && (
                        <Button 
                          variant="secondary" 
                          size="sm"
                          onClick={() => downloadReport(selectedReport.id)}
                        >
                          <Download className="h-4 w-4" />
                        </Button>
                      )}
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
                        <span className="text-[var(--color-text-secondary)]">ID:</span>
                        <span className="font-mono">{selectedReport.id}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Tipo:</span>
                        <span>{getTypeLabel(selectedReport.tipo)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Status:</span>
                        {getStatusBadge(selectedReport.status)}
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Criado em:</span>
                        <span>{formatDate(selectedReport.data_criacao)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Gerado por:</span>
                        <span>{selectedReport.gerado_por}</span>
                      </div>
                      {selectedReport.tamanho_arquivo && (
                        <div className="flex justify-between">
                          <span className="text-[var(--color-text-secondary)]">Tamanho:</span>
                          <span>{selectedReport.tamanho_arquivo}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {selectedReport.metricas && (
                    <div>
                      <h4 className="font-medium mb-2">Métricas Principais</h4>
                      <div className="space-y-2 text-sm">
                        {Object.entries(selectedReport.metricas).map(([key, value]) => (
                          <div key={key} className="flex justify-between">
                            <span className="text-[var(--color-text-secondary)] capitalize">
                              {key.replace(/_/g, ' ')}:
                            </span>
                            <span className="font-medium">
                              {typeof value === 'number' && value > 1000000 
                                ? formatCurrency(value)
                                : formatNumber(value)
                              }
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <div>
                    <h4 className="font-medium mb-2">Ações</h4>
                    <div className="space-y-2">
                      {selectedReport.status === 'concluido' && (
                        <Button 
                          variant="outline" 
                          size="sm" 
                          className="w-full justify-start"
                          onClick={() => downloadReport(selectedReport.id)}
                        >
                          <Download className="h-4 w-4 mr-2" />
                          Baixar Relatório
                        </Button>
                      )}
                      <Button variant="outline" size="sm" className="w-full justify-start">
                        <Share2 className="h-4 w-4 mr-2" />
                        Compartilhar
                      </Button>
                      <Button variant="outline" size="sm" className="w-full justify-start">
                        <Calendar className="h-4 w-4 mr-2" />
                        Agendar Geração
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card>
              <CardContent className="p-8 text-center">
                <FileText className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                <p className="text-[var(--color-text-secondary)]">
                  Selecione um relatório para ver os detalhes
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

