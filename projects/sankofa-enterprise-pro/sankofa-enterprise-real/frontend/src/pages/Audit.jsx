import { useState, useEffect } from 'react';
import { 
  FileText, 
  Download, 
  Search, 
  Filter,
  Calendar,
  User,
  Activity,
  Shield,
  AlertTriangle,
  CheckCircle,
  Clock,
  RefreshCw,
  Eye,
  X
} from 'lucide-react';
import { Button } from '@/components/ui/Button.jsx';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';
import { Badge } from '@/components/ui/Badge.jsx';
import { Input } from '@/components/ui/Input.jsx';

export function Audit() {
  const [auditLogs, setAuditLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [actionFilter, setActionFilter] = useState('all');
  const [severityFilter, setSeverityFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [userFilter, setUserFilter] = useState('all');
  const [selectedLog, setSelectedLog] = useState(null);
  const [stats, setStats] = useState({
    total: 0,
    success: 0,
    failed: 0,
    high_severity: 0
  });

  useEffect(() => {
    loadAuditLogs();
  }, []);

  const loadAuditLogs = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/audit');
      const data = await response.json();
      
      if (data.audit_logs) {
        setAuditLogs(data.audit_logs);
        calculateStats(data.audit_logs);
      }
    } catch (error) {
      console.error('Erro ao carregar logs de auditoria:', error);
    } finally {
      setLoading(false);
    }
  };

  const exportAuditLogs = async (filters = {}) => {
    try {
      const response = await fetch('/api/audit/export', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(filters),
      });

      if (response.ok) {
        const data = await response.json();
        // Abrir URL de download em nova aba
        window.open(data.download_url, '_blank');
      }
    } catch (error) {
      console.error('Erro ao exportar logs de auditoria:', error);
    }
  };

  const calculateStats = (logs) => {
    const total = logs.length;
    const success = logs.filter(log => log.status === 'sucesso').length;
    const failed = logs.filter(log => log.status === 'falha').length;
    const high_severity = logs.filter(log => log.severidade === 'alta').length;

    setStats({
      total,
      success,
      failed,
      high_severity
    });
  };

  // Filtrar logs
  const filteredLogs = auditLogs.filter(log => {
    const matchesSearch = log.acao?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         log.recurso?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         log.usuario?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         log.detalhes?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesAction = actionFilter === 'all' || log.acao === actionFilter;
    const matchesSeverity = severityFilter === 'all' || log.severidade === severityFilter;
    const matchesStatus = statusFilter === 'all' || log.status === statusFilter;
    const matchesUser = userFilter === 'all' || log.usuario === userFilter;
    
    return matchesSearch && matchesAction && matchesSeverity && matchesStatus && matchesUser;
  });

  // Obter usuários únicos para filtro
  const users = [...new Set(auditLogs.map(log => log.usuario).filter(Boolean))];

  // Obter ações únicas para filtro
  const actions = [...new Set(auditLogs.map(log => log.acao).filter(Boolean))];

  const getSeverityBadge = (severity) => {
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
    
    return <Badge variant={variants[severity] || 'secondary'}>{labels[severity] || severity}</Badge>;
  };

  const getStatusBadge = (status) => {
    const variants = {
      sucesso: 'success',
      falha: 'destructive',
      atencao: 'warning',
      pendente: 'secondary'
    };
    
    const labels = {
      sucesso: 'Sucesso',
      falha: 'Falha',
      atencao: 'Atenção',
      pendente: 'Pendente'
    };
    
    return <Badge variant={variants[status] || 'secondary'}>{labels[status] || status}</Badge>;
  };

  const getActionIcon = (action) => {
    const icons = {
      'login': User,
      'logout': User,
      'calibragem': Shield,
      'investigacao': AlertTriangle,
      'relatorio': FileText,
      'configuracao': Shield,
      'download': Download,
      'upload': Download
    };
    
    const actionKey = action?.toLowerCase().split(' ')[0];
    const IconComponent = icons[actionKey] || Activity;
    return <IconComponent className="h-4 w-4" />;
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString('pt-BR');
  };

  const getTimeAgo = (dateString) => {
    const now = new Date();
    const logTime = new Date(dateString);
    const diffMs = now - logTime;
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
          <span>Carregando logs de auditoria...</span>
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
            <span>Trilhas de Auditoria</span>
          </h1>
          <p className="text-[var(--color-text-secondary)] mt-1">
            Registro completo de todas as ações realizadas no sistema
          </p>
        </div>
        <div className="flex space-x-2">
          <Button variant="secondary" onClick={loadAuditLogs}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Atualizar
          </Button>
          <Button variant="secondary" onClick={() => exportAuditLogs()}>
            <Download className="h-4 w-4 mr-2" />
            Exportar
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Activity className="h-4 w-4 text-blue-600" />
              </div>
              <div>
                <p className="text-sm text-[var(--color-text-secondary)]">Total de Logs</p>
                <p className="text-2xl font-bold">{stats.total}</p>
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
                <p className="text-sm text-[var(--color-text-secondary)]">Sucessos</p>
                <p className="text-2xl font-bold">{stats.success}</p>
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
                <p className="text-sm text-[var(--color-text-secondary)]">Falhas</p>
                <p className="text-2xl font-bold">{stats.failed}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <div className="p-2 bg-yellow-100 rounded-lg">
                <Shield className="h-4 w-4 text-yellow-600" />
              </div>
              <div>
                <p className="text-sm text-[var(--color-text-secondary)]">Alta Severidade</p>
                <p className="text-2xl font-bold">{stats.high_severity}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Lista de Logs */}
        <div className="lg:col-span-2 space-y-4">
          {/* Filtros */}
          <Card>
            <CardContent className="p-4">
              <div className="flex flex-wrap gap-4">
                <div className="flex-1 min-w-[200px]">
                  <Input
                    placeholder="Buscar logs..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full"
                  />
                </div>
                <select
                  value={actionFilter}
                  onChange={(e) => setActionFilter(e.target.value)}
                  className="px-3 py-2 border border-[var(--color-border)] rounded-md bg-[var(--color-surface)]"
                >
                  <option value="all">Todas as Ações</option>
                  {actions.map(action => (
                    <option key={action} value={action}>{action}</option>
                  ))}
                </select>
                <select
                  value={severityFilter}
                  onChange={(e) => setSeverityFilter(e.target.value)}
                  className="px-3 py-2 border border-[var(--color-border)] rounded-md bg-[var(--color-surface)]"
                >
                  <option value="all">Todas as Severidades</option>
                  <option value="alta">Alta</option>
                  <option value="media">Média</option>
                  <option value="baixa">Baixa</option>
                </select>
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="px-3 py-2 border border-[var(--color-border)] rounded-md bg-[var(--color-surface)]"
                >
                  <option value="all">Todos os Status</option>
                  <option value="sucesso">Sucesso</option>
                  <option value="falha">Falha</option>
                  <option value="atencao">Atenção</option>
                  <option value="pendente">Pendente</option>
                </select>
                <select
                  value={userFilter}
                  onChange={(e) => setUserFilter(e.target.value)}
                  className="px-3 py-2 border border-[var(--color-border)] rounded-md bg-[var(--color-surface)]"
                >
                  <option value="all">Todos os Usuários</option>
                  {users.map(user => (
                    <option key={user} value={user}>{user}</option>
                  ))}
                </select>
              </div>
            </CardContent>
          </Card>

          {/* Lista */}
          <div className="space-y-3">
            {filteredLogs.length === 0 ? (
              <Card>
                <CardContent className="p-8 text-center">
                  <FileText className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-[var(--color-text-secondary)]">
                    Nenhum log de auditoria encontrado
                  </p>
                </CardContent>
              </Card>
            ) : (
              filteredLogs.map((log) => (
                <Card 
                  key={log.id}
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedLog?.id === log.id ? 'ring-2 ring-[var(--color-brand)]' : ''
                  } ${log.severidade === 'alta' ? 'border-l-4 border-l-red-500' : ''}`}
                  onClick={() => setSelectedLog(log)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-3 flex-1">
                        <div className="p-2 bg-gray-100 rounded-lg">
                          {getActionIcon(log.acao)}
                        </div>
                        
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-1">
                            <h3 className="font-semibold text-[var(--color-text-primary)]">
                              {log.acao}
                            </h3>
                            {getSeverityBadge(log.severidade)}
                            {getStatusBadge(log.status)}
                          </div>
                          
                          <p className="text-sm text-[var(--color-text-secondary)] mb-2">
                            {log.detalhes}
                          </p>
                          
                          <div className="flex items-center space-x-4 text-xs text-[var(--color-text-secondary)]">
                            <span className="flex items-center space-x-1">
                              <User className="h-3 w-3" />
                              <span>{log.usuario}</span>
                            </span>
                            <span>•</span>
                            <span>{log.recurso}</span>
                            <span>•</span>
                            <span>{log.id}</span>
                            <span>•</span>
                            <span>{getTimeAgo(log.timestamp)}</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="text-right text-xs text-[var(--color-text-secondary)]">
                        <p>{formatDate(log.timestamp)}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </div>

        {/* Detalhes do Log */}
        <div className="space-y-4">
          {selectedLog ? (
            <>
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>Detalhes do Log</span>
                    <Button 
                      variant="ghost" 
                      size="sm"
                      onClick={() => setSelectedLog(null)}
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
                        <span className="font-mono">{selectedLog.id}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Ação:</span>
                        <span>{selectedLog.acao}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Usuário:</span>
                        <span>{selectedLog.usuario}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Recurso:</span>
                        <span>{selectedLog.recurso}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Severidade:</span>
                        {getSeverityBadge(selectedLog.severidade)}
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Status:</span>
                        {getStatusBadge(selectedLog.status)}
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[var(--color-text-secondary)]">Timestamp:</span>
                        <span>{formatDate(selectedLog.timestamp)}</span>
                      </div>
                      {selectedLog.ip_address && (
                        <div className="flex justify-between">
                          <span className="text-[var(--color-text-secondary)]">IP:</span>
                          <span className="font-mono">{selectedLog.ip_address}</span>
                        </div>
                      )}
                      {selectedLog.user_agent && (
                        <div className="flex justify-between">
                          <span className="text-[var(--color-text-secondary)]">User Agent:</span>
                          <span className="text-xs break-all">{selectedLog.user_agent}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">Detalhes da Ação</h4>
                    <p className="text-sm bg-gray-50 p-3 rounded-lg border">
                      {selectedLog.detalhes}
                    </p>
                  </div>
                  
                  {selectedLog.metadata && (
                    <div>
                      <h4 className="font-medium mb-2">Metadados</h4>
                      <div className="text-sm bg-gray-50 p-3 rounded-lg border">
                        <pre className="whitespace-pre-wrap text-xs">
                          {JSON.stringify(selectedLog.metadata, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )}
                  
                  <div>
                    <h4 className="font-medium mb-2">Ações</h4>
                    <div className="space-y-2">
                      <Button variant="outline" size="sm" className="w-full justify-start">
                        <Eye className="h-4 w-4 mr-2" />
                        Ver Logs Relacionados
                      </Button>
                      <Button variant="outline" size="sm" className="w-full justify-start">
                        <Download className="h-4 w-4 mr-2" />
                        Exportar Log
                      </Button>
                      <Button variant="outline" size="sm" className="w-full justify-start">
                        <User className="h-4 w-4 mr-2" />
                        Ver Atividade do Usuário
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
                  Selecione um log para ver os detalhes
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

