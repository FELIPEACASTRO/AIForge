import { useState, useEffect, useMemo } from 'react';
import { 
  Search, 
  Filter, 
  Download, 
  Eye,
  MoreHorizontal,
  Calendar,
  RefreshCw
} from 'lucide-react';
import { Button } from '@/components/ui/Button.jsx';
import { Input, FormField } from '@/components/ui/Input.jsx';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';
import { Badge, TransactionStatusBadge, RiskScoreBadge } from '@/components/ui/Badge.jsx';

export function Transactions() {
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('TODOS');
  const [typeFilter, setTypeFilter] = useState('TODOS');
  const [sortField, setSortField] = useState('timestamp');
  const [sortDirection, setSortDirection] = useState('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalTransactions, setTotalTransactions] = useState(0);

  useEffect(() => {
    loadTransactions();
  }, [currentPage, searchQuery, statusFilter, typeFilter]);

  const loadTransactions = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams({
        page: currentPage,
        limit: 50,
        ...(searchQuery && { search: searchQuery }),
        ...(statusFilter !== 'TODOS' && { status: statusFilter }),
        ...(typeFilter !== 'TODOS' && { type: typeFilter })
      });

      const response = await fetch(`/api/transactions?${params}`);
      const data = await response.json();
      
      if (data.success) {
        setTransactions(data.data || []);
        setTotalPages(Math.ceil((data.stats?.total || 0) / 50));
        setTotalTransactions(data.stats?.total || 0);
      } else {
        console.error('Erro na resposta da API:', data.error);
        setTransactions([]);
      }
    } catch (error) {
      console.error('Erro ao carregar transações:', error);
      setTransactions([]);
    } finally {
      setLoading(false);
    }
  };

  // Filtrar e ordenar transações localmente
  const filteredTransactions = useMemo(() => {
    let filtered = [...transactions];

    // Ordenar
    filtered.sort((a, b) => {
      let aValue = a[sortField];
      let bValue = b[sortField];
      
      if (sortField === 'timestamp') {
        aValue = new Date(aValue);
        bValue = new Date(bValue);
      }
      
      if (sortDirection === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    return filtered;
  }, [transactions, sortField, sortDirection]);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const handleRefresh = () => {
    loadTransactions();
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency: 'BRL'
    }).format(value);
  };

  const formatDateTime = (dateString) => {
    // A data já vem formatada da API
    return dateString;
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-h1">Transações</h1>
        </div>
        <div className="h-96 bg-[var(--neutral-100)] animate-pulse rounded-lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-h1">Transações</h1>
          <p className="text-[var(--color-text-secondary)] mt-1">
            Lista e busca de transações processadas em tempo real
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="secondary" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Exportar
          </Button>
          <Button variant="secondary" size="sm" onClick={handleRefresh}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Atualizar
          </Button>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Filter className="h-5 w-5" />
            <span>Filtros</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-4">
            <FormField label="Buscar">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--color-text-secondary)]" />
                <Input
                  placeholder="ID, CPF, cidade..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
            </FormField>

            <FormField label="Status">
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="flex h-10 w-full rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-[var(--color-focus)]"
              >
                <option value="TODOS">Todos</option>
                <option value="APROVADA">Aprovada</option>
                <option value="REJEITADA">Rejeitada</option>
                <option value="PENDENTE">Pendente</option>
                <option value="EM_REVISAO">Em Revisão</option>
              </select>
            </FormField>

            <FormField label="Tipo">
              <select
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value)}
                className="flex h-10 w-full rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-[var(--color-focus)]"
              >
                <option value="TODOS">Todos</option>
                <option value="PIX">PIX</option>
                <option value="CREDITO">Crédito</option>
                <option value="DEBITO">Débito</option>
                <option value="TED">TED</option>
                <option value="DOC">DOC</option>
              </select>
            </FormField>

            <FormField label="Período">
              <Button variant="secondary" className="w-full justify-start">
                <Calendar className="h-4 w-4 mr-2" />
                Últimas 24h
              </Button>
            </FormField>
          </div>
        </CardContent>
      </Card>

      {/* Results Summary */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-[var(--color-text-secondary)]">
          Mostrando {filteredTransactions.length} de {totalTransactions} transações
        </p>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-[var(--color-text-secondary)]">Ordenar por:</span>
          <select
            value={`${sortField}-${sortDirection}`}
            onChange={(e) => {
              const [field, direction] = e.target.value.split('-');
              setSortField(field);
              setSortDirection(direction);
            }}
            className="text-sm border border-[var(--color-border)] rounded px-2 py-1"
          >
            <option value="timestamp-desc">Mais recentes</option>
            <option value="timestamp-asc">Mais antigas</option>
            <option value="valor-desc">Maior valor</option>
            <option value="valor-asc">Menor valor</option>
            <option value="risk_score-desc">Maior risco</option>
            <option value="risk_score-asc">Menor risco</option>
          </select>
        </div>
      </div>

      {/* Transactions Table */}
      <Card>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="border-b border-[var(--color-border)]">
                <tr className="bg-[var(--neutral-50)]">
                  <th className="text-left p-4 font-medium text-sm">ID</th>
                  <th className="text-left p-4 font-medium text-sm">Valor</th>
                  <th className="text-left p-4 font-medium text-sm">Tipo</th>
                  <th className="text-left p-4 font-medium text-sm">Canal</th>
                  <th className="text-left p-4 font-medium text-sm">Localização</th>
                  <th className="text-left p-4 font-medium text-sm">CPF</th>
                  <th className="text-left p-4 font-medium text-sm">Data/Hora</th>
                  <th className="text-left p-4 font-medium text-sm">Status</th>
                  <th className="text-left p-4 font-medium text-sm">Risco</th>
                  <th className="text-left p-4 font-medium text-sm">Ações</th>
                </tr>
              </thead>
              <tbody>
                {filteredTransactions.map((transaction) => (
                  <tr key={transaction.id} className="border-b border-[var(--color-border)] hover:bg-[var(--neutral-50)]">
                    <td className="p-4">
                      <code className="text-sm font-mono bg-[var(--neutral-100)] px-2 py-1 rounded">
                        {transaction.id}
                      </code>
                    </td>
                    <td className="p-4 font-mono">
                      {formatCurrency(transaction.valor)}
                    </td>
                    <td className="p-4">
                      <Badge variant="default" size="sm">
                        {transaction.tipo}
                      </Badge>
                    </td>
                    <td className="p-4 text-sm">
                      {transaction.canal.toUpperCase()}
                    </td>
                    <td className="p-4 text-sm">
                      {transaction.localizacao}
                    </td>
                    <td className="p-4 font-mono text-sm">
                      {transaction.cpf}
                    </td>
                    <td className="p-4 text-sm">
                      {transaction.data_hora}
                    </td>
                    <td className="p-4">
                      <TransactionStatusBadge status={transaction.status} size="sm" />
                    </td>
                    <td className="p-4">
                      <RiskScoreBadge score={transaction.fraud_score} size="sm" />
                    </td>
                    <td className="p-4">
                      <div className="flex items-center space-x-1">
                        <Button variant="ghost" size="sm" aria-label="Ver detalhes">
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button variant="ghost" size="sm" aria-label="Mais opções">
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {filteredTransactions.length === 0 && (
            <div className="text-center py-12">
              <p className="text-[var(--color-text-secondary)]">
                Nenhuma transação encontrada com os filtros aplicados.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-[var(--color-text-secondary)]">
            Página {currentPage} de {totalPages}
          </p>
          <div className="flex items-center space-x-2">
            <Button 
              variant="secondary" 
              size="sm" 
              disabled={currentPage === 1}
              onClick={() => setCurrentPage(currentPage - 1)}
            >
              Anterior
            </Button>
            <Button 
              variant="secondary" 
              size="sm" 
              disabled={currentPage === totalPages}
              onClick={() => setCurrentPage(currentPage + 1)}
            >
              Próxima
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

