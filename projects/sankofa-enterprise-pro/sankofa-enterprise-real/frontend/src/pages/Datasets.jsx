import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { 
  Search, 
  Database, 
  TrendingUp, 
  BarChart3, 
  Filter,
  Info,
  CheckCircle,
  AlertTriangle,
  Globe,
  Lock,
  Activity
} from 'lucide-react';

export function Datasets() {
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('');
  const [activeTab, setActiveTab] = useState('overview');
  
  // Estados dos dados
  const [overview, setOverview] = useState({});
  const [datasetsByUsage, setDatasetsByUsage] = useState([]);
  const [categoriesSummary, setCategoriesSummary] = useState({});
  const [searchResults, setSearchResults] = useState([]);

  useEffect(() => {
    loadDatasetsData();
    // Atualiza dados a cada 30 segundos
    const interval = setInterval(loadDatasetsData, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadDatasetsData = async () => {
    try {
      setLoading(true);
      
      // Carrega visão geral
      const overviewResponse = await fetch('/api/datasets');
      const overviewData = await overviewResponse.json();
      setOverview(overviewData);
      
      // Carrega datasets por uso
      const usageResponse = await fetch('/api/datasets');
      const usageData = await usageResponse.json();
      setDatasetsByUsage(usageData.datasets || []);
      
      // Carrega resumo por categorias
      const categoriesResponse = await fetch('/api/datasets');
      const categoriesData = await categoriesResponse.json();
      setCategoriesSummary(categoriesData);
      
    } catch (error) {
      console.error('Erro ao carregar dados dos datasets:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }
    
    try {
      const params = new URLSearchParams({
        query: searchQuery,
        ...(selectedCategory && { category: selectedCategory })
      });
      
      const response = await fetch(`/api/datasets/search?${params}`);
      const results = await response.json();
      setSearchResults(results);
    } catch (error) {
      console.error('Erro na busca:', error);
    }
  };

  const formatNumber = (num) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num?.toString() || '0';
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800';
      case 'updating': return 'bg-yellow-100 text-yellow-800';
      case 'inactive': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getRealnessIcon = (realness) => {
    if (realness?.includes('Real')) return <Globe className="h-4 w-4 text-green-600" />;
    if (realness?.includes('Sintético')) return <Database className="h-4 w-4 text-yellow-600" />;
    return <Database className="h-4 w-4 text-blue-600" />;
  };

  const tabs = [
    { id: 'overview', label: 'Visão Geral', icon: BarChart3 },
    { id: 'ranking', label: 'Ranking de Uso', icon: TrendingUp },
    { id: 'search', label: 'Busca Avançada', icon: Search },
    { id: 'categories', label: 'Categorias', icon: Filter }
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-2">
          <Database className="h-8 w-8 text-blue-600" />
          Catálogo de Datasets
        </h1>
        <p className="text-gray-600">
          Gestão completa de todos os {overview.total_datasets || 0} datasets da solução em tempo real
        </p>
      </div>

      {/* Cards de Resumo */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-6 bg-gradient-to-br from-blue-500 to-blue-600 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-4xl font-bold">{overview.total_datasets || 0}</p>
              <p className="text-blue-100">Total de Datasets</p>
            </div>
            <Database className="h-10 w-10 text-blue-200" />
          </div>
        </Card>

        <Card className="p-6 bg-gradient-to-br from-purple-500 to-purple-600 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-4xl font-bold">{formatNumber(overview.total_records || 0)}</p>
              <p className="text-purple-100">Total de Registros</p>
            </div>
            <Activity className="h-10 w-10 text-purple-200" />
          </div>
        </Card>

        <Card className="p-6 bg-gradient-to-br from-red-500 to-red-600 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-4xl font-bold">{formatNumber(overview.total_fraud_records || 0)}</p>
              <p className="text-red-100">Registros de Fraude</p>
            </div>
            <AlertTriangle className="h-10 w-10 text-red-200" />
          </div>
        </Card>

        <Card className="p-6 bg-gradient-to-br from-green-500 to-green-600 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-4xl font-bold">{overview.active_datasets || 0}</p>
              <p className="text-green-100">Datasets Ativos</p>
            </div>
            <CheckCircle className="h-10 w-10 text-green-200" />
          </div>
        </Card>
      </div>

      {/* Tabs */}
      <Card>
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {tab.label}
                </button>
              );
            })}
          </nav>
        </div>

        <div className="p-6">
          {/* Tab: Visão Geral */}
          {activeTab === 'overview' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Métricas de Qualidade */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Métricas de Qualidade</h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Taxa Geral de Fraude</span>
                        <span>{((overview.overall_fraud_rate || 0) * 100).toFixed(3)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-red-500 h-2 rounded-full" 
                          style={{ width: `${(overview.overall_fraud_rate || 0) * 100 * 20}%` }}
                        ></div>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Datasets com Rótulos de Fraude</span>
                        <span>{datasetsByUsage.filter(d => d.has_fraud_labels).length}</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full" 
                          style={{ width: `${(datasetsByUsage.filter(d => d.has_fraud_labels).length / datasetsByUsage.length) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Qualidade Média</span>
                        <span>{(datasetsByUsage.reduce((acc, d) => acc + (d.quality_score || 0), 0) / datasetsByUsage.length * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-green-500 h-2 rounded-full" 
                          style={{ width: `${datasetsByUsage.reduce((acc, d) => acc + (d.quality_score || 0), 0) / datasetsByUsage.length * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </Card>

                {/* Status dos Datasets */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Status em Tempo Real</h3>
                  <div className="grid grid-cols-2 gap-3">
                    {datasetsByUsage.slice(0, 8).map((dataset) => (
                      <div key={dataset.id} className="border rounded-lg p-3 hover:bg-gray-50 cursor-pointer">
                        <div className="flex items-center gap-2 mb-2">
                          {getRealnessIcon(dataset.realness)}
                          <span className="font-medium text-sm truncate">
                            {dataset.name?.substring(0, 15)}...
                          </span>
                        </div>
                        <Badge className={getStatusColor(dataset.status)}>
                          {dataset.status}
                        </Badge>
                        <div className="text-xs text-gray-500 mt-1">
                          <div>Uso: {dataset.usage_count || 0}x</div>
                          <div>Qualidade: {((dataset.quality_score || 0) * 100).toFixed(0)}%</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>
            </div>
          )}

          {/* Tab: Ranking de Uso */}
          {activeTab === 'ranking' && (
            <div className="space-y-6">
              <h3 className="text-lg font-semibold">Datasets Mais Utilizados</h3>
              <div className="space-y-3">
                {datasetsByUsage.map((dataset, index) => (
                  <Card key={dataset.id} className="p-4 hover:bg-gray-50 cursor-pointer">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <div className="flex items-center justify-center w-8 h-8 bg-blue-100 text-blue-600 rounded-full font-bold">
                          {index + 1}
                        </div>
                        {getRealnessIcon(dataset.realness)}
                        <div>
                          <h4 className="font-medium">{dataset.name}</h4>
                          <p className="text-sm text-gray-500">{dataset.category}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold">{dataset.usage_count || 0} usos</div>
                        <div className="text-sm text-gray-500">
                          Contribuição: {((dataset.avg_contribution || 0) * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {/* Tab: Busca Avançada */}
          {activeTab === 'search' && (
            <div className="space-y-6">
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Busca Avançada de Datasets</h3>
                <div className="flex gap-4">
                  <div className="flex-1">
                    <Input
                      placeholder="Buscar datasets..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                    />
                  </div>
                  <select
                    value={selectedCategory}
                    onChange={(e) => setSelectedCategory(e.target.value)}
                    className="px-3 py-2 border border-gray-300 rounded-md"
                  >
                    <option value="">Todas as categorias</option>
                    {Object.keys(categoriesSummary).map((category) => (
                      <option key={category} value={category}>
                        {category}
                      </option>
                    ))}
                  </select>
                  <Button onClick={handleSearch}>
                    <Search className="h-4 w-4 mr-2" />
                    Buscar
                  </Button>
                </div>
              </Card>

              {/* Resultados da Busca */}
              <div>
                <h3 className="text-lg font-semibold mb-4">
                  Resultados da Busca ({searchResults.length})
                </h3>
                {searchResults.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {searchResults.map((dataset) => (
                      <Card key={dataset.id} className="p-4 hover:shadow-lg cursor-pointer">
                        <div className="flex items-center gap-2 mb-2">
                          {getRealnessIcon(dataset.realness)}
                          <h4 className="font-medium">{dataset.name}</h4>
                        </div>
                        <p className="text-sm text-gray-600 mb-3">
                          {dataset.description?.substring(0, 100)}...
                        </p>
                        <div className="flex gap-2 mb-2">
                          <Badge variant="outline">{dataset.category}</Badge>
                          <Badge variant="outline">{dataset.subcategory}</Badge>
                        </div>
                        <div className="text-sm text-gray-500">
                          <div>Registros: {formatNumber(dataset.record_count)}</div>
                          {dataset.has_fraud_labels && (
                            <div className="text-red-600">
                              Fraudes: {formatNumber(dataset.fraud_count)} ({(dataset.fraud_rate * 100).toFixed(2)}%)
                            </div>
                          )}
                        </div>
                      </Card>
                    ))}
                  </div>
                ) : searchQuery ? (
                  <p className="text-center text-gray-500 py-8">
                    Nenhum dataset encontrado para "{searchQuery}"
                  </p>
                ) : (
                  <p className="text-center text-gray-500 py-8">
                    Digite um termo para buscar datasets
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Tab: Categorias */}
          {activeTab === 'categories' && (
            <div className="space-y-6">
              <h3 className="text-lg font-semibold">Categorias de Datasets</h3>
              <div className="space-y-4">
                {Object.entries(categoriesSummary).map(([categoryName, categoryData]) => (
                  <Card key={categoryName} className="p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="text-lg font-semibold flex items-center gap-2">
                        <Filter className="h-5 w-5 text-blue-600" />
                        {categoryName}
                      </h4>
                      <div className="flex gap-2">
                        <Badge>{categoryData.count} datasets</Badge>
                        <Badge variant="outline">{formatNumber(categoryData.total_records)} registros</Badge>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <p className="text-sm text-gray-600 mb-2">
                          Esta categoria contém {categoryData.count} datasets com um total de {formatNumber(categoryData.total_records)} registros.
                          {categoryData.fraud_records > 0 && (
                            ` Inclui ${formatNumber(categoryData.fraud_records)} registros de fraude (${(categoryData.fraud_rate * 100).toFixed(2)}% da categoria).`
                          )}
                        </p>
                      </div>
                      
                      <div className="space-y-2">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-blue-600">{categoryData.count}</div>
                          <div className="text-sm text-gray-500">Total de Datasets</div>
                        </div>
                      </div>
                      
                      <div className="space-y-2">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-green-600">
                            {(categoryData.avg_quality * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-gray-500">Qualidade Média</div>
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>
      </Card>

      {loading && (
        <div className="fixed top-0 left-0 right-0 h-1 bg-blue-500 animate-pulse z-50"></div>
      )}
    </div>
  );
}

