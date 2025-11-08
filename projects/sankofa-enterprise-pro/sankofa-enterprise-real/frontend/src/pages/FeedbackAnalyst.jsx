import { useState, useEffect } from 'react';
import { 
  MessageSquare, 
  CheckCircle, 
  XCircle, 
  AlertTriangle,
  User,
  Calendar,
  TrendingUp,
  BarChart3,
  Download,
  Send,
  RefreshCw
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';
import { Badge } from '@/components/ui/Badge.jsx';
import { Button } from '@/components/ui/Button.jsx';

export function FeedbackAnalyst() {
  const [loading, setLoading] = useState(false);
  const [feedbacks, setFeedbacks] = useState([]);
  const [analytics, setAnalytics] = useState({});
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [showSubmitForm, setShowSubmitForm] = useState(false);
  const [submitLoading, setSubmitLoading] = useState(false);
  
  // Form state
  const [formData, setFormData] = useState({
    transaction_id: '',
    model_prediction: '',
    actual_label: '',
    analyst_id: '',
    comments: ''
  });

  const fetchFeedbacks = async (page = 1) => {
    try {
      setLoading(true);
      const response = await fetch(`/api/feedback/list?page=${page}&per_page=20`);
      const data = await response.json();
      
      if (response.ok) {
        setFeedbacks(data.feedbacks || []);
        setCurrentPage(data.page);
        setTotalPages(data.total_pages);
      } else {
        console.error('Erro ao buscar feedbacks:', data.error);
      }
    } catch (error) {
      console.error('Erro ao buscar feedbacks:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchAnalytics = async () => {
    try {
      const response = await fetch('/api/feedback/analytics');
      const data = await response.json();
      
      if (response.ok) {
        setAnalytics(data);
      } else {
        console.error('Erro ao buscar analytics:', data.error);
      }
    } catch (error) {
      console.error('Erro ao buscar analytics:', error);
    }
  };

  const handleSubmitFeedback = async (e) => {
    e.preventDefault();
    
    try {
      setSubmitLoading(true);
      
      const response = await fetch('/api/feedback/submit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...formData,
          model_prediction: parseInt(formData.model_prediction),
          actual_label: parseInt(formData.actual_label)
        }),
      });
      
      const result = await response.json();
      
      if (response.ok) {
        alert('Feedback submetido com sucesso!');
        setFormData({
          transaction_id: '',
          model_prediction: '',
          actual_label: '',
          analyst_id: '',
          comments: ''
        });
        setShowSubmitForm(false);
        fetchFeedbacks(currentPage);
        fetchAnalytics();
      } else {
        alert(`Erro ao submeter feedback: ${result.error}`);
      }
    } catch (error) {
      console.error('Erro ao submeter feedback:', error);
      alert('Erro interno do servidor');
    } finally {
      setSubmitLoading(false);
    }
  };

  const handleExportFeedbacks = async () => {
    try {
      const response = await fetch('/api/feedback/export');
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `feedback_export_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
      } else {
        alert('Erro ao exportar feedbacks');
      }
    } catch (error) {
      console.error('Erro ao exportar:', error);
      alert('Erro interno do servidor');
    }
  };

  useEffect(() => {
    fetchFeedbacks();
    fetchAnalytics();
  }, []);

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString('pt-BR');
  };

  const getPredictionLabel = (prediction) => {
    return prediction === 1 ? 'Fraude' : 'Legítima';
  };

  const getPredictionBadge = (prediction, actual) => {
    const isCorrect = prediction === actual;
    const label = getPredictionLabel(prediction);
    
    if (isCorrect) {
      return <Badge variant="success">{label} ✓</Badge>;
    } else {
      return <Badge variant="destructive">{label} ✗</Badge>;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-h1">Feedback de Analistas</h1>
          <p className="text-[var(--color-text-secondary)] mt-1">
            Sistema de coleta e análise de feedback humano para melhoria contínua do modelo
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="secondary" onClick={() => setShowSubmitForm(!showSubmitForm)}>
            <MessageSquare className="h-4 w-4 mr-2" />
            Novo Feedback
          </Button>
          <Button variant="secondary" onClick={handleExportFeedbacks}>
            <Download className="h-4 w-4 mr-2" />
            Exportar
          </Button>
          <Button variant="secondary" onClick={() => { fetchFeedbacks(currentPage); fetchAnalytics(); }}>
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Atualizar
          </Button>
        </div>
      </div>

      {/* Analytics Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total de Feedbacks</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{analytics.total_feedbacks || 0}</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Acurácia do Modelo</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {analytics.accuracy_metrics ? `${(analytics.accuracy_metrics.model_accuracy * 100).toFixed(1)}%` : '0%'}
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Precisão</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {analytics.accuracy_metrics ? `${(analytics.accuracy_metrics.precision * 100).toFixed(1)}%` : '0%'}
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Recall</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {analytics.accuracy_metrics ? `${(analytics.accuracy_metrics.recall * 100).toFixed(1)}%` : '0%'}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Submit Form */}
      {showSubmitForm && (
        <Card>
          <CardHeader>
            <CardTitle>Submeter Novo Feedback</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmitFeedback} className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="block text-sm font-medium mb-2">ID da Transação</label>
                  <input
                    type="text"
                    value={formData.transaction_id}
                    onChange={(e) => setFormData({...formData, transaction_id: e.target.value})}
                    className="w-full px-3 py-2 border border-[var(--color-border)] rounded-md"
                    required
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">ID do Analista</label>
                  <input
                    type="text"
                    value={formData.analyst_id}
                    onChange={(e) => setFormData({...formData, analyst_id: e.target.value})}
                    className="w-full px-3 py-2 border border-[var(--color-border)] rounded-md"
                    required
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Predição do Modelo</label>
                  <select
                    value={formData.model_prediction}
                    onChange={(e) => setFormData({...formData, model_prediction: e.target.value})}
                    className="w-full px-3 py-2 border border-[var(--color-border)] rounded-md"
                    required
                  >
                    <option value="">Selecione...</option>
                    <option value="0">Legítima (0)</option>
                    <option value="1">Fraude (1)</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Classificação Real</label>
                  <select
                    value={formData.actual_label}
                    onChange={(e) => setFormData({...formData, actual_label: e.target.value})}
                    className="w-full px-3 py-2 border border-[var(--color-border)] rounded-md"
                    required
                  >
                    <option value="">Selecione...</option>
                    <option value="0">Legítima (0)</option>
                    <option value="1">Fraude (1)</option>
                  </select>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">Comentários (Opcional)</label>
                <textarea
                  value={formData.comments}
                  onChange={(e) => setFormData({...formData, comments: e.target.value})}
                  className="w-full px-3 py-2 border border-[var(--color-border)] rounded-md"
                  rows="3"
                  placeholder="Observações sobre a transação..."
                />
              </div>
              
              <div className="flex justify-end space-x-2">
                <Button type="button" variant="secondary" onClick={() => setShowSubmitForm(false)}>
                  Cancelar
                </Button>
                <Button type="submit" disabled={submitLoading}>
                  <Send className={`h-4 w-4 mr-2 ${submitLoading ? 'animate-spin' : ''}`} />
                  {submitLoading ? 'Enviando...' : 'Enviar Feedback'}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      )}

      {/* Feedbacks List */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <BarChart3 className="h-5 w-5" />
            <span>Histórico de Feedbacks</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="h-6 w-6 animate-spin mr-2" />
              <span>Carregando feedbacks...</span>
            </div>
          ) : feedbacks.length === 0 ? (
            <div className="text-center py-8">
              <MessageSquare className="h-12 w-12 mx-auto text-[var(--color-text-secondary)] mb-4" />
              <p className="text-[var(--color-text-secondary)]">Nenhum feedback registrado ainda</p>
            </div>
          ) : (
            <div className="space-y-4">
              {feedbacks.map((feedback, index) => (
                <div key={index} className="border border-[var(--color-border)] rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <User className="h-5 w-5 text-[var(--color-text-secondary)]" />
                      <div>
                        <p className="font-medium">{feedback.analyst_id}</p>
                        <p className="text-sm text-[var(--color-text-secondary)]">
                          Transação: {feedback.transaction_id}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Calendar className="h-4 w-4 text-[var(--color-text-secondary)]" />
                      <span className="text-sm text-[var(--color-text-secondary)]">
                        {formatTimestamp(feedback.feedback_timestamp)}
                      </span>
                    </div>
                  </div>
                  
                  <div className="grid gap-4 md:grid-cols-3 mb-3">
                    <div>
                      <p className="text-sm font-medium mb-1">Predição do Modelo</p>
                      {getPredictionBadge(feedback.model_prediction, feedback.actual_label)}
                    </div>
                    <div>
                      <p className="text-sm font-medium mb-1">Classificação Real</p>
                      <Badge variant={feedback.actual_label === 1 ? 'destructive' : 'success'}>
                        {getPredictionLabel(feedback.actual_label)}
                      </Badge>
                    </div>
                    <div>
                      <p className="text-sm font-medium mb-1">Status</p>
                      {feedback.model_prediction === feedback.actual_label ? (
                        <Badge variant="success">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Correto
                        </Badge>
                      ) : (
                        <Badge variant="destructive">
                          <XCircle className="h-3 w-3 mr-1" />
                          Incorreto
                        </Badge>
                      )}
                    </div>
                  </div>
                  
                  {feedback.comments && (
                    <div className="bg-[var(--neutral-50)] p-3 rounded-md">
                      <p className="text-sm font-medium mb-1">Comentários:</p>
                      <p className="text-sm text-[var(--color-text-secondary)]">{feedback.comments}</p>
                    </div>
                  )}
                </div>
              ))}
              
              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-center space-x-2 pt-4">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => fetchFeedbacks(currentPage - 1)}
                    disabled={currentPage <= 1}
                  >
                    Anterior
                  </Button>
                  <span className="text-sm text-[var(--color-text-secondary)]">
                    Página {currentPage} de {totalPages}
                  </span>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => fetchFeedbacks(currentPage + 1)}
                    disabled={currentPage >= totalPages}
                  >
                    Próxima
                  </Button>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Confusion Matrix */}
      {analytics.confusion_matrix && (
        <Card>
          <CardHeader>
            <CardTitle>Matriz de Confusão</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                  <span className="font-medium">Verdadeiros Positivos</span>
                  <Badge variant="success">{analytics.confusion_matrix.true_positives}</Badge>
                </div>
                <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                  <span className="font-medium">Verdadeiros Negativos</span>
                  <Badge variant="success">{analytics.confusion_matrix.true_negatives}</Badge>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                  <span className="font-medium">Falsos Positivos</span>
                  <Badge variant="destructive">{analytics.confusion_matrix.false_positives}</Badge>
                </div>
                <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                  <span className="font-medium">Falsos Negativos</span>
                  <Badge variant="destructive">{analytics.confusion_matrix.false_negatives}</Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

