import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/Button.jsx';
import { 
  Eye, 
  CheckCircle, 
  XCircle, 
  Clock, 
  AlertTriangle,
  User,
  Calendar,
  DollarSign,
  MapPin,
  CreditCard,
  Filter,
  Search,
  RefreshCw
} from 'lucide-react';

const ManualReview = () => {
  const [reviews, setReviews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedReview, setSelectedReview] = useState(null);
  const [showDialog, setShowDialog] = useState(false);

  useEffect(() => {
    loadReviews();
  }, []);

  const loadReviews = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/manual-review');
      if (response.ok) {
        const data = await response.json();
        setReviews(data.reviews || []);
      }
    } catch (error) {
      console.error('Erro ao carregar revisões:', error);
      // Mock data para demonstração
      setReviews([
        {
          transaction_id: 'TXN_001',
          valor: 15000,
          cpf: '123.456.789-01',
          tipoTransacao: 'PIX',
          canal: 'mobile',
          localizacao: 'São Paulo',
          fraud_score: 0.85,
          risk_level: 'ALTO',
          status: 'PENDENTE',
          flagged_at: new Date().toISOString(),
          explanation: 'Transação de alto valor em horário atípico'
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleComplete = async (transactionId, decision, notes) => {
    try {
      const response = await fetch('/api/manual-review/complete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          transaction_id: transactionId,
          decision,
          analyst_notes: notes,
          confidence: 0.95
        }),
      });

      if (response.ok) {
        await loadReviews();
        setShowDialog(false);
        setSelectedReview(null);
      }
    } catch (error) {
      console.error('Erro ao completar revisão:', error);
    }
  };

  const stats = {
    total: reviews.length,
    pending: reviews.filter(r => r.status === 'PENDENTE').length,
    completed: reviews.filter(r => r.status === 'APROVADA' || r.status === 'REJEITADA').length,
    expired: reviews.filter(r => r.status === 'EXPIRADA').length
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Revisão Manual (Human-in-the-Loop)
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Sistema de revisão manual para transações flagadas pelo auto-learning
          </p>
        </div>
        <Button onClick={loadReviews} variant="outline">
          <RefreshCw className="w-4 h-4 mr-2" />
          Atualizar
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{stats.total}</p>
            </div>
            <Eye className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Pendentes</p>
              <p className="text-2xl font-bold text-yellow-600">{stats.pending}</p>
            </div>
            <Clock className="w-8 h-8 text-yellow-600" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Completadas</p>
              <p className="text-2xl font-bold text-green-600">{stats.completed}</p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-600" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Expiradas</p>
              <p className="text-2xl font-bold text-red-600">{stats.expired}</p>
            </div>
            <XCircle className="w-8 h-8 text-red-600" />
          </div>
        </div>
      </div>

      {/* Reviews Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border overflow-hidden">
        <div className="p-4 border-b">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Transações para Revisão ({reviews.length})
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">ID</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Valor</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">CPF</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Risco</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Status</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Ações</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {loading ? (
                <tr>
                  <td colSpan="6" className="px-4 py-8 text-center text-gray-500">
                    Carregando revisões...
                  </td>
                </tr>
              ) : reviews.length === 0 ? (
                <tr>
                  <td colSpan="6" className="px-4 py-8 text-center text-gray-500">
                    Nenhuma transação pendente de revisão manual.
                  </td>
                </tr>
              ) : (
                reviews.map((review) => (
                  <tr key={review.transaction_id} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                    <td className="px-4 py-4 text-sm font-medium text-gray-900 dark:text-white">
                      {review.transaction_id}
                    </td>
                    <td className="px-4 py-4 text-sm text-gray-900 dark:text-white">
                      R$ {review.valor?.toLocaleString('pt-BR', { minimumFractionDigits: 2 })}
                    </td>
                    <td className="px-4 py-4 text-sm text-gray-900 dark:text-white">
                      {review.cpf}
                    </td>
                    <td className="px-4 py-4">
                      <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                        review.risk_level === 'ALTO' ? 'bg-red-100 text-red-800' :
                        review.risk_level === 'MÉDIO' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-green-100 text-green-800'
                      }`}>
                        {review.risk_level}
                      </span>
                    </td>
                    <td className="px-4 py-4">
                      <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                        review.status === 'PENDENTE' ? 'bg-yellow-100 text-yellow-800' :
                        review.status === 'APROVADA' ? 'bg-green-100 text-green-800' :
                        'bg-red-100 text-red-800'
                      }`}>
                        {review.status}
                      </span>
                    </td>
                    <td className="px-4 py-4">
                      {review.status === 'PENDENTE' && (
                        <Button
                          onClick={() => {
                            setSelectedReview(review);
                            setShowDialog(true);
                          }}
                          size="sm"
                          className="bg-blue-600 hover:bg-blue-700 text-white"
                        >
                          <Eye className="w-4 h-4 mr-1" />
                          Revisar
                        </Button>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Review Dialog */}
      {showDialog && selectedReview && (
        <ReviewDialog
          review={selectedReview}
          onComplete={handleComplete}
          onClose={() => {
            setShowDialog(false);
            setSelectedReview(null);
          }}
        />
      )}
    </div>
  );
};

const ReviewDialog = ({ review, onComplete, onClose }) => {
  const [decision, setDecision] = useState('');
  const [notes, setNotes] = useState('');

  const handleSubmit = () => {
    if (decision && notes.trim()) {
      onComplete(review.transaction_id, decision, notes);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-2xl">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
          Revisão Manual - {review.transaction_id}
        </h2>
        
        <div className="space-y-4">
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
            <h3 className="font-semibold mb-2">Detalhes da Transação</h3>
            <div className="space-y-2 text-sm">
              <div>Valor: R$ {review.valor?.toLocaleString('pt-BR', { minimumFractionDigits: 2 })}</div>
              <div>CPF: {review.cpf}</div>
              <div>Tipo: {review.tipoTransacao}</div>
              <div>Canal: {review.canal}</div>
              <div>Score de Fraude: {(review.fraud_score * 100).toFixed(1)}%</div>
              <div>Nível de Risco: {review.risk_level}</div>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Decisão</label>
            <div className="space-y-2">
              <label className="flex items-center">
                <input
                  type="radio"
                  name="decision"
                  value="APROVADA"
                  checked={decision === 'APROVADA'}
                  onChange={(e) => setDecision(e.target.value)}
                  className="mr-2"
                />
                Aprovar Transação
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  name="decision"
                  value="REJEITADA"
                  checked={decision === 'REJEITADA'}
                  onChange={(e) => setDecision(e.target.value)}
                  className="mr-2"
                />
                Rejeitar Transação
              </label>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Observações</label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className="w-full px-3 py-2 border rounded-md"
              rows="3"
              placeholder="Descreva os motivos da sua decisão..."
            />
          </div>
        </div>

        <div className="flex justify-end space-x-3 mt-6">
          <Button variant="outline" onClick={onClose}>
            Cancelar
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!decision || !notes.trim()}
            className="bg-blue-600 hover:bg-blue-700 text-white"
          >
            Confirmar Decisão
          </Button>
        </div>
      </div>
    </div>
  );
};

export default ManualReview;

