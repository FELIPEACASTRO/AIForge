import { useState, useEffect } from 'react';
import { 
  Settings as SettingsIcon, 
  Save, 
  RotateCcw, 
  AlertTriangle, 
  CheckCircle,
  Database,
  Shield,
  Bell,
  Cpu,
  Globe,
  RefreshCw
} from 'lucide-react';
import { Button } from '@/components/ui/Button.jsx';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';
import { Badge } from '@/components/ui/Badge.jsx';
import { Input } from '@/components/ui/Input.jsx';
import { Switch } from '@/components/ui/Switch.jsx';

export function Settings() {
  const [settings, setSettings] = useState({});
  const [activeTab, setActiveTab] = useState('system');
  const [hasChanges, setHasChanges] = useState(false);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(true);
  const [lastSaved, setLastSaved] = useState(null);

  // Carregar configurações do backend
  useEffect(() => {
    const loadSettings = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/settings');
        if (response.ok) {
          const data = await response.json();
          setSettings(data.settings || {});
        } else {
          console.error('Erro ao carregar configurações');
        }
      } catch (error) {
        console.error('Erro ao conectar com backend:', error);
      } finally {
        setLoading(false);
      }
    };

    loadSettings();
  }, []);

  // Detectar mudanças
  useEffect(() => {
    setHasChanges(Object.keys(settings).length > 0);
  }, [settings]);

  const tabs = [
    { id: 'system', label: 'Sistema', icon: Cpu },
    { id: 'database', label: 'Banco de Dados', icon: Database },
    { id: 'security', label: 'Segurança', icon: Shield },
    { id: 'notifications', label: 'Notificações', icon: Bell },
    { id: 'ai', label: 'IA & ML', icon: Cpu },
    { id: 'api', label: 'API', icon: Globe }
  ];

  const updateSetting = (category, key, value) => {
    setSettings(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [key]: value
      }
    }));
  };

  const saveSettings = async () => {
    setSaving(true);
    try {
      const response = await fetch('/api/settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ settings })
      });

      if (response.ok) {
        const result = await response.json();
        setLastSaved(new Date());
        setHasChanges(false);
        console.log('✅ Configurações salvas com sucesso');
      } else {
        throw new Error('Erro ao salvar configurações');
      }
    } catch (error) {
      console.error('Erro ao salvar configurações:', error);
      alert('Erro ao salvar configurações: ' + error.message);
    } finally {
      setSaving(false);
    }
  };

  const resetSettings = async () => {
    setSaving(true);
    try {
      const response = await fetch('/api/settings/reset', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        const result = await response.json();
        setSettings(result.settings || {});
        setHasChanges(false);
        setLastSaved(new Date());
        console.log('✅ Configurações resetadas com sucesso');
      } else {
        throw new Error('Erro ao resetar configurações');
      }
    } catch (error) {
      console.error('Erro ao resetar configurações:', error);
      alert('Erro ao resetar configurações: ' + error.message);
    } finally {
      setSaving(false);
    }
  };

  const renderTabContent = () => {
    if (loading) {
      return (
        <div className="flex items-center justify-center h-64">
          <div className="flex items-center space-x-2">
            <RefreshCw className="h-6 w-6 animate-spin" />
            <span>Carregando configurações...</span>
          </div>
        </div>
      );
    }

    const systemSettings = settings.system || {};
    const databaseSettings = settings.database || {};
    const securitySettings = settings.security || {};
    const notificationSettings = settings.notifications || {};
    const aiSettings = settings.ai || {};
    const apiSettings = settings.api || {};

    switch (activeTab) {
      case 'system':
        return (
          <div className="space-y-6">
            <Card className="p-6">
              <h4 className="font-medium text-[var(--color-text-primary)] mb-4">
                Configurações do Sistema
              </h4>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Nome do Sistema</label>
                  <Input
                    value={systemSettings.systemName || ''}
                    onChange={(e) => updateSetting('system', 'systemName', e.target.value)}
                    placeholder="Nome do sistema"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Versão</label>
                  <Input
                    value={systemSettings.version || ''}
                    onChange={(e) => updateSetting('system', 'version', e.target.value)}
                    placeholder="Versão do sistema"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Ambiente</label>
                  <select
                    value={systemSettings.environment || 'production'}
                    onChange={(e) => updateSetting('system', 'environment', e.target.value)}
                    className="w-full p-2 border rounded-md"
                  >
                    <option value="development">Desenvolvimento</option>
                    <option value="staging">Homologação</option>
                    <option value="production">Produção</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Timezone</label>
                  <Input
                    value={systemSettings.timezone || ''}
                    onChange={(e) => updateSetting('system', 'timezone', e.target.value)}
                    placeholder="America/Sao_Paulo"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Timeout de Sessão (minutos)</label>
                  <Input
                    type="number"
                    value={systemSettings.sessionTimeout || 30}
                    onChange={(e) => updateSetting('system', 'sessionTimeout', parseInt(e.target.value))}
                    min="5"
                    max="480"
                  />
                </div>
              </div>
            </Card>
          </div>
        );

      case 'database':
        return (
          <div className="space-y-6">
            <Card className="p-6">
              <h4 className="font-medium text-[var(--color-text-primary)] mb-4">
                Configurações do Banco de Dados
              </h4>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Host</label>
                  <Input
                    value={databaseSettings.host || ''}
                    onChange={(e) => updateSetting('database', 'host', e.target.value)}
                    placeholder="localhost"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Porta</label>
                  <Input
                    type="number"
                    value={databaseSettings.port || 5432}
                    onChange={(e) => updateSetting('database', 'port', parseInt(e.target.value))}
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Nome do Banco</label>
                  <Input
                    value={databaseSettings.database || ''}
                    onChange={(e) => updateSetting('database', 'database', e.target.value)}
                    placeholder="sankofa_fraud"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Pool de Conexões</label>
                  <Input
                    type="number"
                    value={databaseSettings.connectionPool || 20}
                    onChange={(e) => updateSetting('database', 'connectionPool', parseInt(e.target.value))}
                    min="5"
                    max="100"
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Backup Automático</label>
                  <Switch
                    checked={databaseSettings.backupEnabled || false}
                    onCheckedChange={(checked) => updateSetting('database', 'backupEnabled', checked)}
                  />
                </div>
              </div>
            </Card>
          </div>
        );

      case 'security':
        return (
          <div className="space-y-6">
            <Card className="p-6">
              <h4 className="font-medium text-[var(--color-text-primary)] mb-4">
                Configurações de Segurança
              </h4>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Autenticação de Dois Fatores</label>
                  <Switch
                    checked={securitySettings.twoFactorEnabled || false}
                    onCheckedChange={(checked) => updateSetting('security', 'twoFactorEnabled', checked)}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Complexidade de Senha</label>
                  <Switch
                    checked={securitySettings.passwordComplexity || false}
                    onCheckedChange={(checked) => updateSetting('security', 'passwordComplexity', checked)}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Criptografia de Sessão</label>
                  <Switch
                    checked={securitySettings.sessionEncryption || false}
                    onCheckedChange={(checked) => updateSetting('security', 'sessionEncryption', checked)}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Log de Auditoria</label>
                  <Switch
                    checked={securitySettings.auditLogging || false}
                    onCheckedChange={(checked) => updateSetting('security', 'auditLogging', checked)}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">SSL Habilitado</label>
                  <Switch
                    checked={securitySettings.sslEnabled || false}
                    onCheckedChange={(checked) => updateSetting('security', 'sslEnabled', checked)}
                  />
                </div>
              </div>
            </Card>
          </div>
        );

      case 'notifications':
        return (
          <div className="space-y-6">
            <Card className="p-6">
              <h4 className="font-medium text-[var(--color-text-primary)] mb-4">
                Configurações de Notificações
              </h4>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Email Habilitado</label>
                  <Switch
                    checked={notificationSettings.emailEnabled || false}
                    onCheckedChange={(checked) => updateSetting('notifications', 'emailEnabled', checked)}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">SMS Habilitado</label>
                  <Switch
                    checked={notificationSettings.smsEnabled || false}
                    onCheckedChange={(checked) => updateSetting('notifications', 'smsEnabled', checked)}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Slack Habilitado</label>
                  <Switch
                    checked={notificationSettings.slackEnabled || false}
                    onCheckedChange={(checked) => updateSetting('notifications', 'slackEnabled', checked)}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Webhook Habilitado</label>
                  <Switch
                    checked={notificationSettings.webhookEnabled || false}
                    onCheckedChange={(checked) => updateSetting('notifications', 'webhookEnabled', checked)}
                  />
                </div>
              </div>
            </Card>
          </div>
        );

      case 'ai':
        return (
          <div className="space-y-6">
            <Card className="p-6">
              <h4 className="font-medium text-[var(--color-text-primary)] mb-4">
                Configurações de IA e Machine Learning
              </h4>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Auto-Learning Habilitado</label>
                  <Switch
                    checked={aiSettings.autoLearningEnabled || false}
                    onCheckedChange={(checked) => updateSetting('ai', 'autoLearningEnabled', checked)}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Detecção de Drift</label>
                  <Switch
                    checked={aiSettings.driftDetectionEnabled || false}
                    onCheckedChange={(checked) => updateSetting('ai', 'driftDetectionEnabled', checked)}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Feedback em Tempo Real</label>
                  <Switch
                    checked={aiSettings.realTimeFeedback || false}
                    onCheckedChange={(checked) => updateSetting('ai', 'realTimeFeedback', checked)}
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Batch Size</label>
                  <Input
                    type="number"
                    value={aiSettings.batchSize || 1000}
                    onChange={(e) => updateSetting('ai', 'batchSize', parseInt(e.target.value))}
                    min="100"
                    max="10000"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Taxa de Aprendizado</label>
                  <Input
                    type="number"
                    step="0.001"
                    value={aiSettings.learningRate || 0.001}
                    onChange={(e) => updateSetting('ai', 'learningRate', parseFloat(e.target.value))}
                    min="0.0001"
                    max="0.1"
                  />
                </div>
              </div>
            </Card>
          </div>
        );

      case 'api':
        return (
          <div className="space-y-6">
            <Card className="p-6">
              <h4 className="font-medium text-[var(--color-text-primary)] mb-4">
                Configurações de API
              </h4>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Rate Limiting Habilitado</label>
                  <Switch
                    checked={apiSettings.rateLimitEnabled || false}
                    onCheckedChange={(checked) => updateSetting('api', 'rateLimitEnabled', checked)}
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Requisições por Minuto</label>
                  <Input
                    type="number"
                    value={apiSettings.requestsPerMinute || 1000}
                    onChange={(e) => updateSetting('api', 'requestsPerMinute', parseInt(e.target.value))}
                    min="100"
                    max="10000"
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">API Key Obrigatória</label>
                  <Switch
                    checked={apiSettings.apiKeyRequired || false}
                    onCheckedChange={(checked) => updateSetting('api', 'apiKeyRequired', checked)}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">CORS Habilitado</label>
                  <Switch
                    checked={apiSettings.corsEnabled || false}
                    onCheckedChange={(checked) => updateSetting('api', 'corsEnabled', checked)}
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Timeout (segundos)</label>
                  <Input
                    type="number"
                    value={apiSettings.timeoutSeconds || 30}
                    onChange={(e) => updateSetting('api', 'timeoutSeconds', parseInt(e.target.value))}
                    min="5"
                    max="300"
                  />
                </div>
              </div>
            </Card>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-h1 flex items-center space-x-2">
            <SettingsIcon className="h-8 w-8" />
            <span>Configurações</span>
          </h1>
          <p className="text-[var(--color-text-secondary)] mt-1">
            Configurações do sistema e parâmetros operacionais
          </p>
        </div>
        <div className="flex items-center space-x-2">
          {lastSaved && (
            <Badge variant="success" className="flex items-center space-x-1">
              <CheckCircle className="h-3 w-3" />
              <span>Salvo às {lastSaved.toLocaleTimeString('pt-BR')}</span>
            </Badge>
          )}
          {hasChanges && (
            <Badge variant="warning" className="flex items-center space-x-1">
              <AlertTriangle className="h-3 w-3" />
              <span>Mudanças pendentes</span>
            </Badge>
          )}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center space-x-2">
        <Button
          onClick={saveSettings}
          disabled={!hasChanges || saving}
          loading={saving}
          className="flex items-center space-x-2"
        >
          <Save className="h-4 w-4" />
          <span>Salvar Configurações</span>
        </Button>
        
        <Button
          variant="secondary"
          onClick={resetSettings}
          disabled={saving}
          className="flex items-center space-x-2"
        >
          <RotateCcw className="h-4 w-4" />
          <span>Resetar Padrões</span>
        </Button>
      </div>

      {/* Tabs */}
      <div className="border-b border-[var(--color-border)]">
        <nav className="flex space-x-8">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-[var(--color-brand)] text-[var(--color-brand)]'
                    : 'border-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      {renderTabContent()}
    </div>
  );
}

