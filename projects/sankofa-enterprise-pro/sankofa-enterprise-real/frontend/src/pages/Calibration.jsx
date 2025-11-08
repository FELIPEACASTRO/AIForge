import { useState, useEffect } from 'react';
import { 
  Settings, 
  Save, 
  RotateCcw, 
  AlertTriangle, 
  CheckCircle,
  Activity,
  Zap,
  Brain,
  Shield,
  Clock,
  TrendingUp,
  Eye,
  History,
  Server,
  Lock,
  Bell,
  BarChart3,
  Database,
  Wifi,
  RefreshCw
} from 'lucide-react';
import { Button } from '@/components/ui/Button.jsx';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';
import { Badge } from '@/components/ui/Badge.jsx';
import { SliderControl } from '@/components/ui/Slider.jsx';
import { SwitchControl } from '@/components/ui/Switch.jsx';
import { SimpleLineChart } from '@/components/charts/SimpleChart';

// Configurações iniciais dos algoritmos
const initialConfig = {
  // Tier 1 - Velocistas (< 1ms)
  ruleBasedEngine: {
    enabled: true,
    threshold: 0.8,
    weight: 0.15,
    maxAmount: 50000,
    rulePriority: 1,
    customRulesEnabled: true,
    description: 'Motor de regras básicas'
  },
  blacklistLookup: {
    enabled: true,
    threshold: 1.0,
    weight: 0.20,
    cacheTimeout: 300,
    updateFrequency: 3600,
    whitelistOverride: true,
    description: 'Verificação de listas negras'
  },
  velocityChecks: {
    enabled: true,
    threshold: 0.7,
    weight: 0.12,
    timeWindow: 3600,
    maxTransactionsPerWindow: 10,
    velocityByAmount: true,
    description: 'Verificação de velocidade'
  },
  geolocationValidation: {
    enabled: true,
    threshold: 0.6,
    weight: 0.10,
    maxDistance: 1000,
    trustedLocationsEnabled: true,
    vpnDetection: true,
    description: 'Validação geográfica'
  },
  basicStatistics: {
    enabled: true,
    threshold: 0.5,
    weight: 0.08,
    lookbackDays: 30,
    outlierSensitivity: 0.05,
    seasonalAdjustment: true,
    description: 'Estatísticas básicas'
  },

  // Tier 2 - Algoritmos Rápidos (1-5ms)
  randomForest: {
    enabled: true,
    threshold: 0.75,
    weight: 0.18,
    nEstimators: 100,
    maxDepth: 10,
    minSamplesSplit: 2,
    featureImportanceThreshold: 0.01,
    description: 'Random Forest Classifier'
  },
  xgboost: {
    enabled: true,
    threshold: 0.80,
    weight: 0.22,
    learningRate: 0.1,
    maxDepth: 6,
    subsample: 0.8,
    colsampleBytree: 0.8,
    gamma: 0.1,
    description: 'XGBoost Gradient Boosting'
  },
  logisticRegression: {
    enabled: true,
    threshold: 0.65,
    weight: 0.14,
    regularization: 0.01,
    solverType: 'lbfgs',
    maxIterations: 1000,
    classWeight: 'balanced',
    description: 'Regressão Logística'
  },
  svm: {
    enabled: true,
    threshold: 0.70,
    weight: 0.16,
    gamma: 0.001,
    kernelType: 'rbf',
    cParameter: 1.0,
    degree: 3,
    description: 'Support Vector Machine'
  },
  naiveBayes: {
    enabled: true,
    threshold: 0.60,
    weight: 0.12,
    smoothing: 1.0,
    priorProbabilities: 'uniform',
    varianceSmoothing: 1e-9,
    description: 'Naive Bayes Classifier'
  },

  // Tier 3 - Algoritmos Avançados (5-15ms)
  neuralNetwork: {
    enabled: true,
    threshold: 0.85,
    weight: 0.25,
    hiddenLayers: 4,
    neuronsPerLayer: 128,
    activationFunction: 'relu',
    dropoutRate: 0.2,
    batchSize: 32,
    epochs: 100,
    description: 'Rede Neural Profunda'
  },
  lstm: {
    enabled: true,
    threshold: 0.82,
    weight: 0.23,
    sequenceLength: 10,
    lstmUnits: 64,
    returnSequences: false,
    statefulMode: false,
    recurrentDropout: 0.1,
    description: 'LSTM Recorrente'
  },
  transformer: {
    enabled: true,
    threshold: 0.88,
    weight: 0.28,
    attentionHeads: 8,
    modelDimension: 512,
    feedForwardDimension: 2048,
    numberOfLayers: 6,
    positionalEncoding: true,
    description: 'Transformer Attention'
  },
  autoencoder: {
    enabled: true,
    threshold: 0.75,
    weight: 0.20,
    latentDim: 8,
    encoderLayers: 3,
    decoderLayers: 3,
    reconstructionLoss: 'mse',
    anomalyThreshold: 0.1,
    description: 'Autoencoder Anomaly'
  },

  // Tier 4 - Algoritmos Supremos (15-50ms)
  graphTransformer: {
    enabled: true,
    threshold: 0.90,
    weight: 0.30,
    graphDepth: 3,
    nodeFeatures: 64,
    edgeFeatures: 32,
    graphPoolingMethod: 'attention',
    messagePassingRounds: 3,
    description: 'Graph Transformer Networks'
  },
  quantumInspired: {
    enabled: true,
    threshold: 0.92,
    weight: 0.32,
    quantumBits: 16,
    entanglementDepth: 4,
    measurementStrategy: 'computational',
    quantumGates: 'universal',
    decoherenceRate: 0.01,
    description: 'Quantum-Inspired Detection'
  },
  federatedLearning: {
    enabled: true,
    threshold: 0.87,
    weight: 0.28,
    participants: 5,
    aggregationMethod: 'fedavg',
    communicationRounds: 10,
    localEpochs: 5,
    privacyBudget: 1.0,
    description: 'Federated Learning System'
  },

  // Configurações Globais Completas
  global: {
    // Ensemble Básico
    ensembleMethod: 'weighted_average',
    finalThreshold: 0.5,
    autoLearningRate: 0.01,
    driftDetectionSensitivity: 0.05,
    realTimeFeedback: true,
    adaptiveThresholds: true,
    
    // Performance & Sistema
    processingTimeout: 5000,
    maxParallelThreads: 8,
    cacheTTL: 300,
    batchSize: 1000,
    memoryLimit: 2048,
    cpuAffinity: 'auto',
    
    // Segurança & Compliance
    auditLogLevel: 'detailed',
    dataRetentionDays: 90,
    encryptionInTransit: true,
    integrityValidation: true,
    lgpdMode: true,
    autoAnonymization: true,
    
    // Alertas & Notificações
    criticalAlertThreshold: 0.95,
    notificationFrequency: 300,
    emailNotifications: true,
    smsNotifications: false,
    webhookNotifications: true,
    autoEscalation: true,
    silentHoursStart: '22:00',
    silentHoursEnd: '06:00',
    
    // Monitoramento & Métricas
    healthCheckInterval: 30,
    performanceMetricsActive: true,
    metricsRetentionDays: 30,
    dashboardAutoRefresh: 30,
    performanceAlerts: true,
    slaMonitoring: true,
    
    // Backup & Recovery
    autoBackup: true,
    backupFrequency: 'daily',
    backupRetentionDays: 30,
    recoveryPointObjective: 60,
    disasterRecoveryEnabled: true,
    configVersioning: true,
    
    // API & Integração
    rateLimitPerSecond: 1000,
    apiTimeout: 30,
    versioningStrategy: 'header',
    corsEnabled: true,
    authenticationMethod: 'jwt',
    webhookEndpoints: []
  }
};

export function Calibration() {
  const [config, setConfig] = useState(initialConfig);
  const [hasChanges, setHasChanges] = useState(false);
  const [applying, setApplying] = useState(false);
  const [lastApplied, setLastApplied] = useState(null);
  const [activeTab, setActiveTab] = useState('tier1');
  const [activeGlobalTab, setActiveGlobalTab] = useState('ensemble');
  const [impactData, setImpactData] = useState([]);
  const [loading, setLoading] = useState(true);

  // Carregar configuração inicial do backend
  useEffect(() => {
    const loadInitialConfig = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/calibration/config');
        if (response.ok) {
          const result = await response.json();
          if (result.config) {
            setConfig(result.config);
          }
        }
        
        // Carregar dados de impacto
        const impactResponse = await fetch('/api/calibration/impact');
        if (impactResponse.ok) {
          const impactResult = await impactResponse.json();
          if (impactResult.impact_data) {
            setImpactData(impactResult.impact_data);
          }
        }
      } catch (error) {
        console.error('Erro ao carregar configuração inicial:', error);
      } finally {
        setLoading(false);
      }
    };

    loadInitialConfig();
  }, []);

  // Detectar mudanças
  useEffect(() => {
    const hasChanged = JSON.stringify(config) !== JSON.stringify(initialConfig);
    setHasChanges(hasChanged);
  }, [config]);

  // Atualizar configuração de um algoritmo
  const updateAlgorithm = (algorithm, field, value) => {
    setConfig(prev => ({
      ...prev,
      [algorithm]: {
        ...prev[algorithm],
        [field]: value
      }
    }));
  };

  // Atualizar configuração global
  const updateGlobal = (field, value) => {
    setConfig(prev => ({
      ...prev,
      global: {
        ...prev.global,
        [field]: value
      }
    }));
  };

  // Aplicar mudanças ao motor
  const applyChanges = async () => {
    setApplying(true);
    try {
      const response = await fetch('/api/calibration/apply', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          config: config
        })
      });

      if (!response.ok) {
        throw new Error(`Erro HTTP: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        const impactResponse = await fetch('/api/calibration/impact');
        if (impactResponse.ok) {
          const impactResult = await impactResponse.json();
          setImpactData(impactResult.impact_data || []);
        }
        
        setLastApplied(new Date());
        setHasChanges(false);
        
        console.log('✅ Mudanças aplicadas com sucesso:', result.message);
      } else {
        throw new Error(result.message || 'Erro ao aplicar mudanças');
      }
    } catch (error) {
      console.error('Erro ao aplicar mudanças:', error);
      alert('Erro ao aplicar mudanças: ' + error.message);
    } finally {
      setApplying(false);
    }
  };

  // Resetar para valores padrão
  const resetToDefaults = async () => {
    setApplying(true);
    try {
      const response = await fetch('/api/calibration/reset', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (!response.ok) {
        throw new Error(`Erro HTTP: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        setConfig(result.config);
        setHasChanges(false);
        setLastApplied(new Date());
        
        const impactResponse = await fetch('/api/calibration/impact');
        if (impactResponse.ok) {
          const impactResult = await impactResponse.json();
          setImpactData(impactResult.impact_data || []);
        }
        
        console.log('✅ Configurações resetadas com sucesso:', result.message);
      } else {
        throw new Error(result.message || 'Erro ao resetar configurações');
      }
    } catch (error) {
      console.error('Erro ao resetar configurações:', error);
      alert('Erro ao resetar configurações: ' + error.message);
    } finally {
      setApplying(false);
    }
  };

  const tabs = [
    { id: 'tier1', label: 'Tier 1 - Velocistas', icon: Zap, count: 5 },
    { id: 'tier2', label: 'Tier 2 - Rápidos', icon: Activity, count: 5 },
    { id: 'tier3', label: 'Tier 3 - Avançados', icon: Brain, count: 4 },
    { id: 'tier4', label: 'Tier 4 - Supremos', icon: Shield, count: 3 },
    { id: 'global', label: 'Configurações Globais', icon: Settings, count: 6 }
  ];

  const globalTabs = [
    { id: 'ensemble', label: 'Ensemble', icon: TrendingUp },
    { id: 'performance', label: 'Performance', icon: Server },
    { id: 'security', label: 'Segurança', icon: Lock },
    { id: 'alerts', label: 'Alertas', icon: Bell },
    { id: 'monitoring', label: 'Monitoramento', icon: BarChart3 },
    { id: 'backup', label: 'Backup', icon: Database },
    { id: 'api', label: 'API', icon: Wifi }
  ];

  const renderAlgorithmControls = (algorithms) => {
    return Object.entries(algorithms).map(([key, algo]) => (
      <Card key={key} className="p-4">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-medium text-[var(--color-text-primary)]">
                {algo.description}
              </h4>
              <p className="text-xs text-[var(--color-text-secondary)]">
                {key}
              </p>
            </div>
            <SwitchControl
              checked={algo.enabled}
              onCheckedChange={(checked) => updateAlgorithm(key, 'enabled', checked)}
            />
          </div>

          {algo.enabled && (
            <div className="space-y-4 pt-2 border-t border-[var(--color-border)]">
              <SliderControl
                label="Threshold"
                value={[algo.threshold]}
                onValueChange={(value) => updateAlgorithm(key, 'threshold', value[0])}
                min={0}
                max={1}
                step={0.01}
                format="percentage"
                description="Limite de decisão do algoritmo"
              />
              
              <SliderControl
                label="Peso no Ensemble"
                value={[algo.weight]}
                onValueChange={(value) => updateAlgorithm(key, 'weight', value[0])}
                min={0}
                max={0.5}
                step={0.01}
                format="decimal"
                description="Importância relativa no resultado final"
              />

              {/* Parâmetros específicos por algoritmo */}
              {renderSpecificParameters(key, algo)}
            </div>
          )}
        </div>
      </Card>
    ));
  };

  const renderSpecificParameters = (key, algo) => {
    const params = [];

    // Tier 1 - Velocistas
    if (key === 'ruleBasedEngine') {
      params.push(
        <SliderControl
          key="maxAmount"
          label="Valor Máximo (R$)"
          value={[algo.maxAmount]}
          onValueChange={(value) => updateAlgorithm(key, 'maxAmount', value[0])}
          min={1000}
          max={100000}
          step={1000}
          format="currency"
          description="Valor máximo para aprovação automática"
        />,
        <SliderControl
          key="rulePriority"
          label="Prioridade das Regras"
          value={[algo.rulePriority]}
          onValueChange={(value) => updateAlgorithm(key, 'rulePriority', value[0])}
          min={1}
          max={10}
          step={1}
          format="number"
          description="Prioridade na execução das regras"
        />,
        <SwitchControl
          key="customRulesEnabled"
          label="Regras Customizadas"
          checked={algo.customRulesEnabled}
          onCheckedChange={(checked) => updateAlgorithm(key, 'customRulesEnabled', checked)}
          description="Permitir regras definidas pelo usuário"
        />
      );
    }

    if (key === 'blacklistLookup') {
      params.push(
        <SliderControl
          key="cacheTimeout"
          label="Cache Timeout (s)"
          value={[algo.cacheTimeout]}
          onValueChange={(value) => updateAlgorithm(key, 'cacheTimeout', value[0])}
          min={60}
          max={3600}
          step={60}
          format="number"
          description="Tempo de cache das listas"
        />,
        <SliderControl
          key="updateFrequency"
          label="Frequência de Atualização (s)"
          value={[algo.updateFrequency]}
          onValueChange={(value) => updateAlgorithm(key, 'updateFrequency', value[0])}
          min={300}
          max={86400}
          step={300}
          format="number"
          description="Frequência de atualização das listas"
        />,
        <SwitchControl
          key="whitelistOverride"
          label="Override por Whitelist"
          checked={algo.whitelistOverride}
          onCheckedChange={(checked) => updateAlgorithm(key, 'whitelistOverride', checked)}
          description="Permitir override por whitelist"
        />
      );
    }

    if (key === 'velocityChecks') {
      params.push(
        <SliderControl
          key="timeWindow"
          label="Janela de Tempo (s)"
          value={[algo.timeWindow]}
          onValueChange={(value) => updateAlgorithm(key, 'timeWindow', value[0])}
          min={60}
          max={86400}
          step={60}
          format="number"
          description="Janela de tempo para análise de velocidade"
        />,
        <SliderControl
          key="maxTransactionsPerWindow"
          label="Máx. Transações por Janela"
          value={[algo.maxTransactionsPerWindow]}
          onValueChange={(value) => updateAlgorithm(key, 'maxTransactionsPerWindow', value[0])}
          min={1}
          max={100}
          step={1}
          format="number"
          description="Máximo de transações permitidas na janela"
        />,
        <SwitchControl
          key="velocityByAmount"
          label="Velocidade por Valor"
          checked={algo.velocityByAmount}
          onCheckedChange={(checked) => updateAlgorithm(key, 'velocityByAmount', checked)}
          description="Considerar valor nas regras de velocidade"
        />
      );
    }

    if (key === 'geolocationValidation') {
      params.push(
        <SliderControl
          key="maxDistance"
          label="Distância Máxima (km)"
          value={[algo.maxDistance]}
          onValueChange={(value) => updateAlgorithm(key, 'maxDistance', value[0])}
          min={10}
          max={10000}
          step={10}
          format="number"
          description="Distância máxima permitida entre transações"
        />,
        <SwitchControl
          key="trustedLocationsEnabled"
          label="Locais Confiáveis"
          checked={algo.trustedLocationsEnabled}
          onCheckedChange={(checked) => updateAlgorithm(key, 'trustedLocationsEnabled', checked)}
          description="Usar lista de locais confiáveis"
        />,
        <SwitchControl
          key="vpnDetection"
          label="Detecção de VPN"
          checked={algo.vpnDetection}
          onCheckedChange={(checked) => updateAlgorithm(key, 'vpnDetection', checked)}
          description="Detectar uso de VPN/Proxy"
        />
      );
    }

    if (key === 'basicStatistics') {
      params.push(
        <SliderControl
          key="lookbackDays"
          label="Dias de Histórico"
          value={[algo.lookbackDays]}
          onValueChange={(value) => updateAlgorithm(key, 'lookbackDays', value[0])}
          min={7}
          max={365}
          step={1}
          format="number"
          description="Dias de histórico para análise"
        />,
        <SliderControl
          key="outlierSensitivity"
          label="Sensibilidade a Outliers"
          value={[algo.outlierSensitivity]}
          onValueChange={(value) => updateAlgorithm(key, 'outlierSensitivity', value[0])}
          min={0.01}
          max={0.2}
          step={0.01}
          format="decimal"
          description="Sensibilidade para detecção de outliers"
        />,
        <SwitchControl
          key="seasonalAdjustment"
          label="Ajuste Sazonal"
          checked={algo.seasonalAdjustment}
          onCheckedChange={(checked) => updateAlgorithm(key, 'seasonalAdjustment', checked)}
          description="Aplicar ajustes sazonais"
        />
      );
    }

    // Tier 2 - Rápidos
    if (key === 'randomForest') {
      params.push(
        <SliderControl
          key="nEstimators"
          label="Número de Árvores"
          value={[algo.nEstimators]}
          onValueChange={(value) => updateAlgorithm(key, 'nEstimators', value[0])}
          min={10}
          max={500}
          step={10}
          format="number"
          description="Número de árvores na floresta"
        />,
        <SliderControl
          key="maxDepth"
          label="Profundidade Máxima"
          value={[algo.maxDepth]}
          onValueChange={(value) => updateAlgorithm(key, 'maxDepth', value[0])}
          min={3}
          max={20}
          step={1}
          format="number"
          description="Profundidade máxima das árvores"
        />,
        <SliderControl
          key="minSamplesSplit"
          label="Min. Amostras para Split"
          value={[algo.minSamplesSplit]}
          onValueChange={(value) => updateAlgorithm(key, 'minSamplesSplit', value[0])}
          min={2}
          max={20}
          step={1}
          format="number"
          description="Mínimo de amostras para dividir um nó"
        />
      );
    }

    if (key === 'xgboost') {
      params.push(
        <SliderControl
          key="learningRate"
          label="Learning Rate"
          value={[algo.learningRate]}
          onValueChange={(value) => updateAlgorithm(key, 'learningRate', value[0])}
          min={0.01}
          max={0.3}
          step={0.01}
          format="decimal"
          description="Taxa de aprendizado do modelo"
        />,
        <SliderControl
          key="maxDepth"
          label="Profundidade Máxima"
          value={[algo.maxDepth]}
          onValueChange={(value) => updateAlgorithm(key, 'maxDepth', value[0])}
          min={3}
          max={15}
          step={1}
          format="number"
          description="Profundidade máxima das árvores"
        />,
        <SliderControl
          key="subsample"
          label="Subsample"
          value={[algo.subsample]}
          onValueChange={(value) => updateAlgorithm(key, 'subsample', value[0])}
          min={0.5}
          max={1.0}
          step={0.1}
          format="decimal"
          description="Fração de amostras para treinamento"
        />,
        <SliderControl
          key="colsampleBytree"
          label="Colsample by Tree"
          value={[algo.colsampleBytree]}
          onValueChange={(value) => updateAlgorithm(key, 'colsampleBytree', value[0])}
          min={0.5}
          max={1.0}
          step={0.1}
          format="decimal"
          description="Fração de features por árvore"
        />
      );
    }

    // Tier 3 - Avançados
    if (key === 'neuralNetwork') {
      params.push(
        <SliderControl
          key="hiddenLayers"
          label="Camadas Ocultas"
          value={[algo.hiddenLayers]}
          onValueChange={(value) => updateAlgorithm(key, 'hiddenLayers', value[0])}
          min={1}
          max={10}
          step={1}
          format="number"
          description="Número de camadas ocultas"
        />,
        <SliderControl
          key="neuronsPerLayer"
          label="Neurônios por Camada"
          value={[algo.neuronsPerLayer]}
          onValueChange={(value) => updateAlgorithm(key, 'neuronsPerLayer', value[0])}
          min={32}
          max={512}
          step={32}
          format="number"
          description="Número de neurônios por camada"
        />,
        <SliderControl
          key="dropoutRate"
          label="Taxa de Dropout"
          value={[algo.dropoutRate]}
          onValueChange={(value) => updateAlgorithm(key, 'dropoutRate', value[0])}
          min={0.0}
          max={0.5}
          step={0.1}
          format="decimal"
          description="Taxa de dropout para regularização"
        />
      );
    }

    return params;
  };

  const renderGlobalControls = () => {
    const globalConfig = config.global;

    switch (activeGlobalTab) {
      case 'ensemble':
        return (
          <div className="space-y-6">
            <Card className="p-6">
              <h4 className="font-medium text-[var(--color-text-primary)] mb-4">
                Configurações do Ensemble
              </h4>
              <div className="space-y-4">
                <SliderControl
                  label="Threshold Final"
                  value={[globalConfig.finalThreshold]}
                  onValueChange={(value) => updateGlobal('finalThreshold', value[0])}
                  min={0}
                  max={1}
                  step={0.01}
                  format="percentage"
                  description="Threshold final para classificação como fraude"
                />
                
                <SliderControl
                  label="Taxa de Auto-Learning"
                  value={[globalConfig.autoLearningRate]}
                  onValueChange={(value) => updateGlobal('autoLearningRate', value[0])}
                  min={0.001}
                  max={0.1}
                  step={0.001}
                  format="decimal"
                  description="Velocidade de adaptação dos modelos"
                />
                
                <SliderControl
                  label="Sensibilidade de Drift"
                  value={[globalConfig.driftDetectionSensitivity]}
                  onValueChange={(value) => updateGlobal('driftDetectionSensitivity', value[0])}
                  min={0.01}
                  max={0.2}
                  step={0.01}
                  format="decimal"
                  description="Sensibilidade para detecção de concept drift"
                />
              </div>
            </Card>

            <Card className="p-6">
              <h4 className="font-medium text-[var(--color-text-primary)] mb-4">
                Funcionalidades Avançadas
              </h4>
              <div className="space-y-4">
                <SwitchControl
                  label="Feedback em Tempo Real"
                  checked={globalConfig.realTimeFeedback}
                  onCheckedChange={(checked) => updateGlobal('realTimeFeedback', checked)}
                  description="Aplicar feedback imediatamente aos modelos"
                />
                
                <SwitchControl
                  label="Thresholds Adaptativos"
                  checked={globalConfig.adaptiveThresholds}
                  onCheckedChange={(checked) => updateGlobal('adaptiveThresholds', checked)}
                  description="Ajustar thresholds automaticamente baseado na performance"
                />
              </div>
            </Card>
          </div>
        );

      case 'performance':
        return (
          <div className="space-y-6">
            <Card className="p-6">
              <h4 className="font-medium text-[var(--color-text-primary)] mb-4">
                Performance do Sistema
              </h4>
              <div className="space-y-4">
                <SliderControl
                  label="Timeout de Processamento (ms)"
                  value={[globalConfig.processingTimeout]}
                  onValueChange={(value) => updateGlobal('processingTimeout', value[0])}
                  min={1000}
                  max={30000}
                  step={1000}
                  format="number"
                  description="Timeout máximo para processamento"
                />
                
                <SliderControl
                  label="Máximo de Threads Paralelas"
                  value={[globalConfig.maxParallelThreads]}
                  onValueChange={(value) => updateGlobal('maxParallelThreads', value[0])}
                  min={1}
                  max={32}
                  step={1}
                  format="number"
                  description="Número máximo de threads paralelas"
                />
                
                <SliderControl
                  label="Cache TTL (segundos)"
                  value={[globalConfig.cacheTTL]}
                  onValueChange={(value) => updateGlobal('cacheTTL', value[0])}
                  min={60}
                  max={3600}
                  step={60}
                  format="number"
                  description="Tempo de vida do cache"
                />
                
                <SliderControl
                  label="Batch Size"
                  value={[globalConfig.batchSize]}
                  onValueChange={(value) => updateGlobal('batchSize', value[0])}
                  min={100}
                  max={10000}
                  step={100}
                  format="number"
                  description="Tamanho do lote para processamento"
                />
                
                <SliderControl
                  label="Limite de Memória (MB)"
                  value={[globalConfig.memoryLimit]}
                  onValueChange={(value) => updateGlobal('memoryLimit', value[0])}
                  min={512}
                  max={8192}
                  step={512}
                  format="number"
                  description="Limite de memória para o sistema"
                />
              </div>
            </Card>
          </div>
        );

      case 'security':
        return (
          <div className="space-y-6">
            <Card className="p-6">
              <h4 className="font-medium text-[var(--color-text-primary)] mb-4">
                Segurança e Compliance
              </h4>
              <div className="space-y-4">
                <SliderControl
                  label="Retenção de Dados (dias)"
                  value={[globalConfig.dataRetentionDays]}
                  onValueChange={(value) => updateGlobal('dataRetentionDays', value[0])}
                  min={30}
                  max={365}
                  step={1}
                  format="number"
                  description="Dias de retenção de dados sensíveis"
                />
                
                <SwitchControl
                  label="Criptografia em Trânsito"
                  checked={globalConfig.encryptionInTransit}
                  onCheckedChange={(checked) => updateGlobal('encryptionInTransit', checked)}
                  description="Criptografar dados em trânsito"
                />
                
                <SwitchControl
                  label="Validação de Integridade"
                  checked={globalConfig.integrityValidation}
                  onCheckedChange={(checked) => updateGlobal('integrityValidation', checked)}
                  description="Validar integridade dos dados"
                />
                
                <SwitchControl
                  label="Modo LGPD/GDPR"
                  checked={globalConfig.lgpdMode}
                  onCheckedChange={(checked) => updateGlobal('lgpdMode', checked)}
                  description="Ativar conformidade LGPD/GDPR"
                />
                
                <SwitchControl
                  label="Anonimização Automática"
                  checked={globalConfig.autoAnonymization}
                  onCheckedChange={(checked) => updateGlobal('autoAnonymization', checked)}
                  description="Anonimizar dados automaticamente"
                />
              </div>
            </Card>
          </div>
        );

      case 'alerts':
        return (
          <div className="space-y-6">
            <Card className="p-6">
              <h4 className="font-medium text-[var(--color-text-primary)] mb-4">
                Configurações de Alertas
              </h4>
              <div className="space-y-4">
                <SliderControl
                  label="Threshold de Alerta Crítico"
                  value={[globalConfig.criticalAlertThreshold]}
                  onValueChange={(value) => updateGlobal('criticalAlertThreshold', value[0])}
                  min={0.8}
                  max={1.0}
                  step={0.01}
                  format="percentage"
                  description="Threshold para alertas críticos"
                />
                
                <SliderControl
                  label="Frequência de Notificações (s)"
                  value={[globalConfig.notificationFrequency]}
                  onValueChange={(value) => updateGlobal('notificationFrequency', value[0])}
                  min={60}
                  max={3600}
                  step={60}
                  format="number"
                  description="Intervalo entre notificações"
                />
                
                <SwitchControl
                  label="Notificações por Email"
                  checked={globalConfig.emailNotifications}
                  onCheckedChange={(checked) => updateGlobal('emailNotifications', checked)}
                  description="Enviar alertas por email"
                />
                
                <SwitchControl
                  label="Notificações por SMS"
                  checked={globalConfig.smsNotifications}
                  onCheckedChange={(checked) => updateGlobal('smsNotifications', checked)}
                  description="Enviar alertas por SMS"
                />
                
                <SwitchControl
                  label="Webhooks"
                  checked={globalConfig.webhookNotifications}
                  onCheckedChange={(checked) => updateGlobal('webhookNotifications', checked)}
                  description="Enviar alertas via webhook"
                />
                
                <SwitchControl
                  label="Escalação Automática"
                  checked={globalConfig.autoEscalation}
                  onCheckedChange={(checked) => updateGlobal('autoEscalation', checked)}
                  description="Escalar alertas automaticamente"
                />
              </div>
            </Card>
          </div>
        );

      case 'monitoring':
        return (
          <div className="space-y-6">
            <Card className="p-6">
              <h4 className="font-medium text-[var(--color-text-primary)] mb-4">
                Monitoramento e Métricas
              </h4>
              <div className="space-y-4">
                <SliderControl
                  label="Intervalo de Health Check (s)"
                  value={[globalConfig.healthCheckInterval]}
                  onValueChange={(value) => updateGlobal('healthCheckInterval', value[0])}
                  min={10}
                  max={300}
                  step={10}
                  format="number"
                  description="Intervalo entre verificações de saúde"
                />
                
                <SliderControl
                  label="Retenção de Métricas (dias)"
                  value={[globalConfig.metricsRetentionDays]}
                  onValueChange={(value) => updateGlobal('metricsRetentionDays', value[0])}
                  min={7}
                  max={90}
                  step={1}
                  format="number"
                  description="Dias de retenção das métricas"
                />
                
                <SliderControl
                  label="Auto-Refresh Dashboard (s)"
                  value={[globalConfig.dashboardAutoRefresh]}
                  onValueChange={(value) => updateGlobal('dashboardAutoRefresh', value[0])}
                  min={10}
                  max={300}
                  step={10}
                  format="number"
                  description="Intervalo de atualização do dashboard"
                />
                
                <SwitchControl
                  label="Métricas de Performance Ativas"
                  checked={globalConfig.performanceMetricsActive}
                  onCheckedChange={(checked) => updateGlobal('performanceMetricsActive', checked)}
                  description="Coletar métricas de performance"
                />
                
                <SwitchControl
                  label="Alertas de Performance"
                  checked={globalConfig.performanceAlerts}
                  onCheckedChange={(checked) => updateGlobal('performanceAlerts', checked)}
                  description="Alertas baseados em performance"
                />
                
                <SwitchControl
                  label="Monitoramento SLA"
                  checked={globalConfig.slaMonitoring}
                  onCheckedChange={(checked) => updateGlobal('slaMonitoring', checked)}
                  description="Monitorar SLAs do sistema"
                />
              </div>
            </Card>
          </div>
        );

      case 'backup':
        return (
          <div className="space-y-6">
            <Card className="p-6">
              <h4 className="font-medium text-[var(--color-text-primary)] mb-4">
                Backup e Recovery
              </h4>
              <div className="space-y-4">
                <SliderControl
                  label="Retenção de Backups (dias)"
                  value={[globalConfig.backupRetentionDays]}
                  onValueChange={(value) => updateGlobal('backupRetentionDays', value[0])}
                  min={7}
                  max={365}
                  step={1}
                  format="number"
                  description="Dias de retenção dos backups"
                />
                
                <SliderControl
                  label="Recovery Point Objective (min)"
                  value={[globalConfig.recoveryPointObjective]}
                  onValueChange={(value) => updateGlobal('recoveryPointObjective', value[0])}
                  min={15}
                  max={1440}
                  step={15}
                  format="number"
                  description="Objetivo de ponto de recuperação"
                />
                
                <SwitchControl
                  label="Backup Automático"
                  checked={globalConfig.autoBackup}
                  onCheckedChange={(checked) => updateGlobal('autoBackup', checked)}
                  description="Realizar backups automaticamente"
                />
                
                <SwitchControl
                  label="Disaster Recovery"
                  checked={globalConfig.disasterRecoveryEnabled}
                  onCheckedChange={(checked) => updateGlobal('disasterRecoveryEnabled', checked)}
                  description="Ativar plano de disaster recovery"
                />
                
                <SwitchControl
                  label="Versionamento de Configurações"
                  checked={globalConfig.configVersioning}
                  onCheckedChange={(checked) => updateGlobal('configVersioning', checked)}
                  description="Versionar mudanças de configuração"
                />
              </div>
            </Card>
          </div>
        );

      case 'api':
        return (
          <div className="space-y-6">
            <Card className="p-6">
              <h4 className="font-medium text-[var(--color-text-primary)] mb-4">
                API e Integração
              </h4>
              <div className="space-y-4">
                <SliderControl
                  label="Rate Limit (req/s)"
                  value={[globalConfig.rateLimitPerSecond]}
                  onValueChange={(value) => updateGlobal('rateLimitPerSecond', value[0])}
                  min={100}
                  max={10000}
                  step={100}
                  format="number"
                  description="Limite de requisições por segundo"
                />
                
                <SliderControl
                  label="API Timeout (s)"
                  value={[globalConfig.apiTimeout]}
                  onValueChange={(value) => updateGlobal('apiTimeout', value[0])}
                  min={5}
                  max={120}
                  step={5}
                  format="number"
                  description="Timeout para chamadas de API"
                />
                
                <SwitchControl
                  label="CORS Habilitado"
                  checked={globalConfig.corsEnabled}
                  onCheckedChange={(checked) => updateGlobal('corsEnabled', checked)}
                  description="Permitir requisições cross-origin"
                />
              </div>
            </Card>
          </div>
        );

      default:
        return null;
    }
  };

  const getTierAlgorithms = (tier) => {
    const tiers = {
      tier1: ['ruleBasedEngine', 'blacklistLookup', 'velocityChecks', 'geolocationValidation', 'basicStatistics'],
      tier2: ['randomForest', 'xgboost', 'logisticRegression', 'svm', 'naiveBayes'],
      tier3: ['neuralNetwork', 'lstm', 'transformer', 'autoencoder'],
      tier4: ['graphTransformer', 'quantumInspired', 'federatedLearning']
    };

    const algorithms = {};
    tiers[tier]?.forEach(key => {
      if (config[key]) {
        algorithms[key] = config[key];
      }
    });
    return algorithms;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center space-x-2">
          <RefreshCw className="h-6 w-6 animate-spin" />
          <span>Carregando configurações de calibragem...</span>
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
            <Settings className="h-8 w-8" />
            <span>Calibragem Manual</span>
          </h1>
          <p className="text-[var(--color-text-secondary)] mt-1">
            Ajuste em tempo real dos parâmetros dos algoritmos de IA
          </p>
        </div>
        <div className="flex items-center space-x-2">
          {lastApplied && (
            <Badge variant="success" className="flex items-center space-x-1">
              <CheckCircle className="h-3 w-3" />
              <span>Aplicado às {lastApplied.toLocaleTimeString('pt-BR')}</span>
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
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Button
            onClick={applyChanges}
            disabled={!hasChanges || applying}
            loading={applying}
            className="flex items-center space-x-2"
          >
            <Save className="h-4 w-4" />
            <span>Aplicar Mudanças ao Motor</span>
          </Button>
          
          <Button
            variant="secondary"
            onClick={resetToDefaults}
            disabled={applying}
            className="flex items-center space-x-2"
          >
            <RotateCcw className="h-4 w-4" />
            <span>Resetar Padrões</span>
          </Button>
        </div>

        <Button
          variant="tertiary"
          className="flex items-center space-x-2"
        >
          <History className="h-4 w-4" />
          <span>Histórico de Mudanças</span>
        </Button>
      </div>

      {/* Impact Chart */}
      {impactData.length > 0 && (
        <SimpleLineChart
          title="Impacto das Calibrações em Tempo Real"
          data={impactData}
          dataKey="f1Score"
          xAxisKey="time"
          height={200}
          formatter={(value) => `${value.toFixed(1)}%`}
        />
      )}

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
                <Badge variant="default" size="sm">{tab.count}</Badge>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Global Tabs */}
      {activeTab === 'global' && (
        <div className="border-b border-[var(--color-border)]">
          <nav className="flex space-x-6">
            {globalTabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveGlobalTab(tab.id)}
                  className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                    activeGlobalTab === tab.id
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
      )}

      {/* Tab Content */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {activeTab === 'global' 
          ? renderGlobalControls()
          : renderAlgorithmControls(getTierAlgorithms(activeTab))
        }
      </div>

      {/* Status Footer */}
      <Card className="p-4 bg-[var(--neutral-50)]">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Activity className="h-4 w-4 text-green-500" />
              <span className="text-sm">Motor Online</span>
            </div>
            <div className="flex items-center space-x-2">
              <Clock className="h-4 w-4 text-blue-500" />
              <span className="text-sm">Latência: {config.global?.systemLatency || 'N/A'}</span>
            </div>
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-4 w-4 text-green-500" />
              <span className="text-sm">Precisão: {config.global?.systemAccuracy || 'N/A'}</span>
            </div>
          </div>
          <div className="text-xs text-[var(--color-text-secondary)]">
            Última atualização: {new Date().toLocaleTimeString('pt-BR')}
          </div>
        </div>
      </Card>
    </div>
  );
}

