import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
import time
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from backend.models.transaction_model import Transaction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UltraPrecisionFraudEngineV4:
    """
    Motor de Detecção de Fraude V4.0 - Ultra Precision
    
    Melhorias implementadas com base na análise do V3.0:
    1. Ensemble mais robusto com 5 modelos diferentes
    2. Calibração de probabilidades para melhor precision
    3. Feature selection automática
    4. Múltiplos scalers para diferentes tipos de features
    5. Validação cruzada estratificada
    6. Threshold otimizado por múltiplas métricas
    7. Sistema de pesos adaptativos para o ensemble
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # Ensemble mais robusto com 5 modelos diferentes
        self.base_models = {
            'rf': RandomForestClassifier(
                random_state=self.random_state, 
                n_estimators=300, 
                max_depth=12, 
                class_weight='balanced',
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'gb': GradientBoostingClassifier(
                random_state=self.random_state, 
                n_estimators=300, 
                learning_rate=0.03, 
                max_depth=6,
                subsample=0.8
            ),
            'et': ExtraTreesClassifier(
                random_state=self.random_state,
                n_estimators=200,
                max_depth=10,
                class_weight='balanced',
                min_samples_split=5
            ),
            'lr': LogisticRegression(
                random_state=self.random_state, 
                solver='liblinear', 
                class_weight='balanced',
                C=0.1
            ),
            'svm': SVC(
                random_state=self.random_state,
                probability=True,
                class_weight='balanced',
                C=0.1,
                gamma='scale'
            )
        }
        
        # Modelos calibrados para melhor precision
        self.calibrated_models = {}
        
        # Meta-modelo mais sofisticado
        self.meta_model = LogisticRegression(
            random_state=self.random_state, 
            solver='liblinear',
            C=1.0
        )
        
        # Múltiplos preprocessadores
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.quantile_transformer = QuantileTransformer(
            output_distribution='normal', 
            random_state=self.random_state,
            n_quantiles=min(1000, 100)  # Ajuste dinâmico
        )
        self.imputer = SimpleImputer(strategy='median')
        
        # Feature selection
        self.feature_selector = SelectKBest(f_classif, k=50)  # Top 50 features
        
        # Pesos adaptativos para o ensemble
        self.model_weights = {}
        
        self.feature_names = []
        self.selected_features = []
        self.is_trained = False
        self.threshold = 0.5
        self.precision_threshold = 0.7  # Threshold específico para alta precision

    def _advanced_feature_engineering(self, df):
        """Feature engineering avançada para melhorar a separação entre classes"""
        
        # Converter timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Features temporais básicas
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Features de valor mais sofisticadas
        df['log_valor'] = np.log1p(df['valor'])
        df['valor_squared'] = df['valor'] ** 2
        df['valor_sqrt'] = np.sqrt(df['valor'])
        
        # Interações temporais com valor
        df['valor_per_hour'] = df['valor'] / (df['hour'] + 1)
        df['valor_weekend_multiplier'] = df['valor'] * df['is_weekend']
        df['valor_night_multiplier'] = df['valor'] * df['is_night']
        
        # Features geográficas avançadas
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Distância do centro financeiro (São Paulo como referência)
            sp_lat, sp_lon = -23.5505, -46.6333
            df['distance_from_sp'] = np.sqrt(
                (df['latitude'] - sp_lat)**2 + (df['longitude'] - sp_lon)**2
            )
            df['is_far_from_center'] = (df['distance_from_sp'] > 5).astype(int)
        
        # Drop colunas originais que não serão usadas
        df = df.drop(columns=["timestamp", "id", "cliente_cpf", "ip_address"], errors='ignore')
        
        # Encoding categórico mais inteligente
        categorical_cols = ["tipo_transacao", "canal", "cidade", "estado", "pais", "device_id", "conta_recebedor"]
        
        # One-hot encoding com limite de categorias
        for col in categorical_cols:
            if col in df.columns:
                # Manter apenas as top 10 categorias mais frequentes
                top_categories = df[col].value_counts().head(10).index
                df[col] = df[col].apply(lambda x: x if x in top_categories else 'OTHER')
        
        df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], drop_first=True)
        
        return df

    def _advanced_preprocessing(self, df, fit=True):
        """Preprocessamento avançado com múltiplos scalers"""
        
        # Separar diferentes tipos de features
        value_features = [col for col in df.columns if 'valor' in col.lower()]
        temporal_features = [col for col in df.columns if any(x in col.lower() for x in ['hour', 'day', 'month', 'weekend', 'night', 'business'])]
        geo_features = [col for col in df.columns if any(x in col.lower() for x in ['latitude', 'longitude', 'distance'])]
        categorical_features = [col for col in df.columns if col not in value_features + temporal_features + geo_features]
        
        # Aplicar diferentes scalers para diferentes tipos de features
        if value_features:
            if fit:
                df[value_features] = self.robust_scaler.fit_transform(df[value_features])
            else:
                df[value_features] = self.robust_scaler.transform(df[value_features])
        
        if temporal_features + geo_features:
            combined_features = temporal_features + geo_features
            if fit:
                df[combined_features] = self.standard_scaler.fit_transform(df[combined_features])
            else:
                df[combined_features] = self.standard_scaler.transform(df[combined_features])
        
        # Imputação
        if fit:
            df = pd.DataFrame(self.imputer.fit_transform(df), columns=df.columns)
        else:
            df = pd.DataFrame(self.imputer.transform(df), columns=df.columns)
        
        # Transformação quantile para normalização final
        if fit:
            # Ajustar n_quantiles baseado no tamanho dos dados
            n_samples = len(df)
            self.quantile_transformer.n_quantiles = min(1000, max(10, n_samples // 10))
            df = pd.DataFrame(self.quantile_transformer.fit_transform(df), columns=df.columns)
        else:
            df = pd.DataFrame(self.quantile_transformer.transform(df), columns=df.columns)
        
        return df

    def _calculate_model_weights(self, X_val, y_val):
        """Calcular pesos adaptativos baseados na performance individual de cada modelo"""
        weights = {}
        
        for name, model in self.calibrated_models.items():
            y_pred = model.predict(X_val)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            # Peso baseado em precision (prioridade) e f1-score
            weight = 0.7 * precision + 0.3 * f1
            weights[name] = max(weight, 0.1)  # Peso mínimo de 0.1
        
        # Normalizar pesos
        total_weight = sum(weights.values())
        self.model_weights = {k: v/total_weight for k, v in weights.items()}
        
        logging.info(f"Pesos dos modelos: {self.model_weights}")

    def train(self, X_train, y_train):
        """Treinamento avançado com validação cruzada e calibração"""
        logging.info("Iniciando treinamento do motor Ultra-Precision V4.0...")
        start_time = time.time()
        
        # Preparar dados
        df_train = pd.DataFrame(X_train)
        df_train["is_fraud"] = y_train
        
        # Feature engineering
        df_train_processed = df_train.drop(columns=["is_fraud"], errors='ignore')
        X_processed = self._advanced_feature_engineering(df_train_processed)
        y_processed = df_train["is_fraud"]
        
        # Preprocessamento
        X_processed = self._advanced_preprocessing(X_processed, fit=True)
        self.feature_names = X_processed.columns.tolist()
        
        # Feature selection
        X_selected = self.feature_selector.fit_transform(X_processed, y_processed)
        self.selected_features = [self.feature_names[i] for i in self.feature_selector.get_support(indices=True)]
        
        logging.info(f"Features selecionadas: {len(self.selected_features)} de {len(self.feature_names)}")
        
        # Aplicar SMOTE para lidar com o desbalanceamento de classes
        logging.info("Aplicando SMOTE para balancear as classes...")
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_selected, y_processed)
        logging.info(f"Dados após SMOTE: {len(X_resampled)} amostras, {np.sum(y_resampled)} fraudes.")

        # Split para validação
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=self.random_state, stratify=y_resampled
        )

        
        # Treinar modelos base com validação cruzada
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        for name, model in self.base_models.items():
            logging.info(f"Treinando modelo base: {name}")
            
            # Validação cruzada para avaliar estabilidade
            cv_scores = cross_val_score(model, X_train_split, y_train_split, cv=cv, scoring='f1')
            logging.info(f"CV F1-Score para {name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Treinar modelo completo
            model.fit(X_train_split, y_train_split)
            
            # Calibrar probabilidades para melhor precision
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
            calibrated_model.fit(X_train_split, y_train_split)
            self.calibrated_models[name] = calibrated_model
        
        # Calcular pesos adaptativos
        self._calculate_model_weights(X_val_split, y_val_split)
        
        # Preparar dados para meta-modelo
        meta_features = np.zeros((len(X_train_split), len(self.calibrated_models)))
        for i, (name, model) in enumerate(self.calibrated_models.items()):
            meta_features[:, i] = model.predict_proba(X_train_split)[:, 1]
        
        # Treinar meta-modelo
        logging.info("Treinando meta-modelo...")
        self.meta_model.fit(meta_features, y_train_split)
        
        self.is_trained = True
        end_time = time.time()
        logging.info(f"Treinamento concluído em {end_time - start_time:.2f} segundos.")
        
        # Calibração avançada de threshold
        self._advanced_threshold_calibration(X_val_split, y_val_split)

    def _advanced_threshold_calibration(self, X_val, y_val):
        """Calibração avançada de threshold otimizando múltiplas métricas"""
        logging.info("Calibrando thresholds para otimização multi-objetivo...")
        
        # Obter probabilidades de validação
        probabilities = self._predict_proba_internal(X_val)[:, 1]
        
        best_f1 = -1
        best_precision = -1
        best_balanced_score = -1
        best_threshold = 0.5
        best_precision_threshold = 0.7
        
        # Testar diferentes thresholds
        for t in np.linspace(0.01, 0.99, 200):
            y_pred = (probabilities >= t).astype(int)
            
            if np.sum(y_pred) == 0:  # Evitar divisão por zero
                continue
                
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            # Score balanceado priorizando precision
            balanced_score = 0.6 * precision + 0.4 * f1
            
            # Threshold para F1-Score geral
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
            
            # Threshold específico para alta precision (>= 70%)
            if precision >= 0.70 and f1 > best_precision:
                best_precision = f1
                best_precision_threshold = t
            
            # Threshold balanceado
            if balanced_score > best_balanced_score:
                best_balanced_score = balanced_score
        
        self.threshold = best_threshold
        self.precision_threshold = best_precision_threshold
        
        logging.info(f"Threshold F1 otimizado: {self.threshold:.4f} (F1-Score: {best_f1:.4f})")
        logging.info(f"Threshold Precision otimizado: {self.precision_threshold:.4f} (Precision: {best_precision:.4f})")

    def _predict_proba_internal(self, X_processed):
        """Predição interna de probabilidades com ensemble ponderado"""
        # Preparar features para meta-modelo
        meta_features = np.zeros((len(X_processed), len(self.calibrated_models)))
        
        for i, (name, model) in enumerate(self.calibrated_models.items()):
            base_proba = model.predict_proba(X_processed)[:, 1]
            # Aplicar peso adaptativo
            weighted_proba = base_proba * self.model_weights[name]
            meta_features[:, i] = weighted_proba
        
        # Predição final do meta-modelo
        return self.meta_model.predict_proba(meta_features)

    def predict_proba(self, X_input):
        """Predição de probabilidades com preprocessamento completo"""
        if not self.is_trained:
            raise ValueError("O motor não foi treinado. Por favor, treine o motor antes de fazer predições.")
        
        # Processar entrada
        if isinstance(X_input, Transaction):
            df_test = pd.DataFrame([X_input.__dict__])
        elif isinstance(X_input, list) and len(X_input) > 0 and isinstance(X_input[0], dict):
            df_test = pd.DataFrame(X_input)
        else:
            df_test = pd.DataFrame([X_input.__dict__] if hasattr(X_input, '__dict__') else [X_input])
        
        # Feature engineering
        X_processed = self._advanced_feature_engineering(df_test.drop(columns=["is_fraud", "fraud_score"], errors='ignore'))
        
        # Preprocessamento
        X_processed = self._advanced_preprocessing(X_processed, fit=False)
        
        # Alinhar features com treinamento
        missing_cols = set(self.feature_names) - set(X_processed.columns)
        for c in missing_cols:
            X_processed[c] = 0
        X_processed = X_processed[self.feature_names]
        
        # Feature selection
        X_selected = self.feature_selector.transform(X_processed)
        
        return self._predict_proba_internal(X_selected)

    def predict(self, X, use_precision_threshold=False):
        """Predição com opção de usar threshold de alta precision"""
        probabilities = self.predict_proba(X)[:, 1]
        threshold = self.precision_threshold if use_precision_threshold else self.threshold
        return (probabilities >= threshold).astype(int)

    def analyze_transaction(self, transaction: Transaction, use_precision_threshold=False) -> Dict[str, Any]:
        """Análise avançada de transação com múltiplos níveis de risco"""
        if not self.is_trained:
            raise ValueError("O motor não foi treinado. Por favor, treine o motor antes de analisar transações.")

        df_single = pd.DataFrame([transaction.__dict__])
        
        # Garantir colunas categóricas
        for col in ['tipo_transacao', 'canal', 'cidade', 'estado', 'pais', 'device_id', 'conta_recebedor']:
            if col not in df_single.columns:
                df_single[col] = ''
        
        # Preprocessamento
        X_processed = self._advanced_feature_engineering(df_single.drop(columns=["is_fraud", "fraud_score"], errors='ignore'))
        X_processed = self._advanced_preprocessing(X_processed, fit=False)
        
        # Alinhar features
        missing_cols = set(self.feature_names) - set(X_processed.columns)
        for c in missing_cols:
            X_processed[c] = 0
        X_processed = X_processed[self.feature_names]
        
        # Feature selection
        X_selected = self.feature_selector.transform(X_processed)
        
        # Predições
        fraud_probability = self._predict_proba_internal(X_selected)[:, 1][0]
        
        # Múltiplos thresholds
        is_fraud_standard = fraud_probability >= self.threshold
        is_fraud_precision = fraud_probability >= self.precision_threshold
        
        # Níveis de risco mais granulares
        if fraud_probability >= self.precision_threshold:
            risk_level = "CRÍTICO"
        elif fraud_probability >= self.threshold:
            risk_level = "ALTO"
        elif fraud_probability >= (self.threshold * 0.6):
            risk_level = "MÉDIO"
        elif fraud_probability >= (self.threshold * 0.3):
            risk_level = "BAIXO"
        else:
            risk_level = "MÍNIMO"
        
        # Confiança da predição
        confidence = abs(fraud_probability - 0.5) * 2  # 0 a 1
        
        return {
            "transaction_id": transaction.id,
            "fraud_score": float(f"{fraud_probability:.6f}"),
            "is_fraud_prediction": bool(is_fraud_standard),
            "is_fraud_high_precision": bool(is_fraud_precision),
            "risk_level": risk_level,
            "confidence": float(f"{confidence:.4f}"),
            "thresholds": {
                "standard": float(f"{self.threshold:.4f}"),
                "high_precision": float(f"{self.precision_threshold:.4f}")
            },
            "analysis_details": "Análise V4.0: Ensemble calibrado com 5 modelos, feature selection automática e thresholds otimizados para alta precision."
        }

# Teste integrado
if __name__ == "__main__":
    logging.info("Iniciando teste do UltraPrecisionFraudEngineV4...")
    
    # Gerar dados sintéticos mais realistas
    def generate_realistic_fraud_data(num_samples=50000, fraud_ratio=0.02):
        """Geração de dados sintéticos mais realistas para teste"""
        np.random.seed(42)
        
        data = {
            'id': [f'TXN_{i}' for i in range(num_samples)],
            'valor': np.random.lognormal(mean=6, sigma=2, size=num_samples),
            'tipo_transacao': np.random.choice(['PIX', 'DEBITO', 'CREDITO', 'TED', 'DOC'], num_samples, p=[0.45, 0.25, 0.15, 0.1, 0.05]),
            'canal': np.random.choice(['MOBILE', 'WEB', 'POS', 'ATM'], num_samples, p=[0.6, 0.25, 0.1, 0.05]),
            'cidade': np.random.choice(['Sao Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Porto Alegre', 'Curitiba', 'Salvador', 'Brasilia'], num_samples),
            'estado': np.random.choice(['SP', 'RJ', 'MG', 'RS', 'PR', 'BA', 'DF'], num_samples),
            'pais': ['BR'] * num_samples,
            'ip_address': [f'192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}' for _ in range(num_samples)],
            'device_id': [f'DEV_{np.random.randint(1, 1000)}' for _ in range(num_samples)],
            'conta_recebedor': [f'REC_{np.random.randint(1, 2000)}' for _ in range(num_samples)],
            'cliente_cpf': [f'CPF_{np.random.randint(10000000000, 99999999999)}' for _ in range(num_samples)],
            'timestamp': pd.date_range(start='2024-01-01', periods=num_samples, freq='30S').astype(str).tolist(),
            'latitude': np.random.normal(-15, 8, num_samples),  # Distribuição mais realista para Brasil
            'longitude': np.random.normal(-50, 10, num_samples),
            'is_fraud': np.random.choice([True, False], num_samples, p=[fraud_ratio, 1-fraud_ratio])
        }
        
        df = pd.DataFrame(data)
        
        # Injetar padrões de fraude mais realistas
        fraud_indices = df[df['is_fraud'] == True].index
        
        if not fraud_indices.empty:
            # Fraudes com valores altos em horários suspeitos
            night_fraud = fraud_indices[:len(fraud_indices)//3]
            df.loc[night_fraud, 'timestamp'] = pd.date_range(start='2024-01-01 02:00:00', periods=len(night_fraud), freq='1H').astype(str).tolist()
            df.loc[night_fraud, 'valor'] = np.random.uniform(3000, 15000, len(night_fraud))
            
            # Fraudes com múltiplas transações do mesmo device
            device_fraud = fraud_indices[len(fraud_indices)//3:2*len(fraud_indices)//3]
            df.loc[device_fraud, 'device_id'] = 'DEV_FRAUD_PATTERN'
            df.loc[device_fraud, 'valor'] = np.random.uniform(500, 2000, len(device_fraud))
            
            # Fraudes geográficas (localizações improváveis)
            geo_fraud = fraud_indices[2*len(fraud_indices)//3:]
            df.loc[geo_fraud, 'latitude'] = np.random.uniform(40, 60, len(geo_fraud))  # Fora do Brasil
            df.loc[geo_fraud, 'longitude'] = np.random.uniform(0, 30, len(geo_fraud))
            df.loc[geo_fraud, 'valor'] = np.random.uniform(1000, 8000, len(geo_fraud))
        
        return df
    
    # Gerar e treinar
    df_data = generate_realistic_fraud_data(num_samples=30000, fraud_ratio=0.03)
    X = df_data.drop(columns=['is_fraud']).to_dict(orient='records')
    y = df_data['is_fraud'].astype(int).tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Treinar motor V4.0
    engine = UltraPrecisionFraudEngineV4()
    engine.train(X_train, y_train)
    
    # Avaliar
    logging.info("Avaliando o motor Ultra-Precision V4.0...")
    start_eval_time = time.time()
    
    y_pred_proba = engine.predict_proba(X_test)[:, 1]
    y_pred_standard = engine.predict(X_test, use_precision_threshold=False)
    y_pred_precision = engine.predict(X_test, use_precision_threshold=True)
    
    end_eval_time = time.time()
    
    # Métricas com threshold padrão
    accuracy_std = accuracy_score(y_test, y_pred_standard)
    precision_std = precision_score(y_test, y_pred_standard, zero_division=0)
    recall_std = recall_score(y_test, y_pred_standard, zero_division=0)
    f1_std = f1_score(y_test, y_pred_standard, zero_division=0)
    
    # Métricas com threshold de alta precision
    accuracy_prec = accuracy_score(y_test, y_pred_precision)
    precision_prec = precision_score(y_test, y_pred_precision, zero_division=0)
    recall_prec = recall_score(y_test, y_pred_precision, zero_division=0)
    f1_prec = f1_score(y_test, y_pred_precision, zero_division=0)
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    logging.info("=== MÉTRICAS THRESHOLD PADRÃO ===")
    logging.info(f"Threshold: {engine.threshold:.4f}")
    logging.info(f"Accuracy: {accuracy_std:.4f}")
    logging.info(f"Precision: {precision_std:.4f}")
    logging.info(f"Recall: {recall_std:.4f}")
    logging.info(f"F1-Score: {f1_std:.4f}")
    
    logging.info("=== MÉTRICAS THRESHOLD ALTA PRECISION ===")
    logging.info(f"Threshold: {engine.precision_threshold:.4f}")
    logging.info(f"Accuracy: {accuracy_prec:.4f}")
    logging.info(f"Precision: {precision_prec:.4f}")
    logging.info(f"Recall: {recall_prec:.4f}")
    logging.info(f"F1-Score: {f1_prec:.4f}")
    
    logging.info(f"AUC-ROC: {roc_auc:.4f}")
    logging.info(f"Tempo de avaliação: {end_eval_time - start_eval_time:.4f} segundos")
    
    # Teste de transação individual
    sample_transaction = Transaction(**X_test[0])
    analysis_result = engine.analyze_transaction(sample_transaction)
    logging.info(f"Análise de transação: {json.dumps(analysis_result, indent=2)}")
    
    logging.info("Teste do UltraPrecisionFraudEngineV4 concluído.")
