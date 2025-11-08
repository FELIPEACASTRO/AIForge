import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class RealFraudModelTrainer:
    """
    Sistema de treinamento de modelos REAIS para detecção de fraude bancária.
    Elimina completamente simulações e mocks, usando dados reais e algoritmos de produção.
    """
    
    def __init__(self, data_path="/home/ubuntu/sankofa-enterprise-real/backend/data/real_banking_dataset.csv"):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.is_trained = False
        print("Sistema de Treinamento de Modelos REAIS inicializado.")
    
    def load_and_prepare_data(self):
        """Carrega e prepara os dados reais para treinamento."""
        print("Carregando dataset bancário real...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset carregado: {len(self.df)} transações")
        
        # Converter timestamp para features temporais
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        self.df['is_night'] = ((self.df['hour'] >= 22) | (self.df['hour'] <= 6)).astype(int)
        
        # Features de valor
        self.df['valor_log'] = np.log1p(self.df['valor'])
        self.df['valor_zscore'] = (self.df['valor'] - self.df['valor'].mean()) / self.df['valor'].std()
        
        # Encoding de variáveis categóricas
        categorical_cols = ['tipo_transacao', 'canal', 'estado']
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
            self.encoders[col] = le
        
        # Features de localização (simplificadas)
        self.df['latitude'] = pd.to_numeric(self.df['latitude'], errors='coerce')
        self.df['longitude'] = pd.to_numeric(self.df['longitude'], errors='coerce')
        self.df['latitude'].fillna(self.df['latitude'].mean(), inplace=True)
        self.df['longitude'].fillna(self.df['longitude'].mean(), inplace=True)
        
        # Selecionar features para treinamento
        self.feature_names = [
            'valor', 'valor_log', 'valor_zscore',
            'hour', 'day_of_week', 'is_weekend', 'is_night',
            'tipo_transacao_encoded', 'canal_encoded', 'estado_encoded',
            'latitude', 'longitude'
        ]
        
        self.X = self.df[self.feature_names]
        self.y = self.df['is_fraud']
        
        print(f"Features preparadas: {len(self.feature_names)}")
        print(f"Distribuição de classes - Legítimas: {(self.y == 0).sum()}, Fraudes: {(self.y == 1).sum()}")
        
        return self.X, self.y
    
    def train_models(self):
        """Treina ensemble de modelos reais com validação temporal."""
        print("\nIniciando treinamento de modelos REAIS...")
        
        # Split temporal para evitar data leakage
        self.df_sorted = self.df.sort_values('timestamp')
        split_idx = int(len(self.df_sorted) * 0.8)
        
        train_data = self.df_sorted.iloc[:split_idx]
        test_data = self.df_sorted.iloc[split_idx:]
        
        X_train = train_data[self.feature_names]
        y_train = train_data['is_fraud']
        X_test = test_data[self.feature_names]
        y_test = test_data['is_fraud']
        
        # Normalização
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Definir modelos
        models_config = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        
        # Treinar cada modelo
        self.trained_models = {}
        self.model_scores = {}
        
        for name, model in models_config.items():
            print(f"\nTreinando {name}...")
            
            if name in ['logistic_regression', 'neural_network']:
                # Modelos que precisam de dados normalizados
                model.fit(X_train_scaled, y_train)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                # Modelos baseados em árvore
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calibrar probabilidades
            if name in ['logistic_regression', 'neural_network']:
                calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                calibrated_model.fit(X_train_scaled, y_train)
                y_pred_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
                self.trained_models[name] = calibrated_model
            else:
                calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                calibrated_model.fit(X_train, y_train)
                y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
                self.trained_models[name] = calibrated_model
            
            # Avaliar modelo
            auc_score = roc_auc_score(y_test, y_pred_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            auprc = auc(recall, precision)
            
            self.model_scores[name] = {
                'auc': auc_score,
                'auprc': auprc
            }
            
            print(f"{name} - AUC: {auc_score:.4f}, AUPRC: {auprc:.4f}")
        
        # Criar ensemble
        self._create_ensemble(X_test, X_test_scaled, y_test)
        
        # Salvar modelos e preprocessadores
        self._save_models()
        
        self.is_trained = True
        print("\n✅ Treinamento de modelos REAIS concluído com sucesso!")
        
        return self.model_scores
    
    def _create_ensemble(self, X_test, X_test_scaled, y_test):
        """Cria ensemble ponderado baseado na performance dos modelos."""
        print("\nCriando ensemble ponderado...")
        
        # Coletar predições de todos os modelos
        ensemble_predictions = []
        weights = []
        
        for name, model in self.trained_models.items():
            if name in ['logistic_regression', 'neural_network']:
                pred = model.predict_proba(X_test_scaled)[:, 1]
            else:
                pred = model.predict_proba(X_test)[:, 1]
            
            ensemble_predictions.append(pred)
            weights.append(self.model_scores[name]['auprc'])  # Usar AUPRC como peso
        
        # Normalizar pesos
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Criar predição ensemble
        ensemble_pred = np.average(ensemble_predictions, axis=0, weights=weights)
        
        # Avaliar ensemble
        ensemble_auc = roc_auc_score(y_test, ensemble_pred)
        precision, recall, _ = precision_recall_curve(y_test, ensemble_pred)
        ensemble_auprc = auc(recall, precision)
        
        self.ensemble_weights = dict(zip(self.trained_models.keys(), weights))
        self.model_scores['ensemble'] = {
            'auc': ensemble_auc,
            'auprc': ensemble_auprc
        }
        
        print(f"Ensemble - AUC: {ensemble_auc:.4f}, AUPRC: {ensemble_auprc:.4f}")
        print(f"Pesos do ensemble: {self.ensemble_weights}")
    
    def _save_models(self):
        """Salva todos os modelos e preprocessadores treinados."""
        model_dir = "/home/ubuntu/sankofa-enterprise-real/backend/ml_engine/trained_models/"
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Salvar modelos
        for name, model in self.trained_models.items():
            joblib.dump(model, f"{model_dir}{name}_model.joblib")
        
        # Salvar preprocessadores
        joblib.dump(self.scaler, f"{model_dir}scaler.joblib")
        joblib.dump(self.encoders, f"{model_dir}encoders.joblib")
        joblib.dump(self.feature_names, f"{model_dir}feature_names.joblib")
        joblib.dump(self.ensemble_weights, f"{model_dir}ensemble_weights.joblib")
        
        # Salvar metadados
        metadata = {
            'is_trained': True,
            'model_scores': self.model_scores,
            'feature_names': self.feature_names,
            'training_date': pd.Timestamp.now().isoformat()
        }
        joblib.dump(metadata, f"{model_dir}metadata.joblib")
        
        print(f"Modelos salvos em: {model_dir}")
    
    def predict_fraud(self, transaction_data):
        """Faz predição de fraude usando o ensemble treinado."""
        if not self.is_trained:
            raise ValueError("Modelos não foram treinados ainda!")
        
        # Preparar dados da transação
        df_trans = pd.DataFrame([transaction_data])
        
        # Aplicar mesmo preprocessing do treinamento
        df_trans['timestamp'] = pd.to_datetime(df_trans['timestamp'])
        df_trans['hour'] = df_trans['timestamp'].dt.hour
        df_trans['day_of_week'] = df_trans['timestamp'].dt.dayofweek
        df_trans['is_weekend'] = (df_trans['day_of_week'] >= 5).astype(int)
        df_trans['is_night'] = ((df_trans['hour'] >= 22) | (df_trans['hour'] <= 6)).astype(int)
        
        df_trans['valor_log'] = np.log1p(df_trans['valor'])
        df_trans['valor_zscore'] = (df_trans['valor'] - self.df['valor'].mean()) / self.df['valor'].std()
        
        # Encoding categórico
        for col in ['tipo_transacao', 'canal', 'estado']:
            if col in self.encoders:
                try:
                    df_trans[f'{col}_encoded'] = self.encoders[col].transform(df_trans[col])
                except ValueError:
                    # Valor não visto no treinamento, usar valor padrão
                    df_trans[f'{col}_encoded'] = 0
        
        df_trans['latitude'] = pd.to_numeric(df_trans['latitude'], errors='coerce')
        df_trans['longitude'] = pd.to_numeric(df_trans['longitude'], errors='coerce')
        df_trans['latitude'].fillna(self.df['latitude'].mean(), inplace=True)
        df_trans['longitude'].fillna(self.df['longitude'].mean(), inplace=True)
        
        # Extrair features
        X_trans = df_trans[self.feature_names]
        X_trans_scaled = self.scaler.transform(X_trans)
        
        # Fazer predições com todos os modelos
        predictions = []
        for name, model in self.trained_models.items():
            if name in ['logistic_regression', 'neural_network']:
                pred = model.predict_proba(X_trans_scaled)[:, 1][0]
            else:
                pred = model.predict_proba(X_trans)[:, 1][0]
            predictions.append(pred)
        
        # Calcular predição ensemble
        ensemble_pred = np.average(predictions, weights=list(self.ensemble_weights.values()))
        
        return {
            'fraud_probability': float(ensemble_pred),
            'is_fraud': bool(ensemble_pred > 0.5),
            'confidence': float(abs(ensemble_pred - 0.5) * 2),
            'individual_predictions': dict(zip(self.trained_models.keys(), predictions))
        }

if __name__ == '__main__':
    print("Iniciando treinamento de modelos REAIS de detecção de fraude...")
    
    trainer = RealFraudModelTrainer()
    
    # Carregar e preparar dados
    X, y = trainer.load_and_prepare_data()
    
    # Treinar modelos
    scores = trainer.train_models()
    
    print("\n--- Resumo Final dos Modelos ---")
    for model_name, metrics in scores.items():
        print(f"{model_name}: AUC={metrics['auc']:.4f}, AUPRC={metrics['auprc']:.4f}")
    
    # Teste de predição
    print("\n--- Teste de Predição ---")
    test_transaction = {
        'valor': 5000.0,
        'tipo_transacao': 'PIX',
        'canal': 'mobile',
        'estado': 'SP',
        'timestamp': '2023-12-01T02:30:00',
        'latitude': '-23.5505',
        'longitude': '-46.6333'
    }
    
    result = trainer.predict_fraud(test_transaction)
    print(f"Transação teste: {test_transaction}")
    print(f"Resultado: {result}")
    
    print("\n✅ Sistema de modelos REAIS implementado com sucesso!")
