import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
import time
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

from backend.models.transaction_model import Transaction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HyperOptimizedFraudEngineV3:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'rf': RandomForestClassifier(random_state=self.random_state, n_estimators=200, max_depth=10, class_weight='balanced'),
            'gb': GradientBoostingClassifier(random_state=self.random_state, n_estimators=200, learning_rate=0.05, max_depth=5),
            'lr': LogisticRegression(random_state=self.random_state, solver='liblinear', class_weight='balanced')
        }
        self.meta_model = LogisticRegression(random_state=self.random_state, solver='liblinear')
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=self.random_state)
        self.feature_names = []
        self.is_trained = False
        self.threshold = 0.5 # Default threshold

    def _preprocess(self, df):
        # Feature Engineering
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df["day_of_month"] = df["timestamp"].dt.day

        # Drop original timestamp and identifier columns early
        df = df.drop(columns=["timestamp", "id", "cliente_cpf", "ip_address"], errors='ignore')
        # Interaction features
        df["valor_per_hour"] = df["valor"] / (df["hour"] + 1) # Avoid division by zero

        # Categorical features encoding (one-hot for simplicity, could use target encoding for better performance)
        categorical_cols = ["tipo_transacao", "canal", "cidade", "estado", "pais", "device_id", "conta_recebedor"]
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


        # Select numerical features for scaling and imputation
        # Start with a base list of numerical columns
        base_numerical_cols = ["valor", "hour", "day_of_week", "month", "day_of_month", "valor_per_hour", "latitude", "longitude"]
        
        # Get all numerical columns after one-hot encoding
        current_numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        # Combine base numerical columns with new dummy numerical columns
        final_numerical_cols = list(set(base_numerical_cols) & set(current_numerical_cols))
        for col in current_numerical_cols:
            if col not in final_numerical_cols:
                final_numerical_cols.append(col)

        # Exclude any non-numeric or identifier columns that might have slipped in
        cols_to_exclude = ["id", "cliente_cpf", "ip_address"]
        final_numerical_cols = [col for col in final_numerical_cols if col not in cols_to_exclude]

        # Ensure all columns in final_numerical_cols actually exist in df before processing
        final_numerical_cols = [col for col in final_numerical_cols if col in df.columns]

        # Imputation and Scaling
        if final_numerical_cols:
            df[final_numerical_cols] = self.imputer.fit_transform(df[final_numerical_cols])
            df[final_numerical_cols] = self.scaler.fit_transform(df[final_numerical_cols])
            df[final_numerical_cols] = self.quantile_transformer.fit_transform(df[final_numerical_cols])

        return df

    def train(self, X_train, y_train):
        logging.info("Iniciando treinamento do motor Hyper-Optimized V3.0...")
        start_time = time.time()
        df_train = pd.DataFrame(X_train)
        df_train["is_fraud"] = y_train

        # Drop identifier and original timestamp columns before preprocessing
        df_train_for_preprocess = df_train.drop(columns=["id", "cliente_cpf", "ip_address", "is_fraud"], errors='ignore')
        X_processed = self._preprocess(df_train_for_preprocess) # X_processed is now a DataFrame
        y_processed = df_train["is_fraud"]
        self.feature_names = X_processed.columns.tolist()

        # Train base models
        oof_preds = np.zeros((len(X_processed), len(self.models)))
        for i, (name, model) in enumerate(self.models.items()):
            logging.info(f"Treinando modelo base: {name}")
            model.fit(X_processed, y_processed)
            oof_preds[:, i] = model.predict_proba(X_processed)[:, 1]

        # Train meta-model
        logging.info("Treinando meta-modelo...")
        self.meta_model.fit(oof_preds, y_processed)

        self.is_trained = True
        end_time = time.time()
        logging.info(f"Treinamento concluído em {end_time - start_time:.2f} segundos.")

        # Dynamic threshold calibration (maximize F1-score on training data)
        self.calibrate_threshold(X_processed, y_processed)

    def calibrate_threshold(self, X_data_processed, y_data):
        logging.info("Calibrando threshold para maximizar F1-Score...")
        # Pass the already processed X_data_processed to predict_proba
        probabilities = self.predict_proba(X_data_processed)[:, 1]
        best_f1 = -1
        best_threshold = 0.5

        for t in np.linspace(0.01, 0.99, 100):
            y_pred = (probabilities >= t).astype(int)
            f1 = f1_score(y_data, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        self.threshold = best_threshold
        logging.info(f"Threshold calibrado: {self.threshold:.4f} (F1-Score: {best_f1:.4f})")

    def predict_proba(self, X_input):
        # If X_input is already a processed DataFrame (e.g., from calibrate_threshold),
        # use it directly. Otherwise, preprocess it.
        if isinstance(X_input, pd.DataFrame) and all(col in self.feature_names for col in X_input.columns):
            X_processed = X_input
        else:
            if isinstance(X_input, Transaction):
                df_test = pd.DataFrame([X_input.__dict__])
            elif isinstance(X_input, list) and len(X_input) > 0 and isinstance(X_input[0], dict):
                df_test = pd.DataFrame(X_input)
            else:
                df_test = pd.DataFrame([X_input.__dict__] if hasattr(X_input, '__dict__') else [X_input])

            # Drop identifier columns before preprocessing
            df_test = df_test.drop(columns=["id", "cliente_cpf", "ip_address"], errors='ignore')
            X_processed = self._preprocess(df_test)

        # Ensure columns are aligned with training data
        missing_cols = set(self.feature_names) - set(X_processed.columns)
        for c in missing_cols:
            X_processed[c] = 0
        X_processed = X_processed.copy()[self.feature_names] # Use .copy() to avoid fragmentation warning

        base_preds = np.zeros((len(X_processed), len(self.models)))
        for i, (name, model) in enumerate(self.models.items()):
            base_preds[:, i] = model.predict_proba(X_processed)[:, 1]
        return self.meta_model.predict_proba(base_preds)

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold).astype(int)

    def analyze_transaction(self, transaction: Transaction) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("O motor não foi treinado. Por favor, treine o motor antes de analisar transações.")

        df_single = pd.DataFrame([transaction.__dict__])
        
        # Ensure all expected columns are present, even if empty for a single transaction
        for col in ['tipo_transacao', 'canal', 'cidade', 'estado', 'pais', 'device_id', 'conta_recebedor']:
            if col not in df_single.columns:
                df_single[col] = ''
        
        # Drop identifier columns before preprocessing
        df_single = df_single.drop(columns=["id", "cliente_cpf", "ip_address"], errors='ignore')

        # Preprocess the single transaction
        X_processed = self._preprocess(df_single.drop(columns=["is_fraud", "fraud_score"], errors='ignore'))

        # Ensure columns are aligned with training data
        missing_cols = set(self.feature_names) - set(X_processed.columns)
        for c in missing_cols:
            X_processed[c] = 0
        X_processed = X_processed.copy()[self.feature_names] # Use .copy() to avoid fragmentation warning

        # Predict fraud score
        fraud_probability = self.predict_proba(X_processed)[:, 1][0]
        is_fraud_prediction = (fraud_probability >= self.threshold).astype(bool)

        # Determine risk level based on calibrated thresholds
        risk_level = "BAIXO"
        if fraud_probability >= self.threshold:
            risk_level = "ALTO"
        elif fraud_probability >= (self.threshold * 0.7): # Example of a medium risk zone
            risk_level = "MÉDIO"

        return {
            "transaction_id": transaction.id,
            "fraud_score": float(f"{fraud_probability:.4f}"),
            "is_fraud_prediction": bool(is_fraud_prediction),
            "risk_level": risk_level,
            "analysis_details": "Análise baseada em ensemble de modelos e calibração dinâmica de threshold."
        }

# --- Teste de Exemplo e Validação --- 
if __name__ == "__main__":
    logging.info("Iniciando teste do HyperOptimizedFraudEngineV3...")

    # Gerar dados sintéticos para treinamento e teste
    def generate_synthetic_data(num_samples=10000, fraud_ratio=0.05):
        data = {
            'id': [f'TXN_{i}' for i in range(num_samples)],
            'valor': np.random.lognormal(mean=7, sigma=1.5, size=num_samples),
            'tipo_transacao': np.random.choice(['PIX', 'DEBITO', 'CREDITO', 'TED', 'DOC'], num_samples, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
            'canal': np.random.choice(['MOBILE', 'WEB', 'POS', 'ATM'], num_samples, p=[0.5, 0.3, 0.1, 0.1]),
            'cidade': np.random.choice(['Sao Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Porto Alegre', 'Curitiba'], num_samples),
            'estado': np.random.choice(['SP', 'RJ', 'MG', 'RS', 'PR'], num_samples),
            'pais': ['BR'] * num_samples,
            'ip_address': [f'192.168.1.{np.random.randint(1, 255)}' for _ in range(num_samples)],
            'device_id': [f'DEV_{np.random.randint(1, 100)}' for _ in range(num_samples)],
            'conta_recebedor': [f'REC_{np.random.randint(1, 500)}' for _ in range(num_samples)],
            'cliente_cpf': [f'CPF_{np.random.randint(10000000000, 99999999999)}' for _ in range(num_samples)],
            'timestamp': pd.to_datetime(pd.date_range(start='2024-01-01', periods=num_samples, freq='min')).astype(str).tolist(),
            'latitude': np.random.uniform(-30, -10, num_samples),
            'longitude': np.random.uniform(-60, -40, num_samples),
            'is_fraud': np.random.choice([True, False], num_samples, p=[fraud_ratio, 1-fraud_ratio])
        }
        df = pd.DataFrame(data)

        # Inject some fraud patterns for realism
        fraud_indices = df[df['is_fraud'] == True].index
        if not fraud_indices.empty:
            half_fraud_len = len(fraud_indices) // 2
            # High value transactions from unusual locations
            df.loc[fraud_indices[:half_fraud_len], 'valor'] = np.random.uniform(5000, 50000, half_fraud_len)
            df.loc[fraud_indices[:half_fraud_len], 'latitude'] = np.random.uniform(30, 50, half_fraud_len)
            df.loc[fraud_indices[:half_fraud_len], 'longitude'] = np.random.uniform(10, 30, half_fraud_len)
            # Multiple transactions from same device in short time
            df.loc[fraud_indices[half_fraud_len:], 'device_id'] = 'DEV_FRAUD_001'
            df.loc[fraud_indices[half_fraud_len:], 'valor'] = np.random.uniform(100, 1000, len(fraud_indices) - half_fraud_len)

        return df

    df_data = generate_synthetic_data(num_samples=50000, fraud_ratio=0.02)
    X = df_data.drop(columns=['is_fraud']).to_dict(orient='records')
    y = df_data['is_fraud'].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    engine = HyperOptimizedFraudEngineV3()
    engine.train(X_train, y_train)

    # Avaliar o motor
    logging.info("Avaliando o motor Hyper-Optimized V3.0 no conjunto de teste...")
    start_eval_time = time.time()
    y_pred_proba = engine.predict_proba(X_test)[:, 1]
    y_pred = engine.predict(X_test)
    end_eval_time = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    logging.info(f"Métricas de Avaliação (Threshold: {engine.threshold:.4f}):")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall: {recall:.4f}")
    logging.info(f"  F1-Score: {f1:.4f}")
    logging.info(f"  AUC-ROC: {roc_auc:.4f}")
    logging.info(f"Tempo de avaliação: {end_eval_time - start_eval_time:.4f} segundos.")

    # Testar análise de transação única
    logging.info("\nTestando análise de transação única...")
    sample_transaction_data = X_test[0]
    sample_transaction = Transaction(**sample_transaction_data)
    analysis_result = engine.analyze_transaction(sample_transaction)
    logging.info(f"Resultado da análise para TXN_{sample_transaction.id}: {json.dumps(analysis_result, indent=2)}")

    # Testar uma transação fraudulenta (simulada)
    logging.info("\nTestando transação fraudulenta simulada...")
    fraud_transaction = Transaction(
        id="TXN_FRAUD_001",
        valor=15000.00,
        tipo_transacao="PIX",
        canal="WEB",
        cidade="Sao Paulo",
        estado="SP",
        pais="BR",
        ip_address="192.168.1.10",
        device_id="DEV_FRAUD_001",
        conta_recebedor="REC_FRAUD_001",
        cliente_cpf="CPF_11122233344",
        timestamp="2025-09-21T10:00:00Z",
        latitude=40.7128,
        longitude=-74.0060,
        is_fraud=True
    )
    fraud_analysis_result = engine.analyze_transaction(fraud_transaction)
    logging.info(f"Resultado da análise para TXN_FRAUD_001: {json.dumps(fraud_analysis_result, indent=2)}")

    # Testar uma transação legítima (simulada)
    logging.info("\nTestando transação legítima simulada...")
    legit_transaction = Transaction(
        id="TXN_LEGIT_001",
        valor=150.00,
        tipo_transacao="DEBITO",
        canal="POS",
        cidade="Belo Horizonte",
        estado="MG",
        pais="BR",
        ip_address="192.168.1.50",
        device_id="DEV_LEGIT_001",
        conta_recebedor="REC_LEGIT_001",
        cliente_cpf="CPF_55566677788",
        timestamp="2025-09-21T11:00:00Z",
        latitude=-19.9167,
        longitude=-43.9345,
        is_fraud=False
    )
    legit_analysis_result = engine.analyze_transaction(legit_transaction)
    logging.info(f"Resultado da análise para TXN_LEGIT_001: {json.dumps(legit_analysis_result, indent=2)}")

    logging.info("Teste do HyperOptimizedFraudEngineV3 concluído.")
