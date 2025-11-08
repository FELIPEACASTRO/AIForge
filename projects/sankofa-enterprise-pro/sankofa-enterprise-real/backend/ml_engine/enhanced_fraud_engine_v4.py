
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from joblib import dump, load
import os
import logging
from scipy.sparse import vstack

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Adiciona o diretório 'backend/data' ao PYTHONPATH
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

from external_dataset_integration import ExternalDatasetIntegration
from brazilian_synthetic_data_generator import generate_brazilian_transactions
from transaction_model import Transaction

class EnhancedFraudEngineV4:
    def __init__(self, random_state=42, sample_size_per_class=5000, oversampling_strategy='smote'):
        self.random_state = random_state
        self.sample_size_per_class = sample_size_per_class
        self.oversampling_strategy = oversampling_strategy
        self.preprocessor = None
        self.model_pipeline = None
        self.optimal_threshold = 0.5 # Default threshold
        self.numerical_features = []
        self.categorical_features = []

    def _create_preprocessor(self, X_sample):
        # Identificar features numéricas e categóricas do DataFrame de amostra
        numeric_features = X_sample.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X_sample.select_dtypes(include='object').columns.tolist()

        # Remover 'isFraud' e 'isFlaggedFraud' se estiverem nas features numéricas
        if 'isFraud' in numeric_features: numeric_features.remove('isFraud')
        if 'isFlaggedFraud' in numeric_features: numeric_features.remove('isFlaggedFraud')

        self.numerical_features = sorted(list(set(numeric_features)))
        self.categorical_features = sorted(list(set(categorical_features)))

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ], 
            remainder='passthrough' # Manter colunas não transformadas
        )

    def _apply_balancing_strategy(self, X_train_processed, y_train):
        if self.oversampling_strategy == 'smote':
            logging.info("Aplicando SMOTE para balanceamento de classes.")
            # Ajustar sampling_strategy para 'auto' ou um valor que permita oversampling
            smote = SMOTE(random_state=self.random_state, sampling_strategy='auto')
            X_res, y_res = smote.fit_resample(X_train_processed, y_train)
        elif self.oversampling_strategy == 'adasyn':
            logging.info("Aplicando ADASYN para balanceamento de classes.")
            adasyn = ADASYN(random_state=self.random_state, sampling_strategy='auto')
            X_res, y_res = adasyn.fit_resample(X_train_processed, y_train)
        elif self.oversampling_strategy == 'borderline_smote':
            logging.info("Aplicando BorderlineSMOTE para balanceamento de classes.")
            bsmote = BorderlineSMOTE(random_state=self.random_state, sampling_strategy='auto')
            X_res, y_res = bsmote.fit_resample(X_train_processed, y_train)
        elif self.oversampling_strategy == 'smoteenn':
            logging.info("Aplicando SMOTEENN para balanceamento de classes.")
            smoteenn = SMOTEENN(random_state=self.random_state, sampling_strategy='auto')
            X_res, y_res = smoteenn.fit_resample(X_train_processed, y_train)
        elif self.oversampling_strategy == 'smotetomek':
            logging.info("Aplicando SMOTETomek para balanceamento de classes.")
            smotetomek = SMOTETomek(random_state=self.random_state, sampling_strategy='auto')
            X_res, y_res = smotetomek.fit_resample(X_train_processed, y_train)
        else:
            logging.info("Nenhuma estratégia de oversampling aplicada.")
            X_res, y_res = X_train_processed, y_train
        
        return X_res, y_res

    def _stratified_sample_df(self, df, sample_size_per_class, random_state):
        if 'isFraud' not in df.columns:
            return df # Cannot sample if target column is missing

        df_fraud = df[df['isFraud'] == 1]
        df_non_fraud = df[df['isFraud'] == 0]

        # Sample fraud cases (take all if less than sample_size_per_class)
        if len(df_fraud) > 0:
            n_fraud_samples = min(len(df_fraud), sample_size_per_class)
            df_fraud_sampled = df_fraud.sample(n=n_fraud_samples, random_state=random_state)
        else:
            df_fraud_sampled = pd.DataFrame(columns=df.columns) # Empty DataFrame

        # Sample non-fraud cases (to maintain a reasonable ratio, e.g., 1:10 or 1:20 with fraud)
        # Adjust this ratio based on desired balance for initial training
        n_non_fraud_samples = min(len(df_non_fraud), sample_size_per_class * 10) # Example: 10x non-fraud for each fraud sample
        df_non_fraud_sampled = df_non_fraud.sample(n=n_non_fraud_samples, random_state=random_state)

        sampled_df = pd.concat([df_fraud_sampled, df_non_fraud_sampled])
        logging.info(f"  Amostragem estratificada: {len(df_fraud_sampled)} fraudes, {len(df_non_fraud_sampled)} não-fraudes.")
        return sampled_df

    def train_model(self, datasets_dict):
        logging.info("Iniciando o treinamento do modelo.")
        all_data = []
        
        # Primeiro, coletar uma amostra representativa de todos os datasets para criar o preprocessor
        # Isso garante que o preprocessor seja ajustado a todas as features possíveis
        sample_for_preprocessor = []
        for name, df in datasets_dict.items():
            if 'isFraud' not in df.columns:
                continue
            # Pegar uma pequena amostra de cada DF para o preprocessor
            sample_for_preprocessor.append(self._stratified_sample_df(df, sample_size_per_class=100, random_state=self.random_state))
        
        if not sample_for_preprocessor:
            logging.error("Nenhum dado válido para criar o preprocessor.")
            return

        combined_sample_for_preprocessor = pd.concat(sample_for_preprocessor, ignore_index=True)
        self._create_preprocessor(combined_sample_for_preprocessor.drop(columns=['isFraud'], errors='ignore'))
        logging.info("Preprocessor criado com sucesso.")

        # Agora, processar e amostrar os datasets para o treinamento real
        for name, df in datasets_dict.items():
            logging.info(f"Processando dataset: {name}")
            if 'isFraud' not in df.columns:
                logging.warning(f"Dataset {name} não possui a coluna 'isFraud'. Ignorando.")
                continue
            
            df_copy = df.copy()
            if 'isFlaggedFraud' in df_copy.columns:
                df_copy = df_copy.drop(columns=['isFlaggedFraud'])
            
            # Amostragem estratificada para lidar com desbalanceamento e memória
            sampled_df = self._stratified_sample_df(df_copy, self.sample_size_per_class, self.random_state)
            all_data.append(sampled_df)

        if not all_data:
            logging.error("Nenhum dado válido para treinamento após o processamento dos datasets.")
            return

        combined_df = pd.concat(all_data, ignore_index=True)
        logging.info(f"Total de amostras combinadas para treinamento: {len(combined_df)}")

        X = combined_df.drop(columns=['isFraud'])
        y = combined_df['isFraud']

        # Garantir que X tenha as mesmas colunas que o preprocessor foi ajustado
        # Adicionar colunas ausentes e remover extras para corresponder ao preprocessor
        expected_features = self.numerical_features + self.categorical_features
        for col in expected_features:
            if col not in X.columns:
                if col in self.numerical_features:
                    X[col] = 0.0
                else:
                    X[col] = 'missing_category'
        X = X[expected_features] # Reordenar colunas

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)

        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        # Aplicar estratégia de balanceamento
        X_train_res, y_train_res = self._apply_balancing_strategy(X_train_processed, y_train)
        logging.info(f"Shape de X_train_res após balanceamento: {X_train_res.shape}")
        logging.info(f"Contagem de classes de y_train_res após balanceamento:\n{pd.Series(y_train_res).value_counts()}")

        # Treinar modelos base
        logging.info("Treinando modelos base...")
        oof_preds = np.zeros((len(X_train_res), len(self.base_models)))
        test_preds = np.zeros((len(X_test_processed), len(self.base_models)))

        for i, (name, model) in enumerate(self.base_models.items()):
            logging.info(f"Treinando {name}...")
            model.fit(X_train_res, y_train_res)
            oof_preds[:, i] = model.predict_proba(X_train_res)[:, 1]
            test_preds[:, i] = model.predict_proba(X_test_processed)[:, 1]

        # Treinar meta-modelo
        logging.info("Treinando meta-modelo...")
        self.meta_model.fit(oof_preds, y_train_res)

        # Avaliar o modelo final no conjunto de teste
        final_predictions_proba = self.meta_model.predict_proba(test_preds)[:, 1]
        
        # Encontrar o limiar ótimo para precisão >= 0.999
        precisions, recalls, thresholds = precision_recall_curve(y_test, final_predictions_proba)
        
        # Filtrar thresholds onde a precisão é >= 0.999
        high_precision_thresholds = thresholds[precisions[:-1] >= 0.999]
        
        if len(high_precision_thresholds) > 0:
            # Escolher o threshold que maximiza o recall entre aqueles com alta precisão
            optimal_threshold_idx = np.argmax(recalls[:-1][precisions[:-1] >= 0.999])
            self.optimal_threshold = high_precision_thresholds[optimal_threshold_idx]
            logging.info(f"Limiar de decisão ótimo para precisão >= 0.999: {self.optimal_threshold:.4f}")
        else:
            logging.warning("Não foi possível encontrar um limiar com precisão >= 0.999. Usando limiar padrão de 0.5.")
            self.optimal_threshold = 0.5

        final_predictions = (final_predictions_proba >= self.optimal_threshold).astype(int)

        accuracy = accuracy_score(y_test, final_predictions)
        precision = precision_score(y_test, final_predictions)
        recall = recall_score(y_test, final_predictions)
        f1 = f1_score(y_test, final_predictions)
        roc_auc = roc_auc_score(y_test, final_predictions_proba)

        logging.info(f"Métricas do Modelo Final:")
        logging.info(f"  Acurácia: {accuracy:.4f}")
        logging.info(f"  Precisão: {precision:.4f}")
        logging.info(f"  Recall: {recall:.4f}")
        logging.info(f"  F1-Score: {f1:.4f}")
        logging.info(f"  ROC AUC: {roc_auc:.4f}")

        self.model_pipeline = {
            'preprocessor': self.preprocessor, 
            'base_models': self.base_models, 
            'meta_model': self.meta_model,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
        }
        logging.info("Treinamento do modelo concluído.")
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'optimal_threshold': self.optimal_threshold
        }

    def predict_proba(self, X_new):
        if self.model_pipeline is None:
            raise Exception("Modelo não treinado. Por favor, treine o modelo primeiro.")
        
        # Certificar que X_new é um DataFrame
        if not isinstance(X_new, pd.DataFrame):
            X_new = pd.DataFrame([X_new])

        # Garantir que X_new tenha as mesmas colunas que o preprocessor foi ajustado
        expected_features = self.model_pipeline['numerical_features'] + self.model_pipeline['categorical_features']
        for col in expected_features:
            if col not in X_new.columns:
                if col in self.model_pipeline['numerical_features']:
                    X_new[col] = 0.0
                else:
                    X_new[col] = 'missing_category'
        X_new = X_new[expected_features] # Reordenar colunas

        X_new_processed = self.model_pipeline['preprocessor'].transform(X_new)
        
        base_preds = np.zeros((len(X_new_processed), len(self.model_pipeline['base_models'])))
        for i, (name, model) in enumerate(self.model_pipeline['base_models'].items()):
            base_preds[:, i] = model.predict_proba(X_new_processed)[:, 1]
        
        return self.model_pipeline['meta_model'].predict_proba(base_preds)[:, 1]

    def predict(self, X_new):
        probabilities = self.predict_proba(X_new)
        return (probabilities >= self.optimal_threshold).astype(int)

    def save_model(self, path="fraud_detection_model.joblib"):
        if self.model_pipeline is None:
            raise Exception("Modelo não treinado. Não há modelo para salvar.")
        dump(self.model_pipeline, path)
        logging.info(f"Modelo salvo em {path}")

    def load_model(self, path="fraud_detection_model.joblib"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo não encontrado em {path}")
        self.model_pipeline = load(path)
        self.preprocessor = self.model_pipeline['preprocessor']
        self.numerical_features = self.model_pipeline['numerical_features']
        self.categorical_features = self.model_pipeline['categorical_features']
        self.base_models = self.model_pipeline['base_models']
        self.meta_model = self.model_pipeline['meta_model']
        logging.info(f"Modelo carregado de {path}")

# Exemplo de uso (para teste)
if __name__ == "__main__":
    # Simular dados de datasets
    # Em um ambiente real, você carregaria os dados de external_dataset_integration.py
    # e brazilian_synthetic_data_generator.py
    
    # Dataset 1: creditcard_kaggle (simulado)
    data_kaggle = {
        'V1': np.random.rand(10000) * 10 - 5,
        'V2': np.random.rand(10000) * 10 - 5,
        'V3': np.random.rand(10000) * 10 - 5,
        'Amount': np.random.rand(10000) * 1000,
        'Time': np.arange(10000),
        'isFraud': np.random.randint(0, 2, 10000)
    }
    # Aumentar um pouco a proporção de fraude para o exemplo
    data_kaggle['isFraud'][np.random.choice(10000, 50, replace=False)] = 1
    df_kaggle = pd.DataFrame(data_kaggle)

    # Dataset 2: brasileiro_sintetico (simulado)
    data_br = {
        'valor': np.random.rand(10000) * 5000,
        'tipo_transacao': np.random.choice(['DEBITO', 'CREDITO', 'PIX'], 10000),
        'canal': np.random.choice(['pos', 'atm', 'web'], 10000),
        'cidade': np.random.choice(['Sao Paulo', 'Rio de Janeiro'], 10000),
        'estado': np.random.choice(['SP', 'RJ'], 10000),
        'pais': 'BR',
        'ip_address': np.random.choice([f'192.168.1.{i}' for i in range(10)], 10000),
        'device_id': np.random.choice(['mobile', 'desktop', 'pos_terminal'], 10000),
        'conta_recebedor': np.random.choice([f'merchant_{i}' for i in range(20)], 10000),
        'cliente_cpf': np.random.choice([f'{i:011d}' for i in range(1000)], 10000),
        'timestamp': pd.to_datetime('2025-01-01') + pd.to_timedelta(np.arange(10000), unit='s'),
        'isFraud': np.random.randint(0, 2, 10000)
    }
    # Aumentar um pouco a proporção de fraude para o exemplo
    data_br['isFraud'][np.random.choice(10000, 50, replace=False)] = 1
    df_br = pd.DataFrame(data_br)

    # Renomear colunas para padronizar (exemplo simplificado)
    df_kaggle = df_kaggle.rename(columns={'Amount': 'valor_transacao'})
    # df_br já tem 'valor_transacao'

    # Para o exemplo, vamos simular a unificação de features
    # Em um cenário real, isso seria muito mais complexo e envolveria engenharia de features
    # para criar um conjunto comum de features a partir de ambos os datasets.
    # Aqui, vamos apenas garantir que os nomes das colunas sejam únicos para concatenação.
    # E que as colunas sejam compatíveis para o preprocessor

    # Selecionar um subconjunto de colunas comuns ou criar um mapeamento
    # Para este exemplo, vamos simplificar e usar apenas algumas colunas numéricas e categóricas
    # Em um cenário real, você faria um feature engineering mais robusto
    # Vamos criar um conjunto de features que ambos os DFs podem ter
    common_numerical_features = ['valor_transacao']
    common_categorical_features = ['tipo_transacao', 'canal']

    # Adicionar colunas V1, V2, V3 apenas para o df_kaggle_processed
    df_kaggle_processed = df_kaggle[['V1', 'V2', 'V3', 'valor_transacao', 'isFraud']]
    df_br_processed = df_br[['valor_transacao', 'tipo_transacao', 'canal', 'isFraud']]

    datasets = {
        'creditcard_kaggle': df_kaggle_processed,
        'brasileiro_sintetico': df_br_processed
    }

    engine = EnhancedFraudEngineV4(sample_size_per_class=5000, oversampling_strategy='smote')
    metrics = engine.train_model(datasets)

    # Testar predição
    sample_transaction = pd.DataFrame({
        'V1': [0.1], 'V2': [0.2], 'V3': [0.3], 'valor_transacao': [100.0],
        'tipo_transacao': ['CREDITO'], 'canal': ['web']
    })
    
    prediction = engine.predict(sample_transaction)
    proba = engine.predict_proba(sample_transaction)
    logging.info(f"Predição para transação de exemplo: {prediction[0]}")
    logging.info(f"Probabilidade de fraude para transação de exemplo: {proba[0]:.4f}")

    engine.save_model("final_fraud_model.joblib")
    new_engine = EnhancedFraudEngineV4()
    new_engine.load_model("final_fraud_model.joblib")
    logging.info("Modelo carregado e pronto para uso.")


