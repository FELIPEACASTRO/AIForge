import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import TimeSeriesSplit
import threading
import time
import logging

class ContinuousLearningSystem:
    """
    Sistema de Aprendizado Contínuo que cresce automaticamente com dados de produção.
    
    Funcionalidades:
    1. Inicia com dataset mínimo (bootstrap)
    2. Coleta cada transação processada
    3. Incorpora feedback de analistas
    4. Retreina modelos automaticamente
    5. Valida performance continuamente
    """
    
    def __init__(self, db_path="/home/ubuntu/sankofa-enterprise-real/backend/data/production_data.db"):
        self.db_path = db_path
        self.model_path = "/home/ubuntu/sankofa-enterprise-real/backend/ml_engine/continuous_models/"
        self.is_learning_active = False
        self.retrain_threshold = 1000  # Retreinar a cada 1000 novas transações
        self.min_fraud_samples = 50    # Mínimo de fraudes para retreinar
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Definir feature names
        self.feature_names = [
            'valor', 'valor_log', 'valor_zscore',
            'hour', 'day_of_week', 'is_weekend', 'is_night',
            'tipo_transacao_encoded', 'canal_encoded', 'estado_encoded',
            'latitude', 'longitude'
        ]
        
        # Inicializar encoders
        self.encoders = {}
        
        # Inicializar sistema
        self._initialize_database()
        self._initialize_models()
        
        print("Sistema de Aprendizado Contínuo inicializado.")
    
    def _initialize_database(self):
        """Inicializa banco de dados para armazenar transações de produção."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de transações processadas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id TEXT PRIMARY KEY,
                valor REAL,
                tipo_transacao TEXT,
                canal TEXT,
                cidade TEXT,
                estado TEXT,
                pais TEXT,
                ip_address TEXT,
                device_id TEXT,
                conta_recebedor TEXT,
                cliente_cpf TEXT,
                timestamp TEXT,
                latitude TEXT,
                longitude TEXT,
                predicted_fraud_prob REAL,
                predicted_is_fraud INTEGER,
                actual_is_fraud INTEGER DEFAULT NULL,
                analyst_feedback TEXT DEFAULT NULL,
                feedback_timestamp TEXT DEFAULT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de métricas de modelo
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT,
                auc_score REAL,
                auprc_score REAL,
                precision_at_95_recall REAL,
                training_samples INTEGER,
                fraud_samples INTEGER,
                training_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de logs de retreino
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS retrain_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger_reason TEXT,
                samples_used INTEGER,
                fraud_samples INTEGER,
                old_auc REAL,
                new_auc REAL,
                improvement REAL,
                retrain_duration REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        self.logger.info("Banco de dados inicializado.")
    
    def _initialize_models(self):
        """Inicializa modelos com dataset mínimo ou carrega modelos existentes."""
        os.makedirs(self.model_path, exist_ok=True)
        
        model_file = os.path.join(self.model_path, "current_model.joblib")
        scaler_file = os.path.join(self.model_path, "current_scaler.joblib")
        encoders_file = os.path.join(self.model_path, "current_encoders.joblib")
        
        if os.path.exists(model_file):
            # Carregar modelos existentes
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            self.encoders = joblib.load(encoders_file)
            self.logger.info("Modelos existentes carregados.")
        else:
            # Criar modelos iniciais com dataset bootstrap
            self._create_bootstrap_models()
        

    
    def _create_bootstrap_models(self):
        """Cria modelos iniciais com dataset mínimo para bootstrap."""
        self.logger.info("Criando modelos bootstrap...")
        
        # Gerar dataset mínimo (10k transações)
        import sys
        sys.path.append('/home/ubuntu/sankofa-enterprise-real/backend/data')
        from real_data_generation import RealDataGenerator
        generator = RealDataGenerator(num_customers=1000, num_merchants=200)
        generator.create_customers()
        generator.create_merchants()
        bootstrap_df = generator.generate_transactions(n_transactions=10000)
        
        # Preparar dados
        X, y = self._prepare_features(bootstrap_df)
        
        # Inicializar componentes
        self.scaler = StandardScaler()
        self.encoders = {}
        
        # Treinar modelo inicial
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Normalização e treinamento
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Salvar modelos
        self._save_current_models()
        
        # Registrar métricas iniciais
        self._log_model_metrics("bootstrap", len(bootstrap_df), y.sum(), 0.0, 0.0, 0.0)
        
        self.logger.info("Modelos bootstrap criados e salvos.")
    
    def _prepare_features(self, df):
        """Prepara features a partir do DataFrame."""
        # Features temporais
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Features de valor
        df['valor_log'] = np.log1p(df['valor'])
        df['valor_zscore'] = (df['valor'] - df['valor'].mean()) / df['valor'].std()
        
        # Encoding categórico (criar encoders se não existirem)
        for col in ['tipo_transacao', 'canal', 'estado']:
            if col not in self.encoders:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                self.encoders[col] = le
            else:
                # Usar encoder existente
                try:
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col])
                except ValueError:
                    # Valores novos, expandir encoder
                    unique_values = df[col].unique()
                    for val in unique_values:
                        if val not in self.encoders[col].classes_:
                            self.encoders[col].classes_ = np.append(self.encoders[col].classes_, val)
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col])
        
        # Features de localização
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df.loc[:, 'latitude'] = df['latitude'].fillna(df['latitude'].mean())
        df.loc[:, 'longitude'] = df['longitude'].fillna(df['longitude'].mean())
        
        X = df[self.feature_names]
        y = df['is_fraud'] if 'is_fraud' in df.columns else None
        
        return X, y
    
    def store_transaction(self, transaction_data, prediction_result):
        """Armazena transação processada no banco de dados."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO transactions (
                id, valor, tipo_transacao, canal, cidade, estado, pais,
                ip_address, device_id, conta_recebedor, cliente_cpf,
                timestamp, latitude, longitude, predicted_fraud_prob,
                predicted_is_fraud
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            transaction_data['id'],
            transaction_data['valor'],
            transaction_data['tipo_transacao'],
            transaction_data['canal'],
            transaction_data['cidade'],
            transaction_data['estado'],
            transaction_data['pais'],
            transaction_data['ip_address'],
            transaction_data['device_id'],
            transaction_data['conta_recebedor'],
            transaction_data['cliente_cpf'],
            transaction_data['timestamp'],
            transaction_data['latitude'],
            transaction_data['longitude'],
            prediction_result['fraud_probability'],
            int(prediction_result['is_fraud'])
        ))
        
        conn.commit()
        conn.close()
        
        # Verificar se deve retreinar
        self._check_retrain_trigger()
    
    def add_analyst_feedback(self, transaction_id, is_fraud, analyst_notes=""):
        """Adiciona feedback do analista sobre uma transação."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE transactions 
            SET actual_is_fraud = ?, analyst_feedback = ?, feedback_timestamp = ?
            WHERE id = ?
        ''', (int(is_fraud), analyst_notes, datetime.now().isoformat(), transaction_id))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Feedback adicionado para transação {transaction_id}: {'FRAUDE' if is_fraud else 'LEGÍTIMA'}")
        
        # Verificar se deve retreinar com novo feedback
        self._check_retrain_trigger()
    
    def _check_retrain_trigger(self):
        """Verifica se deve disparar retreino automático."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Contar transações com feedback
        cursor.execute('''
            SELECT COUNT(*) as total, 
                   SUM(CASE WHEN actual_is_fraud = 1 THEN 1 ELSE 0 END) as frauds
            FROM transactions 
            WHERE actual_is_fraud IS NOT NULL
        ''')
        
        result = cursor.fetchone()
        total_feedback, fraud_feedback = result[0], result[1]
        
        conn.close()
        
        # Disparar retreino se atingir thresholds
        if (total_feedback >= self.retrain_threshold and 
            fraud_feedback >= self.min_fraud_samples):
            
            self.logger.info(f"Trigger de retreino ativado: {total_feedback} transações, {fraud_feedback} fraudes")
            self._trigger_retrain("threshold_reached")
    
    def _trigger_retrain(self, reason):
        """Dispara retreino automático em thread separada."""
        if self.is_learning_active:
            self.logger.info("Retreino já em andamento, ignorando trigger.")
            return
        
        def retrain_worker():
            self.is_learning_active = True
            start_time = time.time()
            
            try:
                self.logger.info(f"Iniciando retreino automático. Razão: {reason}")
                
                # Carregar dados com feedback
                conn = sqlite3.connect(self.db_path)
                df = pd.read_sql_query('''
                    SELECT * FROM transactions 
                    WHERE actual_is_fraud IS NOT NULL
                    ORDER BY timestamp
                ''', conn)
                conn.close()
                
                if len(df) < 100:
                    self.logger.warning("Dados insuficientes para retreino.")
                    return
                
                # Preparar dados
                X, y = self._prepare_features(df)
                
                # Encoding categórico para novos valores
                for col in ['tipo_transacao', 'canal', 'estado']:
                    unique_values = df[col].unique()
                    for val in unique_values:
                        if val not in self.encoders[col].classes_:
                            # Adicionar nova classe ao encoder
                            self.encoders[col].classes_ = np.append(self.encoders[col].classes_, val)
                    
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col])
                
                # Avaliar modelo atual
                X_scaled = self.scaler.transform(X)
                old_predictions = self.model.predict_proba(X_scaled)[:, 1]
                old_auc = roc_auc_score(y, old_predictions)
                
                # Treinar novo modelo
                new_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Usar validação temporal
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    new_model.fit(X_train, y_train)
                    val_pred = new_model.predict_proba(X_val)[:, 1]
                    scores.append(roc_auc_score(y_val, val_pred))
                
                new_auc = np.mean(scores)
                improvement = new_auc - old_auc
                
                # Atualizar modelo se houve melhoria
                if improvement > 0.01:  # Melhoria mínima de 1%
                    # Treinar no dataset completo
                    new_model.fit(X_scaled, y)
                    
                    # Atualizar scaler se necessário
                    self.scaler.fit(X)
                    
                    # Substituir modelo atual
                    self.model = new_model
                    self._save_current_models()
                    
                    # Calcular métricas finais
                    final_pred = self.model.predict_proba(X_scaled)[:, 1]
                    precision, recall, _ = precision_recall_curve(y, final_pred)
                    auprc = auc(recall, precision)
                    
                    # Registrar métricas
                    self._log_model_metrics(
                        f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        len(df), y.sum(), new_auc, auprc, 0.0
                    )
                    
                    self.logger.info(f"Modelo atualizado! AUC: {old_auc:.4f} → {new_auc:.4f} (+{improvement:.4f})")
                else:
                    self.logger.info(f"Modelo não atualizado. Melhoria insuficiente: {improvement:.4f}")
                
                # Log do retreino
                duration = time.time() - start_time
                self._log_retrain(reason, len(df), y.sum(), old_auc, new_auc, improvement, duration)
                
            except Exception as e:
                self.logger.error(f"Erro durante retreino: {str(e)}")
            finally:
                self.is_learning_active = False
        
        # Executar retreino em thread separada
        retrain_thread = threading.Thread(target=retrain_worker)
        retrain_thread.daemon = True
        retrain_thread.start()
    
    def _save_current_models(self):
        """Salva modelos atuais."""
        joblib.dump(self.model, os.path.join(self.model_path, "current_model.joblib"))
        joblib.dump(self.scaler, os.path.join(self.model_path, "current_scaler.joblib"))
        joblib.dump(self.encoders, os.path.join(self.model_path, "current_encoders.joblib"))
    
    def _log_model_metrics(self, version, samples, fraud_samples, auc, auprc, precision_95):
        """Registra métricas do modelo."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_metrics (
                model_version, auc_score, auprc_score, precision_at_95_recall,
                training_samples, fraud_samples, training_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (version, auc, auprc, precision_95, samples, fraud_samples, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def _log_retrain(self, reason, samples, fraud_samples, old_auc, new_auc, improvement, duration):
        """Registra log de retreino."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO retrain_logs (
                trigger_reason, samples_used, fraud_samples, old_auc, 
                new_auc, improvement, retrain_duration
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (reason, samples, fraud_samples, old_auc, new_auc, improvement, duration))
        
        conn.commit()
        conn.close()
    
    def predict_fraud(self, transaction_data):
        """Faz predição e armazena transação para aprendizado futuro."""
        # Preparar dados da transação
        df_trans = pd.DataFrame([transaction_data])
        X, _ = self._prepare_features(df_trans)
        
        # Encoding categórico
        for col in ['tipo_transacao', 'canal', 'estado']:
            if col in self.encoders:
                try:
                    df_trans[f'{col}_encoded'] = self.encoders[col].transform(df_trans[col])
                except ValueError:
                    # Valor não visto, usar valor padrão
                    df_trans[f'{col}_encoded'] = 0
        
        # Fazer predição
        X_scaled = self.scaler.transform(X)
        fraud_prob = self.model.predict_proba(X_scaled)[:, 1][0]
        
        result = {
            'fraud_probability': float(fraud_prob),
            'is_fraud': bool(fraud_prob > 0.5),
            'confidence': float(abs(fraud_prob - 0.5) * 2),
            'model_version': 'continuous_learning'
        }
        
        # Armazenar transação para aprendizado futuro
        self.store_transaction(transaction_data, result)
        
        return result
    
    def get_learning_stats(self):
        """Retorna estatísticas do sistema de aprendizado."""
        conn = sqlite3.connect(self.db_path)
        
        # Estatísticas gerais
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                COUNT(*) as total_transactions,
                SUM(CASE WHEN actual_is_fraud IS NOT NULL THEN 1 ELSE 0 END) as feedback_count,
                SUM(CASE WHEN actual_is_fraud = 1 THEN 1 ELSE 0 END) as confirmed_frauds,
                AVG(predicted_fraud_prob) as avg_fraud_score
            FROM transactions
        ''')
        
        stats = cursor.fetchone()
        
        # Últimas métricas do modelo
        cursor.execute('''
            SELECT * FROM model_metrics 
            ORDER BY created_at DESC LIMIT 1
        ''')
        
        latest_metrics = cursor.fetchone()
        
        # Logs de retreino
        cursor.execute('''
            SELECT COUNT(*) FROM retrain_logs
        ''')
        
        retrain_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_transactions': stats[0],
            'feedback_count': stats[1],
            'confirmed_frauds': stats[2],
            'avg_fraud_score': stats[3],
            'latest_model_auc': latest_metrics[2] if latest_metrics else 0.0,
            'retrain_count': retrain_count,
            'is_learning_active': self.is_learning_active
        }

if __name__ == '__main__':
    print("Testando Sistema de Aprendizado Contínuo...")
    
    # Inicializar sistema
    cls = ContinuousLearningSystem()
    
    # Simular algumas transações
    test_transactions = [
        {
            'id': 'test_001',
            'valor': 150.0,
            'tipo_transacao': 'PIX',
            'canal': 'mobile',
            'cidade': 'São Paulo',
            'estado': 'SP',
            'pais': 'BR',
            'ip_address': '192.168.1.1',
            'device_id': 'device_001',
            'conta_recebedor': 'merchant_001',
            'cliente_cpf': '12345678901',
            'timestamp': '2023-12-01T14:30:00',
            'latitude': '-23.5505',
            'longitude': '-46.6333'
        },
        {
            'id': 'test_002',
            'valor': 8000.0,
            'tipo_transacao': 'CREDITO',
            'canal': 'web',
            'cidade': 'Rio de Janeiro',
            'estado': 'RJ',
            'pais': 'BR',
            'ip_address': '10.0.0.1',
            'device_id': 'device_002',
            'conta_recebedor': 'merchant_002',
            'cliente_cpf': '98765432109',
            'timestamp': '2023-12-01T02:15:00',
            'latitude': '-22.9068',
            'longitude': '-43.1729'
        }
    ]
    
    # Processar transações
    for trans in test_transactions:
        result = cls.predict_fraud(trans)
        print(f"Transação {trans['id']}: {result}")
    
    # Simular feedback de analista
    cls.add_analyst_feedback('test_001', False, "Transação legítima confirmada")
    cls.add_analyst_feedback('test_002', True, "Fraude confirmada - valor alto + madrugada")
    
    # Mostrar estatísticas
    stats = cls.get_learning_stats()
    print(f"\nEstatísticas do Sistema: {stats}")
    
    print("\n✅ Sistema de Aprendizado Contínuo implementado com sucesso!")
