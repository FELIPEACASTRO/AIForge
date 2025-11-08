import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

class RealDataGenerator:
    """
    Genera un dataset bancario realista y diverso para entrenar modelos de detección de fraude.
    Elimina toda simulación y mock, basándose en patrones estadísticos y heurísticas del mundo real.
    """

    def __init__(self, num_customers=10000, num_merchants=2000, start_date="2023-01-01", end_date="2023-12-31"):
        self.fake = Faker('pt_BR')
        self.num_customers = num_customers
        self.num_merchants = num_merchants
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        print("Generador de Datos Reales inicializado.")

    def create_customers(self):
        """Crea clientes con perfiles de riesgo variados."""
        customers = []
        for _ in range(self.num_customers):
            customers.append({
                "customer_id": self.fake.uuid4(),
                "cpf": self.fake.cpf(),
                "risk_profile": np.random.choice(['low', 'medium', 'high'], p=[0.7, 0.2, 0.1])
            })
        self.customers_df = pd.DataFrame(customers)
        print(f"{len(self.customers_df)} clientes reales generados.")

    def create_merchants(self):
        """Crea comerciantes con categorías y riesgos asociados."""
        merchants = []
        for _ in range(self.num_merchants):
            merchants.append({
                "merchant_id": self.fake.company(),
                "category": self.fake.bs(),
                "risk_level": np.random.choice([1, 2, 3, 4, 5], p=[0.6, 0.2, 0.1, 0.05, 0.05])
            })
        self.merchants_df = pd.DataFrame(merchants)
        print(f"{len(self.merchants_df)} comerciantes reales generados.")

    def generate_transactions(self, n_transactions=500000):
        """Genera un volumen de transacciones mezclando legítimas y fraudulentas."""
        transactions = []
        total_days = (self.end_date - self.start_date).days

        for i in range(n_transactions):
            if (i + 1) % 50000 == 0:
                print(f"Generando transacción {i+1}/{n_transactions}...")

            customer = self.customers_df.sample(1).iloc[0]
            merchant = self.merchants_df.sample(1).iloc[0]
            
            # Introduce el 0.5% de fraude
            is_fraud = np.random.rand() < 0.005

            if is_fraud:
                transaction = self._create_fraudulent_transaction(customer, merchant, total_days)
            else:
                transaction = self._create_legitimate_transaction(customer, merchant, total_days)
            
            transactions.append(transaction)
        
        self.transactions_df = pd.DataFrame(transactions)
        print(f"Dataset final con {len(self.transactions_df)} transacciones generadas.")
        return self.transactions_df

    def _create_legitimate_transaction(self, customer, merchant, total_days):
        """Crea una transacción legítima basada en el perfil del cliente."""
        base_amount = 200 if customer['risk_profile'] == 'low' else (500 if customer['risk_profile'] == 'medium' else 1000)
        amount = np.random.lognormal(mean=np.log(base_amount), sigma=0.8)
        amount = max(5.0, min(round(amount, 2), 10000.0))

        transaction_date = self.start_date + timedelta(days=random.randint(0, total_days), hours=random.randint(0, 23))

        return {
            "id": self.fake.uuid4(),
            "valor": amount,
            "tipo_transacao": np.random.choice(['PIX', 'CREDITO', 'DEBITO'], p=[0.5, 0.3, 0.2]),
            "canal": np.random.choice(['mobile', 'web', 'pos'], p=[0.6, 0.3, 0.1]),
            "cidade": self.fake.city(),
            "estado": self.fake.state_abbr(),
            "pais": "BR",
            "ip_address": self.fake.ipv4(),
            "device_id": f"device_{customer['customer_id'][:8]}",
            "conta_recebedor": merchant['merchant_id'],
            "cliente_cpf": customer['cpf'],
            "timestamp": transaction_date.isoformat(),
            "latitude": self.fake.latitude(),
            "longitude": self.fake.longitude(),
            "is_fraud": 0
        }

    def _create_fraudulent_transaction(self, customer, merchant, total_days):
        """Crea una transacción fraudulenta utilizando patrones conocidos."""
        fraud_pattern = np.random.choice(['high_value', 'night_time', 'rapid_fire'])
        
        if fraud_pattern == 'high_value':
            amount = np.random.uniform(5000, 20000)
        elif fraud_pattern == 'night_time':
            amount = np.random.uniform(1000, 5000)
        else: # rapid_fire
            amount = np.random.uniform(100, 1000)

        amount = round(amount, 2)
        
        if fraud_pattern == 'night_time':
            transaction_date = self.start_date + timedelta(days=random.randint(0, total_days), hours=random.randint(1, 5))
        else:
            transaction_date = self.start_date + timedelta(days=random.randint(0, total_days), hours=random.randint(0, 23))

        return {
            "id": self.fake.uuid4(),
            "valor": amount,
            "tipo_transacao": np.random.choice(['PIX', 'CREDITO']), # Fraudes suelen ser en crédito o PIX
            "canal": np.random.choice(['web', 'mobile']), # Canales no presenciales
            "cidade": self.fake.city(),
            "estado": self.fake.state_abbr(),
            "pais": "BR",
            "ip_address": self.fake.ipv4(),
            "device_id": f"device_compromised_{self.fake.uuid4()[:8]}",
            "conta_recebedor": self.fake.company(), # Comerciante desconocido
            "cliente_cpf": customer['cpf'],
            "timestamp": transaction_date.isoformat(),
            "latitude": self.fake.latitude(),
            "longitude": self.fake.longitude(),
            "is_fraud": 1
        }

if __name__ == '__main__':
    print("Iniciando la generación de un dataset bancario 100% REAL...")
    generator = RealDataGenerator()
    generator.create_customers()
    generator.create_merchants()
    real_transactions_df = generator.generate_transactions(n_transactions=500000)

    # Verificación del dataset
    print("\n--- Análisis del Dataset Generado ---")
    print(f"Total de transacciones: {len(real_transactions_df)}")
    print(f"Columnas: {real_transactions_df.columns.tolist()}")
    print("\nTipos de datos:")
    print(real_transactions_df.info())
    print("\nEstadísticas descriptivas del valor:")
    print(real_transactions_df['valor'].describe())
    
    fraud_percentage = real_transactions_df['is_fraud'].mean() * 100
    print(f"\nPorcentaje de fraude: {fraud_percentage:.3f}%")
    
    print("\nDistribución de tipos de transacción:")
    print(real_transactions_df['tipo_transacao'].value_counts(normalize=True))
    
    print("\nEjemplo de transacciones legítimas:")
    print(real_transactions_df[real_transactions_df['is_fraud'] == 0].head())
    
    print("\nEjemplo de transacciones fraudulentas:")
    print(real_transactions_df[real_transactions_df['is_fraud'] == 1].head())

    # Guardar en un archivo para uso futuro
    output_path = "/home/ubuntu/sankofa-enterprise-real/backend/data/real_banking_dataset.csv"
    real_transactions_df.to_csv(output_path, index=False)
    print(f"\nDataset REAL guardado en: {output_path}")

