#!/usr/bin/env python3
"""
Gerador de Transações em Tempo Real para Sankofa Enterprise Pro
Simula transações bancárias realistas para demonstrar o sistema funcionando.
"""

import random
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import threading
import uuid
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class Transaction:
    id: str
    valor: float
    tipo_transacao: str
    canal: str
    cidade: str
    estado: str
    pais: str
    ip_address: str
    device_id: str
    conta_recebedor: str
    cliente_cpf: str
    timestamp: str
    latitude: float = 0.0
    longitude: float = 0.0
    is_fraud: bool = False
    fraud_score: float = 0.0

class RealTimeTransactionGenerator:
    """Gerador de transações em tempo real"""
    
    def __init__(self):
        self.transactions = []
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Dados para geração realística
        self.tipos = ["PIX", "DEBITO", "CREDITO", "TED", "DOC"]
        self.canais = ["MOBILE", "WEB", "POS", "ATM"]
        self.cidades = [
            "Sao Paulo, SP", "Rio de Janeiro, RJ", "Belo Horizonte, MG",
            "Salvador, BA", "Fortaleza, CE", "Brasilia, DF", "Curitiba, PR",
            "Recife, PE", "Porto Alegre, RS", "Manaus, AM", "Belem, PA",
            "Goiania, GO", "Campinas, SP", "Sao Luis, MA", "Maceio, AL"
        ]
        
    def generate_cpf(self) -> str:
        """Gera um CPF fictício mas válido"""
        def calculate_digit(cpf_digits):
            total = sum(int(digit) * weight for digit, weight in zip(cpf_digits, range(len(cpf_digits) + 1, 1, -1)))
            remainder = total % 11
            return '0' if remainder < 2 else str(11 - remainder)
        
        # Gera os primeiros 9 dígitos
        cpf_digits = [str(random.randint(0, 9)) for _ in range(9)]
        
        # Calcula os dígitos verificadores
        cpf_digits.append(calculate_digit(cpf_digits))
        cpf_digits.append(calculate_digit(cpf_digits))
        
        cpf = ''.join(cpf_digits)
        return f"{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}"
    
    def generate_transaction(self) -> Transaction:
        """Gera uma transação realística"""
        transaction_id = f"TXN_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Valor baseado em distribuição realística
        if random.random() < 0.6:  # 60% transações pequenas
            valor = round(random.uniform(10, 500), 2)
        elif random.random() < 0.9:  # 30% transações médias
            valor = round(random.uniform(500, 5000), 2)
        else:  # 10% transações grandes
            valor = round(random.uniform(5000, 50000), 2)
        
        tipo_transacao = random.choice(self.tipos)
        canal = random.choice(self.canais)
        localizacao = random.choice(self.cidades)
        cidade, estado = localizacao.split(', ')
        pais = "BR"
        cliente_cpf = self.generate_cpf()
        conta_recebedor = f"REC_{random.randint(100000, 999999)}"
        
        # Simula análise de fraude
        fraud_score = random.uniform(0.01, 0.99)
        
        # Ocasionalmente gera transações suspeitas
        if random.random() < 0.05:  # 5% de chance
            is_fraud = True
            valor = round(random.uniform(10000, 100000), 2)
        else:
            is_fraud = False

        return Transaction(
            id=transaction_id,
            valor=valor,
            tipo_transacao=tipo_transacao,
            canal=canal,
            cidade=cidade,
            estado=estado,
            pais=pais,
            ip_address=f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
            device_id=f"DEV_{random.randint(100000, 999999)}",
            conta_recebedor=conta_recebedor,
            cliente_cpf=cliente_cpf,
            timestamp=datetime.now().isoformat(),
            latitude=random.uniform(-30, -10),
            longitude=random.uniform(-60, -40),
            is_fraud=is_fraud,
            fraud_score=round(fraud_score, 3)
        )

    
    def start_generation(self, interval: float = 2.0):
        """Inicia a geração de transações em tempo real"""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._generate_loop, args=(interval,))
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Gerador de transações iniciado com intervalo de {interval}s")
    
    def stop_generation(self):
        """Para a geração de transações"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        logger.info("Gerador de transações parado")
    
    def _generate_loop(self, interval: float):
        """Loop principal de geração"""
        while self.is_running:
            try:
                # Gera 1-3 transações por vez
                num_transactions = random.randint(1, 3)
                
                for _ in range(num_transactions):
                    transaction = self.generate_transaction()
                    
                    with self.lock:
                        self.transactions.append(transaction)
                        
                        # Mantém apenas as últimas 1000 transações
                        if len(self.transactions) > 1000:
                            self.transactions = self.transactions[-1000:]
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Erro na geração de transações: {e}")
                time.sleep(1)
    
    def get_transactions(self, limit: int = 100, filters: Dict[str, Any] = None) -> List[Dict]:
        """Retorna transações com filtros opcionais"""
        with self.lock:
            transactions = self.transactions.copy()
        
        # Aplica filtros se fornecidos
        if filters:
            if filters.get('status') and filters['status'] != 'Todos':
                transactions = [t for t in transactions if t.status == filters['status']]
            
            if filters.get('tipo') and filters['tipo'] != 'Todos':
                transactions = [t for t in transactions if t.tipo == filters['tipo']]
            
            if filters.get('search'):
                search_term = filters['search'].lower()
                transactions = [
                    t for t in transactions 
                    if search_term in t.id.lower() or \
                       search_term in t.cpf.lower() or \
                       search_term in t.localizacao.lower()
                ]
        
        # Ordena (mais recentes primeiro por padrão)
        transactions.sort(key=lambda x: x.data_hora, reverse=True)
        
        # Aplica limite
        transactions = transactions[:limit]
        
        # Converte para dict
        return [asdict(t) for t in transactions]
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas das transações"""
        with self.lock:
            transactions = self.transactions.copy()
        
        if not transactions:
            return {
                "total": 0,
                "aprovadas": 0,
                "rejeitadas": 0,
                "em_revisao": 0,
                "valor_total": 0,
                "fraudes_detectadas": 0
            }
        
        total = len(transactions)
        aprovadas = len([t for t in transactions if t.status == "Aprovada"])
        rejeitadas = len([t for t in transactions if t.status == "Rejeitada"])
        em_revisao = len([t for t in transactions if t.status == "Em Revisão"])
        valor_total = sum(t.valor for t in transactions)
        fraudes_detectadas = len([t for t in transactions if t.fraud_score > 0.7])
        
        return {
            "total": total,
            "aprovadas": aprovadas,
            "rejeitadas": rejeitadas,
            "em_revisao": em_revisao,
            "valor_total": round(valor_total, 2),
            "fraudes_detectadas": fraudes_detectadas,
            "taxa_aprovacao": round((aprovadas / total) * 100, 2) if total > 0 else 0
        }

# Instância global do gerador
transaction_generator = RealTimeTransactionGenerator()

if __name__ == "__main__":
    # Teste do gerador
    generator = RealTimeTransactionGenerator()
    generator.start_generation(1.0)  # Uma transação por segundo
    
    try:
        time.sleep(10)  # Roda por 10 segundos
        transactions = generator.get_transactions(limit=10)
        print(f"Geradas {len(transactions)} transações:")
        for t in transactions:
            print(f"  {t['id']}: R$ {t['valor']} - {t['status']} ({t['risco']})")
        
        stats = generator.get_stats()
        print(f"\nEstatísticas: {stats}")
        
    finally:
        generator.stop_generation()

