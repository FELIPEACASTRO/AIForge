#!/usr/bin/env python3
"""
Analisador de Fraude Simplificado
Sistema otimizado para análise rápida de transações em lote
"""

import numpy as np
import pandas as pd
from datetime import datetime
import random
import hashlib

class SimpleFraudAnalyzer:
    """Analisador de fraude otimizado para testes de performance"""
    
    def __init__(self):
        # Pesos dos fatores de risco (baseados em análise real)
        self.risk_weights = {
            'valor': 0.25,
            'horario': 0.15,
            'canal': 0.20,
            'localizacao': 0.15,
            'frequencia': 0.10,
            'padrao_comportamental': 0.15
        }
        
        # Thresholds de risco
        self.high_risk_threshold = 0.7
        self.medium_risk_threshold = 0.4
        
        # Cache de análises recentes
        self.analysis_cache = {}
        
        print("🔍 Analisador de Fraude Simplificado inicializado")
    
    def analyze_transaction(self, transaction):
        """Analisa uma transação e retorna score de fraude"""
        try:
            # Gerar hash da transação para cache
            tx_hash = self._generate_transaction_hash(transaction)
            
            # Verificar cache
            if tx_hash in self.analysis_cache:
                return self.analysis_cache[tx_hash]
            
            # Calcular score de fraude
            fraud_score = self._calculate_fraud_score(transaction)
            
            # Determinar status
            if fraud_score >= self.high_risk_threshold:
                status = 'REJECT'
                risk_level = 'Alto'
            elif fraud_score >= self.medium_risk_threshold:
                status = 'REVIEW'
                risk_level = 'Médio'
            else:
                status = 'APPROVE'
                risk_level = 'Baixo'
            
            # Resultado da análise
            result = {
                'transaction_id': transaction.get('id', 'unknown'),
                'fraud_score': fraud_score,
                'status': status,
                'risk_level': risk_level,
                'analysis_timestamp': datetime.now().isoformat(),
                'factors': self._get_risk_factors(transaction, fraud_score)
            }
            
            # Armazenar no cache (limitado a 10000 entradas)
            if len(self.analysis_cache) < 10000:
                self.analysis_cache[tx_hash] = result
            
            return result
            
        except Exception as e:
            # Em caso de erro, retornar análise padrão
            return {
                'transaction_id': transaction.get('id', 'unknown'),
                'fraud_score': 0.0,
                'status': 'APPROVE',
                'risk_level': 'Baixo',
                'analysis_timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _generate_transaction_hash(self, transaction):
        """Gera hash único para a transação"""
        key_fields = [
            str(transaction.get('valor', 0)),
            str(transaction.get('cpf', '')),
            str(transaction.get('canal', '')),
            str(transaction.get('tipo', '')),
            str(transaction.get('localizacao', ''))
        ]
        hash_input = '|'.join(key_fields)
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _calculate_fraud_score(self, transaction):
        """Calcula o score de fraude baseado em múltiplos fatores"""
        
        # Fator 1: Análise de Valor
        valor_risk = self._analyze_value_risk(transaction.get('valor', 0))
        
        # Fator 2: Análise de Horário
        horario_risk = self._analyze_time_risk(transaction.get('data_hora', ''))
        
        # Fator 3: Análise de Canal
        canal_risk = self._analyze_channel_risk(transaction.get('canal', ''))
        
        # Fator 4: Análise de Localização
        location_risk = self._analyze_location_risk(transaction.get('localizacao', ''))
        
        # Fator 5: Análise de Frequência (simulada)
        frequency_risk = self._analyze_frequency_risk(transaction.get('cpf', ''))
        
        # Fator 6: Padrão Comportamental (simulado)
        behavioral_risk = self._analyze_behavioral_pattern(transaction)
        
        # Calcular score ponderado
        fraud_score = (
            valor_risk * self.risk_weights['valor'] +
            horario_risk * self.risk_weights['horario'] +
            canal_risk * self.risk_weights['canal'] +
            location_risk * self.risk_weights['localizacao'] +
            frequency_risk * self.risk_weights['frequencia'] +
            behavioral_risk * self.risk_weights['padrao_comportamental']
        )
        
        # Adicionar ruído aleatório para simular variabilidade real
        noise = random.uniform(-0.05, 0.05)
        fraud_score = max(0.0, min(1.0, fraud_score + noise))
        
        return fraud_score
    
    def _analyze_value_risk(self, valor):
        """Analisa risco baseado no valor da transação"""
        if valor <= 0:
            return 0.1
        
        # Valores muito altos ou muito baixos são suspeitos
        if valor > 50000:  # Acima de R$ 50k
            return 0.8
        elif valor > 10000:  # Entre R$ 10k e R$ 50k
            return 0.6
        elif valor < 1:  # Valores muito baixos
            return 0.7
        elif valor < 10:  # Valores baixos
            return 0.4
        else:
            return 0.2  # Valores normais
    
    def _analyze_time_risk(self, data_hora):
        """Analisa risco baseado no horário da transação"""
        try:
            if isinstance(data_hora, str):
                # Extrair hora da string
                if ':' in data_hora:
                    hora = int(data_hora.split(' ')[-1].split(':')[0])
                else:
                    hora = 12  # Padrão
            else:
                hora = 12  # Padrão
            
            # Horários de madrugada são mais suspeitos
            if 2 <= hora <= 5:
                return 0.8
            elif 22 <= hora <= 23 or 0 <= hora <= 1:
                return 0.6
            elif 6 <= hora <= 8 or 18 <= hora <= 21:
                return 0.3
            else:
                return 0.2  # Horário comercial
                
        except:
            return 0.2  # Padrão em caso de erro
    
    def _analyze_channel_risk(self, canal):
        """Analisa risco baseado no canal da transação"""
        canal = canal.upper() if canal else ''
        
        channel_risks = {
            'INTERNET': 0.5,
            'MOBILE': 0.4,
            'ATM': 0.3,
            'POS': 0.3,
            'AGENCIA': 0.1,
            'CALL_CENTER': 0.6
        }
        
        return channel_risks.get(canal, 0.4)
    
    def _analyze_location_risk(self, localizacao):
        """Analisa risco baseado na localização"""
        if not localizacao:
            return 0.5
        
        # Localizações de alto risco (simulado)
        high_risk_locations = ['Exterior', 'Fronteira', 'Area Rural']
        medium_risk_locations = ['Interior', 'Periferia']
        
        localizacao_upper = localizacao.upper()
        
        for location in high_risk_locations:
            if location.upper() in localizacao_upper:
                return 0.8
        
        for location in medium_risk_locations:
            if location.upper() in localizacao_upper:
                return 0.5
        
        return 0.2  # Localização normal
    
    def _analyze_frequency_risk(self, cpf):
        """Analisa risco baseado na frequência de transações do CPF"""
        if not cpf:
            return 0.5
        
        # Simular análise de frequência baseada no hash do CPF
        cpf_hash = hashlib.md5(cpf.encode()).hexdigest()
        frequency_indicator = int(cpf_hash[:2], 16) / 255.0
        
        # CPFs com alta frequência podem ser suspeitos
        if frequency_indicator > 0.8:
            return 0.7
        elif frequency_indicator > 0.6:
            return 0.4
        else:
            return 0.2
    
    def _analyze_behavioral_pattern(self, transaction):
        """Analisa padrões comportamentais suspeitos"""
        risk_score = 0.0
        
        # Verificar combinações suspeitas
        valor = transaction.get('valor', 0)
        canal = transaction.get('canal', '').upper()
        tipo = transaction.get('tipo', '').upper()
        
        # Padrões suspeitos
        if valor > 20000 and canal == 'INTERNET':
            risk_score += 0.3
        
        if tipo == 'PIX' and valor > 5000:
            risk_score += 0.2
        
        if canal == 'ATM' and valor > 10000:
            risk_score += 0.4
        
        # Adicionar ruído baseado em características da transação
        transaction_hash = self._generate_transaction_hash(transaction)
        noise = int(transaction_hash[:2], 16) / 255.0 * 0.2
        
        return min(1.0, risk_score + noise)
    
    def _get_risk_factors(self, transaction, fraud_score):
        """Retorna os fatores de risco identificados"""
        factors = []
        
        valor = transaction.get('valor', 0)
        canal = transaction.get('canal', '')
        
        if valor > 20000:
            factors.append('Valor alto')
        
        if canal.upper() == 'INTERNET' and valor > 10000:
            factors.append('Transação online de alto valor')
        
        if fraud_score > 0.7:
            factors.append('Múltiplos indicadores de risco')
        elif fraud_score > 0.4:
            factors.append('Alguns indicadores de risco')
        
        return factors if factors else ['Transação normal']
    
    def get_cache_stats(self):
        """Retorna estatísticas do cache"""
        return {
            'cache_size': len(self.analysis_cache),
            'cache_limit': 10000,
            'cache_usage': len(self.analysis_cache) / 10000 * 100
        }
