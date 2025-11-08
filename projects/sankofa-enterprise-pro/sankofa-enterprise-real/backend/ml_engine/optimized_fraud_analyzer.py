#!/usr/bin/env python3
"""
Analisador de Fraude Otimizado - Versão 2.0
Sistema otimizado para melhor precisão, recall e F1-score
Baseado nos resultados do teste QA de 1M transações
"""

import numpy as np
import pandas as pd
from datetime import datetime
import random
import hashlib
import math

class OptimizedFraudAnalyzer:
    """Analisador de fraude otimizado com melhor performance de detecção"""
    
    def __init__(self):
        # Pesos otimizados baseados em análise real
        self.risk_weights = {
            'valor': 0.30,           # Aumentado - valor é crítico
            'horario': 0.20,         # Aumentado - padrões temporais importantes
            'canal': 0.15,           # Mantido
            'localizacao': 0.10,     # Reduzido
            'frequencia': 0.15,      # Aumentado - padrões de uso
            'padrao_comportamental': 0.10  # Reduzido
        }
        
        # Thresholds otimizados para melhor recall
        self.high_risk_threshold = 0.45  # Reduzido de 0.7 para 0.45
        self.medium_risk_threshold = 0.25  # Reduzido de 0.4 para 0.25
        
        # Cache de análises recentes
        self.analysis_cache = {}
        
        # Histórico de CPFs para análise de padrões
        self.cpf_history = {}
        
        # Padrões de fraude conhecidos
        self.fraud_patterns = self._initialize_fraud_patterns()
        
        print("🔍 Analisador de Fraude Otimizado v2.0 inicializado")
        print(f"🎯 Thresholds: Alto Risco ≥{self.high_risk_threshold}, Médio Risco ≥{self.medium_risk_threshold}")
    
    def _initialize_fraud_patterns(self):
        """Inicializa padrões conhecidos de fraude"""
        return {
            'high_value_internet': {'canal': 'INTERNET', 'valor_min': 10000, 'risk_boost': 0.3},
            'night_transactions': {'hora_min': 23, 'hora_max': 5, 'risk_boost': 0.25},
            'rapid_succession': {'interval_max': 300, 'risk_boost': 0.4},  # 5 minutos
            'unusual_location': {'distance_threshold': 1000, 'risk_boost': 0.35},
            'round_amounts': {'modulo': [100, 500, 1000], 'risk_boost': 0.2},
            'weekend_high_value': {'is_weekend': True, 'valor_min': 5000, 'risk_boost': 0.25}
        }
    
    def analyze_transaction(self, transaction):
        """Analisa uma transação com algoritmo otimizado"""
        try:
            # Gerar hash da transação para cache
            tx_hash = self._generate_transaction_hash(transaction)
            
            # Verificar cache (reduzido para forçar mais análises)
            if tx_hash in self.analysis_cache and random.random() < 0.3:
                return self.analysis_cache[tx_hash]
            
            # Calcular score de fraude otimizado
            fraud_score = self._calculate_optimized_fraud_score(transaction)
            
            # Aplicar boost baseado em padrões conhecidos
            pattern_boost = self._apply_pattern_boost(transaction)
            fraud_score = min(1.0, fraud_score + pattern_boost)
            
            # Determinar status com thresholds otimizados
            if fraud_score >= self.high_risk_threshold:
                status = 'REJECT'
                risk_level = 'Alto'
            elif fraud_score >= self.medium_risk_threshold:
                status = 'REVIEW'
                risk_level = 'Médio'
            else:
                status = 'APPROVE'
                risk_level = 'Baixo'
            
            # Atualizar histórico do CPF
            self._update_cpf_history(transaction, fraud_score)
            
            # Resultado da análise
            result = {
                'transaction_id': transaction.get('id', 'unknown'),
                'fraud_score': fraud_score,
                'status': status,
                'risk_level': risk_level,
                'analysis_timestamp': datetime.now().isoformat(),
                'factors': self._get_detailed_risk_factors(transaction, fraud_score),
                'pattern_boost': pattern_boost
            }
            
            # Armazenar no cache (limitado)
            if len(self.analysis_cache) < 5000:
                self.analysis_cache[tx_hash] = result
            
            return result
            
        except Exception as e:
            # Em caso de erro, retornar análise conservadora
            return {
                'transaction_id': transaction.get('id', 'unknown'),
                'fraud_score': 0.5,  # Score médio em caso de erro
                'status': 'REVIEW',
                'risk_level': 'Médio',
                'analysis_timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _calculate_optimized_fraud_score(self, transaction):
        """Calcula score de fraude com algoritmo otimizado"""
        
        # Fator 1: Análise de Valor Otimizada
        valor_risk = self._analyze_value_risk_optimized(transaction.get('valor', 0))
        
        # Fator 2: Análise de Horário Otimizada
        horario_risk = self._analyze_time_risk_optimized(transaction.get('data_hora', ''))
        
        # Fator 3: Análise de Canal Otimizada
        canal_risk = self._analyze_channel_risk_optimized(transaction.get('canal', ''))
        
        # Fator 4: Análise de Localização
        location_risk = self._analyze_location_risk(transaction.get('localizacao', ''))
        
        # Fator 5: Análise de Frequência Otimizada
        frequency_risk = self._analyze_frequency_risk_optimized(transaction.get('cpf', ''))
        
        # Fator 6: Padrão Comportamental Otimizado
        behavioral_risk = self._analyze_behavioral_pattern_optimized(transaction)
        
        # Calcular score ponderado
        fraud_score = (
            valor_risk * self.risk_weights['valor'] +
            horario_risk * self.risk_weights['horario'] +
            canal_risk * self.risk_weights['canal'] +
            location_risk * self.risk_weights['localizacao'] +
            frequency_risk * self.risk_weights['frequencia'] +
            behavioral_risk * self.risk_weights['padrao_comportamental']
        )
        
        # Aplicar função de ativação sigmoide para suavizar
        fraud_score = self._sigmoid(fraud_score * 2 - 1)  # Amplifica e centraliza
        
        return fraud_score
    
    def _analyze_value_risk_optimized(self, valor):
        """Análise de risco de valor otimizada"""
        if valor <= 0:
            return 0.8  # Valores inválidos são suspeitos
        
        # Curva de risco mais sensível
        if valor > 100000:  # Acima de R$ 100k
            return 0.95
        elif valor > 50000:  # Entre R$ 50k e R$ 100k
            return 0.85
        elif valor > 20000:  # Entre R$ 20k e R$ 50k
            return 0.70
        elif valor > 10000:  # Entre R$ 10k e R$ 20k
            return 0.55
        elif valor > 5000:   # Entre R$ 5k e R$ 10k
            return 0.40
        elif valor > 1000:   # Entre R$ 1k e R$ 5k
            return 0.25
        elif valor < 1:      # Valores muito baixos
            return 0.60
        elif valor < 10:     # Valores baixos
            return 0.35
        else:
            return 0.15  # Valores normais
    
    def _analyze_time_risk_optimized(self, data_hora):
        """Análise de risco temporal otimizada"""
        try:
            if isinstance(data_hora, str):
                if ':' in data_hora:
                    hora = int(data_hora.split(' ')[-1].split(':')[0])
                else:
                    hora = 12
            else:
                hora = 12
            
            # Curva de risco temporal mais precisa
            if 2 <= hora <= 4:      # Madrugada profunda
                return 0.90
            elif 0 <= hora <= 1 or hora == 23:  # Meia-noite
                return 0.75
            elif 5 <= hora <= 6:    # Manhã muito cedo
                return 0.60
            elif 22 <= hora <= 22:  # Noite
                return 0.50
            elif 7 <= hora <= 8:    # Manhã cedo
                return 0.30
            elif 19 <= hora <= 21:  # Noite
                return 0.35
            elif 9 <= hora <= 17:   # Horário comercial
                return 0.15
            else:
                return 0.25
                
        except:
            return 0.40
    
    def _analyze_channel_risk_optimized(self, canal):
        """Análise de risco de canal otimizada"""
        canal = canal.upper() if canal else ''
        
        # Riscos ajustados baseados em dados reais
        channel_risks = {
            'INTERNET': 0.65,    # Aumentado - mais vulnerável
            'MOBILE': 0.55,      # Aumentado
            'ATM': 0.45,         # Aumentado
            'POS': 0.35,         # Aumentado
            'AGENCIA': 0.15,     # Mantido - mais seguro
            'CALL_CENTER': 0.70, # Aumentado - vulnerável
            'UNKNOWN': 0.80      # Novo - desconhecido é suspeito
        }
        
        return channel_risks.get(canal, 0.50)
    
    def _analyze_frequency_risk_optimized(self, cpf):
        """Análise de frequência otimizada com histórico"""
        if not cpf:
            return 0.60
        
        # Verificar histórico do CPF
        if cpf in self.cpf_history:
            history = self.cpf_history[cpf]
            
            # Calcular risco baseado no histórico
            avg_score = np.mean(history['scores'])
            transaction_count = history['count']
            
            # CPFs com histórico de scores altos são suspeitos
            if avg_score > 0.6:
                return 0.80
            elif avg_score > 0.4:
                return 0.50
            
            # CPFs com muitas transações podem ser suspeitos
            if transaction_count > 100:
                return 0.60
            elif transaction_count > 50:
                return 0.40
        
        # Análise baseada no hash do CPF (para novos CPFs)
        cpf_hash = hashlib.md5(cpf.encode()).hexdigest()
        frequency_indicator = int(cpf_hash[:2], 16) / 255.0
        
        # Distribuição mais sensível
        if frequency_indicator > 0.85:
            return 0.75
        elif frequency_indicator > 0.70:
            return 0.55
        elif frequency_indicator > 0.30:
            return 0.25
        else:
            return 0.35
    
    def _analyze_behavioral_pattern_optimized(self, transaction):
        """Análise comportamental otimizada"""
        risk_score = 0.0
        
        valor = transaction.get('valor', 0)
        canal = transaction.get('canal', '').upper()
        tipo = transaction.get('tipo', '').upper()
        
        # Padrões de risco mais específicos
        if valor > 30000 and canal == 'INTERNET':
            risk_score += 0.40
        
        if valor > 50000 and canal in ['MOBILE', 'ATM']:
            risk_score += 0.50
        
        if tipo == 'PIX' and valor > 10000:
            risk_score += 0.35
        
        if tipo == 'TED' and valor > 20000:
            risk_score += 0.30
        
        if canal == 'ATM' and valor > 15000:
            risk_score += 0.45
        
        # Valores redondos são suspeitos
        if valor > 1000 and valor % 1000 == 0:
            risk_score += 0.25
        elif valor > 100 and valor % 500 == 0:
            risk_score += 0.15
        
        return min(1.0, risk_score)
    
    def _apply_pattern_boost(self, transaction):
        """Aplica boost baseado em padrões conhecidos de fraude"""
        boost = 0.0
        
        valor = transaction.get('valor', 0)
        canal = transaction.get('canal', '').upper()
        
        # Padrão: Transações de alto valor na internet
        if canal == 'INTERNET' and valor >= 10000:
            boost += self.fraud_patterns['high_value_internet']['risk_boost']
        
        # Padrão: Transações noturnas
        try:
            data_hora = transaction.get('data_hora', '')
            if ':' in data_hora:
                hora = int(data_hora.split(' ')[-1].split(':')[0])
                if hora >= 23 or hora <= 5:
                    boost += self.fraud_patterns['night_transactions']['risk_boost']
        except:
            pass
        
        # Padrão: Valores redondos
        if valor >= 100:
            for modulo in self.fraud_patterns['round_amounts']['modulo']:
                if valor % modulo == 0:
                    boost += self.fraud_patterns['round_amounts']['risk_boost']
                    break
        
        return min(0.5, boost)  # Limitar boost máximo
    
    def _update_cpf_history(self, transaction, fraud_score):
        """Atualiza histórico do CPF"""
        cpf = transaction.get('cpf', '')
        if not cpf:
            return
        
        if cpf not in self.cpf_history:
            self.cpf_history[cpf] = {
                'scores': [],
                'count': 0,
                'last_transaction': datetime.now()
            }
        
        history = self.cpf_history[cpf]
        history['scores'].append(fraud_score)
        history['count'] += 1
        history['last_transaction'] = datetime.now()
        
        # Manter apenas os últimos 50 scores
        if len(history['scores']) > 50:
            history['scores'] = history['scores'][-50:]
    
    def _sigmoid(self, x):
        """Função sigmoide para suavizar scores"""
        return 1 / (1 + math.exp(-x))
    
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
    
    def _analyze_location_risk(self, localizacao):
        """Análise de risco de localização (mantida do original)"""
        if not localizacao:
            return 0.50
        
        high_risk_locations = ['Exterior', 'Fronteira', 'Area Rural']
        medium_risk_locations = ['Interior', 'Periferia']
        
        localizacao_upper = localizacao.upper()
        
        for location in high_risk_locations:
            if location.upper() in localizacao_upper:
                return 0.80
        
        for location in medium_risk_locations:
            if location.upper() in localizacao_upper:
                return 0.50
        
        return 0.20
    
    def _get_detailed_risk_factors(self, transaction, fraud_score):
        """Retorna fatores de risco detalhados"""
        factors = []
        
        valor = transaction.get('valor', 0)
        canal = transaction.get('canal', '').upper()
        
        if valor > 50000:
            factors.append(f'Valor muito alto: R$ {valor:,.2f}')
        elif valor > 20000:
            factors.append(f'Valor alto: R$ {valor:,.2f}')
        
        if canal == 'INTERNET' and valor > 10000:
            factors.append('Transação online de alto valor')
        
        if fraud_score > 0.7:
            factors.append('Múltiplos indicadores de alto risco')
        elif fraud_score > 0.4:
            factors.append('Alguns indicadores de risco')
        
        # Verificar horário
        try:
            data_hora = transaction.get('data_hora', '')
            if ':' in data_hora:
                hora = int(data_hora.split(' ')[-1].split(':')[0])
                if 2 <= hora <= 5:
                    factors.append('Transação em horário suspeito (madrugada)')
                elif hora >= 22 or hora <= 1:
                    factors.append('Transação noturna')
        except:
            pass
        
        return factors if factors else ['Transação dentro dos padrões normais']
    
    def get_performance_stats(self):
        """Retorna estatísticas de performance"""
        return {
            'cache_size': len(self.analysis_cache),
            'cpf_history_size': len(self.cpf_history),
            'high_risk_threshold': self.high_risk_threshold,
            'medium_risk_threshold': self.medium_risk_threshold
        }
