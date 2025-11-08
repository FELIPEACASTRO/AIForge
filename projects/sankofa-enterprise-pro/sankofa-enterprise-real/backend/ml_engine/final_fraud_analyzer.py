#!/usr/bin/env python3
"""
Analisador de Fraude Final - Versão 3.0
Sistema de ensemble com múltiplas técnicas e thresholds otimizados
Baseado nos resultados dos testes QA anteriores
"""

import numpy as np
import pandas as pd
from datetime import datetime
import random
import hashlib
import math
from collections import defaultdict

class FinalFraudAnalyzer:
    """Analisador de fraude final com ensemble de técnicas"""
    
    def __init__(self):
        # Múltiplos conjuntos de pesos para ensemble
        self.ensemble_weights = [
            # Modelo 1: Foco em valor
            {'valor': 0.40, 'horario': 0.15, 'canal': 0.15, 'localizacao': 0.10, 'frequencia': 0.10, 'comportamental': 0.10},
            # Modelo 2: Foco em comportamento
            {'valor': 0.20, 'horario': 0.25, 'canal': 0.20, 'localizacao': 0.10, 'frequencia': 0.15, 'comportamental': 0.10},
            # Modelo 3: Foco em canal e localização
            {'valor': 0.25, 'horario': 0.15, 'canal': 0.25, 'localizacao': 0.20, 'frequencia': 0.10, 'comportamental': 0.05},
            # Modelo 4: Balanceado
            {'valor': 0.30, 'horario': 0.20, 'canal': 0.15, 'localizacao': 0.15, 'frequencia': 0.15, 'comportamental': 0.05},
            # Modelo 5: Foco em padrões temporais
            {'valor': 0.25, 'horario': 0.30, 'canal': 0.15, 'localizacao': 0.10, 'frequencia': 0.15, 'comportamental': 0.05}
        ]
        
        # Thresholds otimizados para melhor recall
        self.high_risk_threshold = 0.35  # Reduzido ainda mais
        self.medium_risk_threshold = 0.20  # Reduzido ainda mais
        self.precision_recall_balance = 0.5 # 0.0 para mais recall, 1.0 para mais precisão

    def set_threshold_balance(self, balance: float):
        """Ajusta os thresholds para balancear precisão e recall.
        balance: float entre 0.0 (mais recall) e 1.0 (mais precisão).
        """
        if not 0.0 <= balance <= 1.0:
            raise ValueError("Balance deve ser entre 0.0 e 1.0")
        
        self.precision_recall_balance = balance
        
        # Ajuste linear simples para demonstração
        # Em um cenário real, isso seria baseado em curvas PR ou otimização bayesiana
        self.high_risk_threshold = 0.35 + (balance * 0.20) # Aumenta para mais precisão
        self.medium_risk_threshold = 0.20 + (balance * 0.10) # Aumenta para mais precisão
        
        print(f"🎯 Thresholds ajustados para balance={balance}: Alto ≥{self.high_risk_threshold:.2f}, Médio ≥{self.medium_risk_threshold:.2f}")
        
        # Cache e histórico
        self.analysis_cache = {}
        self.cpf_history = defaultdict(list)
        self.global_stats = {
            'total_analyzed': 0,
            'fraud_detected': 0,
            'avg_fraud_score': 0.0
        }
        
        # Padrões de fraude mais agressivos
        self.fraud_patterns = self._initialize_aggressive_patterns()
        
        print("🔍 Analisador de Fraude Final v3.0 inicializado")
        print(f"🎯 Thresholds Agressivos: Alto ≥{self.high_risk_threshold}, Médio ≥{self.medium_risk_threshold}")
        print(f"🤖 Ensemble de {len(self.ensemble_weights)} modelos")
    
    def _initialize_aggressive_patterns(self):
        """Inicializa padrões mais agressivos de detecção"""
        return {
            'high_value_any': {'valor_min': 5000, 'risk_boost': 0.25},
            'very_high_value': {'valor_min': 20000, 'risk_boost': 0.40},
            'night_any': {'hora_min': 22, 'hora_max': 6, 'risk_boost': 0.30},
            'dawn_transactions': {'hora_min': 2, 'hora_max': 5, 'risk_boost': 0.45},
            'internet_medium': {'canal': 'INTERNET', 'valor_min': 2000, 'risk_boost': 0.25},
            'mobile_high': {'canal': 'MOBILE', 'valor_min': 5000, 'risk_boost': 0.30},
            'round_amounts': {'modulo': [100, 500, 1000], 'risk_boost': 0.20},
            'sequential_cpf': {'pattern': 'sequential', 'risk_boost': 0.15},
            'weekend_activity': {'is_weekend': True, 'valor_min': 1000, 'risk_boost': 0.20}
        }
    
    def analyze_transaction(self, transaction):
        """Analisa transação com ensemble de modelos"""
        try:
            # Gerar hash para cache
            tx_hash = self._generate_transaction_hash(transaction)
            
            # Cache com probabilidade reduzida para mais análises frescas
            if tx_hash in self.analysis_cache and random.random() < 0.1:
                return self.analysis_cache[tx_hash]
            
            # Calcular scores com ensemble
            ensemble_scores = []
            for i, weights in enumerate(self.ensemble_weights):
                score = self._calculate_model_score(transaction, weights, i)
                ensemble_scores.append(score)
            
            # Combinar scores do ensemble (média ponderada com boost para scores altos)
            base_score = np.mean(ensemble_scores)
            max_score = np.max(ensemble_scores)
            min_score = np.min(ensemble_scores)
            
            # Score final com boost para consenso alto
            if max_score > 0.7:
                fraud_score = (base_score * 0.7) + (max_score * 0.3)
            else:
                fraud_score = (base_score * 0.8) + (max_score * 0.2)
            
            # Aplicar boost de padrões agressivos
            pattern_boost = self._apply_aggressive_pattern_boost(transaction)
            fraud_score = min(1.0, fraud_score + pattern_boost)
            
            # Aplicar boost baseado em histórico do CPF
            history_boost = self._apply_history_boost(transaction)
            fraud_score = min(1.0, fraud_score + history_boost)
            
            # Determinar status com thresholds agressivos
            if fraud_score >= self.high_risk_threshold:
                status = 'REJECT'
                risk_level = 'Alto'
            elif fraud_score >= self.medium_risk_threshold:
                status = 'REVIEW'
                risk_level = 'Médio'
            else:
                status = 'APPROVE'
                risk_level = 'Baixo'
            
            # Atualizar estatísticas globais
            self._update_global_stats(fraud_score, status)
            
            # Atualizar histórico do CPF
            self._update_cpf_history(transaction, fraud_score)
            
            # Resultado final
            result = {
                'transaction_id': transaction.get('id', 'unknown'),
                'fraud_score': fraud_score,
                'status': status,
                'risk_level': risk_level,
                'analysis_timestamp': datetime.now().isoformat(),
                'ensemble_scores': ensemble_scores,
                'pattern_boost': pattern_boost,
                'history_boost': history_boost,
                'factors': self._get_comprehensive_risk_factors(transaction, fraud_score),
                'explanation': self._explain_fraud_score(transaction, ensemble_scores, fraud_score)
            }
            
            # Cache limitado
            if len(self.analysis_cache) < 3000:
                self.analysis_cache[tx_hash] = result
            
            return result
            
        except Exception as e:
            # Em caso de erro, ser mais conservador (assumir risco médio)
            return {
                'transaction_id': transaction.get('id', 'unknown'),
                'fraud_score': 0.4,  # Score médio-alto em caso de erro
                'status': 'REVIEW',
                'risk_level': 'Médio',
                'analysis_timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _calculate_model_score(self, transaction, weights, model_id):
        """Calcula score para um modelo específico do ensemble"""
        
        # Fatores base
        valor_risk = self._analyze_value_risk_aggressive(transaction.get('valor', 0))
        horario_risk = self._analyze_time_risk_aggressive(transaction.get('data_hora', ''))
        canal_risk = self._analyze_channel_risk_aggressive(transaction.get('canal', ''))
        location_risk = self._analyze_location_risk_aggressive(transaction.get('localizacao', ''))
        frequency_risk = self._analyze_frequency_risk_aggressive(transaction.get('cpf', ''))
        behavioral_risk = self._analyze_behavioral_risk_aggressive(transaction)
        
        # Score ponderado para este modelo
        score = (
            valor_risk * weights['valor'] +
            horario_risk * weights['horario'] +
            canal_risk * weights['canal'] +
            location_risk * weights['localizacao'] +
            frequency_risk * weights['frequencia'] +
            behavioral_risk * weights['comportamental']
        )
        
        # Aplicar função de ativação específica por modelo
        if model_id == 0:  # Modelo de valor - mais sensível a valores altos
            score = self._sigmoid(score * 2.5 - 1.2)
        elif model_id == 1:  # Modelo comportamental - mais sensível a padrões
            score = self._sigmoid(score * 2.2 - 1.0)
        elif model_id == 2:  # Modelo de canal - mais sensível a canais de risco
            score = self._sigmoid(score * 2.0 - 0.8)
        elif model_id == 3:  # Modelo balanceado
            score = self._sigmoid(score * 2.0 - 1.0)
        else:  # Modelo temporal - mais sensível a horários
            score = self._sigmoid(score * 2.3 - 1.1)
        
        return score
    
    def _analyze_value_risk_aggressive(self, valor):
        """Análise de valor mais agressiva"""
        if valor <= 0:
            return 0.9
        
        # Curva mais agressiva
        if valor > 100000:
            return 0.98
        elif valor > 50000:
            return 0.90
        elif valor > 20000:
            return 0.80
        elif valor > 10000:
            return 0.65
        elif valor > 5000:
            return 0.50
        elif valor > 2000:
            return 0.35
        elif valor > 1000:
            return 0.25
        elif valor > 500:
            return 0.20
        elif valor < 1:
            return 0.70
        elif valor < 10:
            return 0.40
        else:
            return 0.15
    
    def _analyze_time_risk_aggressive(self, data_hora):
        """Análise temporal mais agressiva"""
        try:
            if isinstance(data_hora, str) and ':' in data_hora:
                hora = int(data_hora.split(' ')[-1].split(':')[0])
            else:
                hora = 12
            
            # Muito mais agressivo com horários suspeitos
            if 2 <= hora <= 4:
                return 0.95
            elif 0 <= hora <= 1 or hora == 23:
                return 0.85
            elif 5 <= hora <= 6:
                return 0.70
            elif 22 <= hora <= 22:
                return 0.60
            elif 7 <= hora <= 8:
                return 0.35
            elif 19 <= hora <= 21:
                return 0.40
            elif 9 <= hora <= 17:
                return 0.15
            else:
                return 0.30
                
        except:
            return 0.50
    
    def _analyze_channel_risk_aggressive(self, canal):
        """Análise de canal mais agressiva"""
        canal = canal.upper() if canal else ''
        
        # Riscos aumentados
        channel_risks = {
            'INTERNET': 0.75,
            'MOBILE': 0.65,
            'ATM': 0.55,
            'POS': 0.45,
            'AGENCIA': 0.20,
            'CALL_CENTER': 0.80,
            'UNKNOWN': 0.90
        }
        
        return channel_risks.get(canal, 0.60)
    
    def _analyze_location_risk_aggressive(self, localizacao):
        """Análise de localização mais agressiva"""
        if not localizacao:
            return 0.70
        
        localizacao_upper = localizacao.upper()
        
        # Padrões mais específicos
        high_risk_patterns = ['EXTERIOR', 'FRONTEIRA', 'RURAL', 'DESCONHECIDO']
        medium_risk_patterns = ['INTERIOR', 'PERIFERIA', 'SUBURBIO']
        
        for pattern in high_risk_patterns:
            if pattern in localizacao_upper:
                return 0.85
        
        for pattern in medium_risk_patterns:
            if pattern in localizacao_upper:
                return 0.55
        
        return 0.25
    
    def _analyze_frequency_risk_aggressive(self, cpf):
        """Análise de frequência mais agressiva"""
        if not cpf:
            return 0.70
        
        # Verificar histórico
        if cpf in self.cpf_history:
            scores = self.cpf_history[cpf]
            if len(scores) > 0:
                avg_score = np.mean(scores)
                recent_scores = scores[-5:] if len(scores) >= 5 else scores
                recent_avg = np.mean(recent_scores)
                
                # CPF com histórico de scores altos
                if recent_avg > 0.6:
                    return 0.85
                elif recent_avg > 0.4:
                    return 0.60
                elif avg_score > 0.5:
                    return 0.70
        
        # Análise baseada no hash (mais agressiva)
        cpf_hash = hashlib.md5(cpf.encode()).hexdigest()
        frequency_indicator = int(cpf_hash[:2], 16) / 255.0
        
        if frequency_indicator > 0.80:
            return 0.80
        elif frequency_indicator > 0.60:
            return 0.60
        elif frequency_indicator < 0.20:
            return 0.45  # CPFs "muito normais" também podem ser suspeitos
        else:
            return 0.30
    
    def _analyze_behavioral_risk_aggressive(self, transaction):
        """Análise comportamental mais agressiva"""
        risk_score = 0.0
        
        valor = transaction.get('valor', 0)
        canal = transaction.get('canal', '').upper()
        tipo = transaction.get('tipo', '').upper()
        
        # Padrões mais agressivos
        if valor > 15000 and canal == 'INTERNET':
            risk_score += 0.50
        
        if valor > 30000 and canal in ['MOBILE', 'ATM']:
            risk_score += 0.60
        
        if tipo == 'PIX' and valor > 5000:
            risk_score += 0.40
        
        if tipo == 'TED' and valor > 10000:
            risk_score += 0.35
        
        if canal == 'ATM' and valor > 8000:
            risk_score += 0.50
        
        # Valores redondos mais suspeitos
        if valor >= 500:
            if valor % 1000 == 0:
                risk_score += 0.35
            elif valor % 500 == 0:
                risk_score += 0.25
            elif valor % 100 == 0:
                risk_score += 0.15
        
        # Combinações específicas
        if canal == 'INTERNET' and tipo == 'PIX' and valor > 3000:
            risk_score += 0.30
        
        return min(1.0, risk_score)
    
    def _apply_aggressive_pattern_boost(self, transaction):
        """Aplica boost agressivo baseado em padrões"""
        boost = 0.0
        
        valor = transaction.get('valor', 0)
        canal = transaction.get('canal', '').upper()
        
        # Padrões mais agressivos
        if valor >= 5000:
            boost += self.fraud_patterns['high_value_any']['risk_boost']
        
        if valor >= 20000:
            boost += self.fraud_patterns['very_high_value']['risk_boost']
        
        if canal == 'INTERNET' and valor >= 2000:
            boost += self.fraud_patterns['internet_medium']['risk_boost']
        
        if canal == 'MOBILE' and valor >= 5000:
            boost += self.fraud_patterns['mobile_high']['risk_boost']
        
        # Horário
        try:
            data_hora = transaction.get('data_hora', '')
            if ':' in data_hora:
                hora = int(data_hora.split(' ')[-1].split(':')[0])
                if 22 <= hora <= 23 or 0 <= hora <= 6:
                    boost += self.fraud_patterns['night_any']['risk_boost']
                if 2 <= hora <= 5:
                    boost += self.fraud_patterns['dawn_transactions']['risk_boost']
        except:
            pass
        
        # Valores redondos
        if valor >= 100:
            for modulo in self.fraud_patterns['round_amounts']['modulo']:
                if valor % modulo == 0:
                    boost += self.fraud_patterns['round_amounts']['risk_boost']
                    break
        
        return min(0.6, boost)  # Boost máximo aumentado
    
    def _apply_history_boost(self, transaction):
        """Aplica boost baseado no histórico do CPF"""
        cpf = transaction.get('cpf', '')
        if not cpf or cpf not in self.cpf_history:
            return 0.0
        
        scores = self.cpf_history[cpf]
        if len(scores) < 3:
            return 0.0
        
        # Boost baseado na tendência recente
        recent_scores = scores[-3:]
        recent_avg = np.mean(recent_scores)
        
        if recent_avg > 0.6:
            return 0.25
        elif recent_avg > 0.4:
            return 0.15
        elif recent_avg < 0.2:
            return -0.10  # Redução para CPFs consistentemente baixos
        
        return 0.0
    
    def _update_global_stats(self, fraud_score, status):
        """Atualiza estatísticas globais"""
        self.global_stats['total_analyzed'] += 1
        if status in ['REJECT', 'REVIEW']:
            self.global_stats['fraud_detected'] += 1
        
        # Média móvel do score
        total = self.global_stats['total_analyzed']
        current_avg = self.global_stats['avg_fraud_score']
        self.global_stats['avg_fraud_score'] = ((current_avg * (total - 1)) + fraud_score) / total
    
    def _update_cpf_history(self, transaction, fraud_score):
        """Atualiza histórico do CPF"""
        cpf = transaction.get('cpf', '')
        if not cpf:
            return
        
        self.cpf_history[cpf].append(fraud_score)
        
        # Manter apenas os últimos 20 scores
        if len(self.cpf_history[cpf]) > 20:
            self.cpf_history[cpf] = self.cpf_history[cpf][-20:]
    
    def _sigmoid(self, x):
        """Função sigmoide"""
        return 1 / (1 + math.exp(-x))
    
    def _generate_transaction_hash(self, transaction):
        """Gera hash da transação"""
        key_fields = [
            str(transaction.get('valor', 0)),
            str(transaction.get('cpf', '')),
            str(transaction.get('canal', '')),
            str(transaction.get('tipo', '')),
            str(transaction.get('localizacao', ''))
        ]
        hash_input = '|'.join(key_fields)
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_comprehensive_risk_factors(self, transaction, fraud_score):
        """Retorna fatores de risco abrangentes"""
        factors = []
        
        valor = transaction.get('valor', 0)
        canal = transaction.get('canal', '').upper()
        
        if valor > 50000:
            factors.append(f'Valor extremamente alto: R$ {valor:,.2f}')
        elif valor > 20000:
            factors.append(f'Valor muito alto: R$ {valor:,.2f}')
        elif valor > 5000:
            factors.append(f'Valor alto: R$ {valor:,.2f}')
        
        if canal in ['INTERNET', 'MOBILE'] and valor > 2000:
            factors.append(f'Transação {canal.lower()} de valor elevado')
        
        if fraud_score > 0.7:
            factors.append('Múltiplos indicadores críticos de risco')
        elif fraud_score > 0.5:
            factors.append('Vários indicadores de risco detectados')
        elif fraud_score > 0.3:
            factors.append('Alguns indicadores de risco presentes')
        
        # Verificar horário
        try:
            data_hora = transaction.get('data_hora', '')
            if ':' in data_hora:
                hora = int(data_hora.split(' ')[-1].split(':')[0])
                if 2 <= hora <= 4:
                    factors.append('Transação em horário crítico (madrugada)')
                elif 22 <= hora <= 1:
                    factors.append('Transação em horário suspeito (noturno)')
        except:
            pass
        
        # Valores redondos
        if valor >= 100 and valor % 1000 == 0:
            factors.append('Valor redondo suspeito')
        
        return factors if factors else ['Transação dentro dos padrões aceitáveis']
    
    def get_system_stats(self):
        """Retorna estatísticas do sistema"""
        return {
            'global_stats': self.global_stats,
            'cache_size': len(self.analysis_cache),
            'cpf_history_size': len(self.cpf_history),
            'thresholds': {
                'high_risk': self.high_risk_threshold,
                'medium_risk': self.medium_risk_threshold
            },
            'ensemble_models': len(self.ensemble_weights)
        }


    def _explain_fraud_score(self, transaction, ensemble_scores, final_score):
        """Gera uma explicação humanamente legível para o score de fraude."""
        explanation = []
        
        # Explicação baseada no score final
        if final_score >= self.high_risk_threshold:
            explanation.append(f"**Decisão Final: ALTO RISCO de fraude ({final_score:.2f})** - A transação foi rejeitada devido a múltiplos indicadores críticos.")
        elif final_score >= self.medium_risk_threshold:
            explanation.append(f"**Decisão Final: MÉDIO RISCO de fraude ({final_score:.2f})** - A transação requer revisão manual.")
        else:
            explanation.append(f"**Decisão Final: BAIXO RISCO de fraude ({final_score:.2f})** - A transação foi aprovada.")
            
        explanation.append("\n**Fatores Contribuintes:**")

        # Contribuição dos modelos do ensemble
        model_names = [
            "Modelo de Valor", "Modelo Comportamental", "Modelo de Canal",
            "Modelo Balanceado", "Modelo Temporal"
        ]
        for i, score in enumerate(ensemble_scores):
            if score > 0.5:
                explanation.append(f"- O {model_names[i]} indicou um risco elevado ({score:.2f}).")

        # Análise dos fatores de risco individuais
        valor = transaction.get("valor", 0)
        canal = transaction.get("canal", "").upper()
        hora = int(datetime.fromisoformat(transaction.get("data_hora", datetime.now().isoformat())).hour)
        localizacao = transaction.get("localizacao", "").upper()
        cpf = transaction.get("cpf", "")

        if valor > 20000:
            explanation.append(f"- **Valor Elevado**: Transação de R${valor:,.2f} (acima do padrão).")
        elif valor < 1:
            explanation.append(f"- **Valor Irrisório**: Transação de R${valor:,.2f} (pode indicar teste de cartão).")

        if 2 <= hora <= 5:
            explanation.append(f"- **Horário Suspeito**: Transação realizada na madrugada ({hora}h).")
        elif 22 <= hora <= 23 or 0 <= hora <= 1:
            explanation.append(f"- **Horário Noturno**: Transação realizada em período de baixo movimento ({hora}h).")

        if canal in ["INTERNET", "MOBILE", "CALL_CENTER"] and valor > 5000:
            explanation.append(f"- **Canal de Risco**: Transação de alto valor via {canal}.")

        if "EXTERIOR" in localizacao or "FRONTEIRA" in localizacao:
            explanation.append(f"- **Localização de Risco**: Transação em área de alto risco ({localizacao}).")

        # Histórico do CPF
        if cpf in self.cpf_history:
            scores_history = self.cpf_history[cpf]
            if len(scores_history) > 3 and np.mean(scores_history[-3:]) > 0.6:
                explanation.append(f"- **Histórico do CPF**: O CPF {cpf} possui histórico recente de transações de alto risco.")

        # Padrões agressivos
        if valor >= 500 and (valor % 1000 == 0 or valor % 500 == 0):
            explanation.append(f"- **Valor Redondo**: Transação com valor redondo (R${valor:,.2f}), um padrão comum em fraudes.")

        return "\n".join(explanation)

    def _get_comprehensive_risk_factors(self, transaction, fraud_score):
        """Retorna uma lista de fatores de risco mais detalhada."""
        factors = []
        valor = transaction.get("valor", 0)
        canal = transaction.get("canal", "").upper()
        tipo = transaction.get("tipo", "").upper()
        data_hora_str = transaction.get("data_hora", datetime.now().isoformat())
        hora = int(datetime.fromisoformat(data_hora_str).hour)
        localizacao = transaction.get("localizacao", "").upper()
        cpf = transaction.get("cpf", "")

        if fraud_score >= self.high_risk_threshold:
            factors.append("Múltiplos indicadores críticos de risco")
        elif fraud_score >= self.medium_risk_threshold:
            factors.append("Combinação de indicadores de risco médio")
        else:
            factors.append("Indicadores de risco baixos ou ausentes")

        if valor > 100000:
            factors.append(f"Valor extremamente alto: R${valor:,.2f}")
        elif valor > 20000:
            factors.append(f"Valor alto: R${valor:,.2f}")
        elif valor < 1:
            factors.append(f"Valor irrisório: R${valor:,.2f}")

        if 2 <= hora <= 4:
            factors.append(f"Transação na madrugada ({hora}h)")
        elif 0 <= hora <= 1 or hora == 23:
            factors.append(f"Transação em horário noturno ({hora}h)")

        if canal in ["INTERNET", "MOBILE", "CALL_CENTER"]:
            factors.append(f"Canal de alto risco: {canal}")

        if "EXTERIOR" in localizacao or "FRONTEIRA" in localizacao or "DESCONHECIDO" in localizacao:
            factors.append(f"Localização de alto risco: {localizacao}")

        if valor >= 500 and (valor % 1000 == 0 or valor % 500 == 0 or valor % 100 == 0):
            factors.append(f"Valor redondo: R${valor:,.2f}")

        if cpf in self.cpf_history and len(self.cpf_history[cpf]) > 0:
            avg_recent_score = np.mean(self.cpf_history[cpf][-5:])
            if avg_recent_score > 0.6:
                factors.append(f"Histórico recente de alto risco para CPF {cpf}")
            elif avg_recent_score > 0.4:
                factors.append(f"Histórico recente de médio risco para CPF {cpf}")

        return factors
