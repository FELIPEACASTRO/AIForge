"""
Sistema Avançado de Personalização e Configuração para Sankofa Enterprise Pro
Permite que usuários de negócio ajustem regras, thresholds e visualizem impactos em tempo real
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConfigurationRule:
    """Representa uma regra de configuração"""
    rule_id: str
    rule_name: str
    rule_type: str  # 'threshold', 'business_rule', 'ml_parameter'
    category: str  # 'fraud_detection', 'performance', 'compliance'
    current_value: Union[float, int, str, bool]
    default_value: Union[float, int, str, bool]
    min_value: Optional[Union[float, int]] = None
    max_value: Optional[Union[float, int]] = None
    allowed_values: Optional[List[Any]] = None
    description: str = ""
    impact_level: str = "medium"  # 'low', 'medium', 'high', 'critical'
    requires_approval: bool = False
    last_modified: Optional[datetime.datetime] = None
    modified_by: Optional[str] = None

@dataclass
class ConfigurationChange:
    """Representa uma mudança de configuração"""
    change_id: str
    rule_id: str
    old_value: Union[float, int, str, bool]
    new_value: Union[float, int, str, bool]
    changed_by: str
    changed_at: datetime.datetime
    reason: str
    status: str  # 'pending', 'approved', 'rejected', 'applied'
    impact_simulation: Dict[str, Any]

@dataclass
class ImpactSimulation:
    """Resultado de simulação de impacto"""
    rule_id: str
    old_value: Union[float, int, str, bool]
    new_value: Union[float, int, str, bool]
    estimated_impact: Dict[str, float]
    confidence_level: float
    simulation_timestamp: datetime.datetime

class AdvancedConfigurationSystem:
    """
    Sistema avançado de configuração e personalização
    """
    
    def __init__(self, config_path: str = "/home/ubuntu/sankofa-enterprise-real/config"):
        self.config_path = Path(config_path)
        self.config_path.mkdir(exist_ok=True)
        
        # Arquivos de configuração
        self.rules_file = self.config_path / "configuration_rules.json"
        self.changes_file = self.config_path / "configuration_changes.json"
        
        # Estado do sistema
        self.rules = self._load_rules()
        self.changes_history = self._load_changes_history()
        
        # Salva regras se foram criadas pela primeira vez
        if not self.rules_file.exists():
            self._save_rules()
        
        # Cache de simulações
        self.simulation_cache = {}
        
        logger.info("Sistema Avançado de Configuração inicializado")
    
    def _load_rules(self) -> Dict[str, ConfigurationRule]:
        """Carrega regras de configuração"""
        if self.rules_file.exists():
            with open(self.rules_file, 'r') as f:
                data = json.load(f)
                rules = {}
                for rule_id, rule_data in data.items():
                    if rule_data.get('last_modified'):
                        rule_data['last_modified'] = datetime.datetime.fromisoformat(rule_data['last_modified'])
                    rules[rule_id] = ConfigurationRule(**rule_data)
                return rules
        
        # Regras padrão
        return self._create_default_rules()
    
    def _create_default_rules(self) -> Dict[str, ConfigurationRule]:
        """Cria regras de configuração padrão"""
        default_rules = {
            # Thresholds de Detecção de Fraude
            "fraud_high_risk_threshold": ConfigurationRule(
                rule_id="fraud_high_risk_threshold",
                rule_name="Threshold Alto Risco",
                rule_type="threshold",
                category="fraud_detection",
                current_value=0.8,
                default_value=0.8,
                min_value=0.5,
                max_value=0.95,
                description="Threshold para classificar transação como alto risco de fraude",
                impact_level="high",
                requires_approval=True
            ),
            "fraud_medium_risk_threshold": ConfigurationRule(
                rule_id="fraud_medium_risk_threshold",
                rule_name="Threshold Médio Risco",
                rule_type="threshold",
                category="fraud_detection",
                current_value=0.5,
                default_value=0.5,
                min_value=0.2,
                max_value=0.8,
                description="Threshold para classificar transação como médio risco de fraude",
                impact_level="medium",
                requires_approval=False
            ),
            "fraud_low_risk_threshold": ConfigurationRule(
                rule_id="fraud_low_risk_threshold",
                rule_name="Threshold Baixo Risco",
                rule_type="threshold",
                category="fraud_detection",
                current_value=0.2,
                default_value=0.2,
                min_value=0.05,
                max_value=0.5,
                description="Threshold para classificar transação como baixo risco de fraude",
                impact_level="low",
                requires_approval=False
            ),
            
            # Regras de Negócio
            "max_transaction_amount": ConfigurationRule(
                rule_id="max_transaction_amount",
                rule_name="Valor Máximo de Transação",
                rule_type="business_rule",
                category="fraud_detection",
                current_value=50000.0,
                default_value=50000.0,
                min_value=1000.0,
                max_value=1000000.0,
                description="Valor máximo permitido para transações sem revisão manual",
                impact_level="high",
                requires_approval=True
            ),
            "daily_transaction_limit": ConfigurationRule(
                rule_id="daily_transaction_limit",
                rule_name="Limite Diário de Transações",
                rule_type="business_rule",
                category="fraud_detection",
                current_value=10,
                default_value=10,
                min_value=1,
                max_value=100,
                description="Número máximo de transações por CPF por dia",
                impact_level="medium",
                requires_approval=False
            ),
            "suspicious_hour_start": ConfigurationRule(
                rule_id="suspicious_hour_start",
                rule_name="Início Horário Suspeito",
                rule_type="business_rule",
                category="fraud_detection",
                current_value=23,
                default_value=23,
                min_value=0,
                max_value=23,
                description="Hora de início do período considerado suspeito",
                impact_level="low",
                requires_approval=False
            ),
            "suspicious_hour_end": ConfigurationRule(
                rule_id="suspicious_hour_end",
                rule_name="Fim Horário Suspeito",
                rule_type="business_rule",
                category="fraud_detection",
                current_value=6,
                default_value=6,
                min_value=0,
                max_value=23,
                description="Hora de fim do período considerado suspeito",
                impact_level="low",
                requires_approval=False
            ),
            
            # Parâmetros de Performance
            "max_response_time_ms": ConfigurationRule(
                rule_id="max_response_time_ms",
                rule_name="Tempo Máximo de Resposta",
                rule_type="ml_parameter",
                category="performance",
                current_value=50.0,
                default_value=50.0,
                min_value=10.0,
                max_value=1000.0,
                description="Tempo máximo de resposta para análise de fraude (ms)",
                impact_level="medium",
                requires_approval=False
            ),
            "batch_size": ConfigurationRule(
                rule_id="batch_size",
                rule_name="Tamanho do Batch",
                rule_type="ml_parameter",
                category="performance",
                current_value=100,
                default_value=100,
                min_value=10,
                max_value=1000,
                description="Tamanho do batch para processamento de transações",
                impact_level="low",
                requires_approval=False
            ),
            
            # Configurações de Compliance
            "data_retention_days": ConfigurationRule(
                rule_id="data_retention_days",
                rule_name="Retenção de Dados (dias)",
                rule_type="business_rule",
                category="compliance",
                current_value=2555,  # 7 anos
                default_value=2555,
                min_value=365,  # 1 ano mínimo
                max_value=3650,  # 10 anos máximo
                description="Período de retenção de dados de transações",
                impact_level="critical",
                requires_approval=True
            ),
            "audit_log_level": ConfigurationRule(
                rule_id="audit_log_level",
                rule_name="Nível de Log de Auditoria",
                rule_type="business_rule",
                category="compliance",
                current_value="INFO",
                default_value="INFO",
                allowed_values=["DEBUG", "INFO", "WARNING", "ERROR"],
                description="Nível de detalhamento dos logs de auditoria",
                impact_level="medium",
                requires_approval=False
            )
        }
        
        # Retorna regras padrão (serão salvas após inicialização)
        return default_rules
    
    def _load_changes_history(self) -> List[ConfigurationChange]:
        """Carrega histórico de mudanças"""
        if self.changes_file.exists():
            with open(self.changes_file, 'r') as f:
                data = json.load(f)
                changes = []
                for change_data in data:
                    change_data['changed_at'] = datetime.datetime.fromisoformat(change_data['changed_at'])
                    changes.append(ConfigurationChange(**change_data))
                return changes
        
        return []
    
    def _save_rules(self):
        """Salva regras de configuração"""
        data = {}
        for rule_id, rule in self.rules.items():
            rule_data = asdict(rule)
            if rule_data['last_modified']:
                rule_data['last_modified'] = rule.last_modified.isoformat()
            data[rule_id] = rule_data
        
        with open(self.rules_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_changes_history(self):
        """Salva histórico de mudanças"""
        data = []
        for change in self.changes_history:
            change_data = asdict(change)
            change_data['changed_at'] = change.changed_at.isoformat()
            
            # Converte datetime em impact_simulation se existir
            if 'impact_simulation' in change_data and 'simulation_timestamp' in change_data['impact_simulation']:
                if isinstance(change_data['impact_simulation']['simulation_timestamp'], datetime.datetime):
                    change_data['impact_simulation']['simulation_timestamp'] = change_data['impact_simulation']['simulation_timestamp'].isoformat()
            
            data.append(change_data)
        
        with open(self.changes_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_all_rules(self, category: Optional[str] = None) -> Dict[str, ConfigurationRule]:
        """Retorna todas as regras ou filtradas por categoria"""
        if category:
            return {
                rule_id: rule for rule_id, rule in self.rules.items()
                if rule.category == category
            }
        return self.rules.copy()
    
    def get_rule(self, rule_id: str) -> Optional[ConfigurationRule]:
        """Retorna uma regra específica"""
        return self.rules.get(rule_id)
    
    def simulate_impact(self, rule_id: str, new_value: Union[float, int, str, bool]) -> ImpactSimulation:
        """
        Simula o impacto de uma mudança de configuração
        """
        rule = self.rules.get(rule_id)
        if not rule:
            raise ValueError(f"Regra {rule_id} não encontrada")
        
        # Cache key para evitar recálculos
        cache_key = f"{rule_id}_{rule.current_value}_{new_value}"
        
        if cache_key in self.simulation_cache:
            return self.simulation_cache[cache_key]
        
        # Simula impacto baseado no tipo de regra
        estimated_impact = self._calculate_impact(rule, new_value)
        
        # Calcula nível de confiança
        confidence_level = self._calculate_confidence(rule, new_value)
        
        simulation = ImpactSimulation(
            rule_id=rule_id,
            old_value=rule.current_value,
            new_value=new_value,
            estimated_impact=estimated_impact,
            confidence_level=confidence_level,
            simulation_timestamp=datetime.datetime.now()
        )
        
        # Cache resultado
        self.simulation_cache[cache_key] = simulation
        
        return simulation
    
    def _calculate_impact(self, rule: ConfigurationRule, new_value: Union[float, int, str, bool]) -> Dict[str, float]:
        """Calcula impacto estimado da mudança"""
        impact = {}
        
        if rule.category == "fraud_detection":
            if rule.rule_type == "threshold":
                # Simula impacto em métricas de fraude
                if isinstance(new_value, (int, float)) and isinstance(rule.current_value, (int, float)):
                    threshold_change = (new_value - rule.current_value) / rule.current_value
                    
                    # Impacto em precision/recall (simulado)
                    if "high_risk" in rule.rule_id:
                        impact["precision_change"] = threshold_change * 0.1  # Threshold maior = mais precisão
                        impact["recall_change"] = -threshold_change * 0.15  # Threshold maior = menos recall
                        impact["false_positive_rate_change"] = -threshold_change * 0.2
                    elif "medium_risk" in rule.rule_id:
                        impact["precision_change"] = threshold_change * 0.05
                        impact["recall_change"] = -threshold_change * 0.1
                        impact["false_positive_rate_change"] = -threshold_change * 0.1
                    
                    # Impacto em volume de transações bloqueadas
                    impact["blocked_transactions_change"] = threshold_change * 0.3
            
            elif rule.rule_type == "business_rule":
                if "max_transaction_amount" in rule.rule_id:
                    if isinstance(new_value, (int, float)) and isinstance(rule.current_value, (int, float)):
                        amount_change = (new_value - rule.current_value) / rule.current_value
                        impact["high_value_transactions_blocked_change"] = -amount_change * 0.5
                        impact["customer_friction_change"] = -amount_change * 0.2
                
                elif "daily_transaction_limit" in rule.rule_id:
                    if isinstance(new_value, (int, float)) and isinstance(rule.current_value, (int, float)):
                        limit_change = (new_value - rule.current_value) / rule.current_value
                        impact["legitimate_transactions_blocked_change"] = -limit_change * 0.3
                        impact["fraud_detection_rate_change"] = limit_change * 0.1
        
        elif rule.category == "performance":
            if "response_time" in rule.rule_id:
                if isinstance(new_value, (int, float)) and isinstance(rule.current_value, (int, float)):
                    time_change = (new_value - rule.current_value) / rule.current_value
                    impact["system_throughput_change"] = -time_change * 0.2
                    impact["user_experience_change"] = -time_change * 0.3
            
            elif "batch_size" in rule.rule_id:
                if isinstance(new_value, (int, float)) and isinstance(rule.current_value, (int, float)):
                    batch_change = (new_value - rule.current_value) / rule.current_value
                    impact["processing_efficiency_change"] = batch_change * 0.1
                    impact["memory_usage_change"] = batch_change * 0.2
        
        elif rule.category == "compliance":
            if "data_retention" in rule.rule_id:
                if isinstance(new_value, (int, float)) and isinstance(rule.current_value, (int, float)):
                    retention_change = (new_value - rule.current_value) / rule.current_value
                    impact["storage_cost_change"] = retention_change * 0.8
                    impact["compliance_risk_change"] = -retention_change * 0.1
        
        return impact
    
    def _calculate_confidence(self, rule: ConfigurationRule, new_value: Union[float, int, str, bool]) -> float:
        """Calcula nível de confiança da simulação"""
        # Confiança baseada em fatores como:
        # - Histórico de mudanças similares
        # - Proximidade do valor padrão
        # - Tipo de regra
        
        base_confidence = 0.7
        
        # Ajusta baseado na proximidade do valor padrão
        if isinstance(new_value, (int, float)) and isinstance(rule.default_value, (int, float)):
            if rule.default_value != 0:
                deviation = abs(new_value - rule.default_value) / abs(rule.default_value)
                confidence_penalty = min(deviation * 0.2, 0.3)
                base_confidence -= confidence_penalty
        
        # Ajusta baseado no nível de impacto
        if rule.impact_level == "critical":
            base_confidence -= 0.1
        elif rule.impact_level == "low":
            base_confidence += 0.1
        
        return max(0.1, min(1.0, base_confidence))
    
    def propose_change(self, rule_id: str, new_value: Union[float, int, str, bool], 
                      changed_by: str, reason: str) -> str:
        """Propõe uma mudança de configuração"""
        rule = self.rules.get(rule_id)
        if not rule:
            raise ValueError(f"Regra {rule_id} não encontrada")
        
        # Valida novo valor
        self._validate_value(rule, new_value)
        
        # Simula impacto
        impact_simulation = self.simulate_impact(rule_id, new_value)
        
        # Cria mudança
        change_id = f"change_{rule_id}_{int(datetime.datetime.now().timestamp())}"
        
        change = ConfigurationChange(
            change_id=change_id,
            rule_id=rule_id,
            old_value=rule.current_value,
            new_value=new_value,
            changed_by=changed_by,
            changed_at=datetime.datetime.now(),
            reason=reason,
            status='pending' if rule.requires_approval else 'approved',
            impact_simulation=asdict(impact_simulation)
        )
        
        self.changes_history.append(change)
        self._save_changes_history()
        
        # Se não requer aprovação, aplica imediatamente
        if not rule.requires_approval:
            self.apply_change(change_id)
        
        logger.info(f"Mudança proposta: {change_id} para regra {rule_id}")
        return change_id
    
    def _validate_value(self, rule: ConfigurationRule, value: Union[float, int, str, bool]):
        """Valida se o valor está dentro dos limites permitidos"""
        if rule.allowed_values and value not in rule.allowed_values:
            raise ValueError(f"Valor {value} não está na lista de valores permitidos: {rule.allowed_values}")
        
        if isinstance(value, (int, float)):
            if rule.min_value is not None and value < rule.min_value:
                raise ValueError(f"Valor {value} é menor que o mínimo permitido: {rule.min_value}")
            
            if rule.max_value is not None and value > rule.max_value:
                raise ValueError(f"Valor {value} é maior que o máximo permitido: {rule.max_value}")
    
    def approve_change(self, change_id: str, approved_by: str) -> bool:
        """Aprova uma mudança pendente"""
        change = self._find_change(change_id)
        if not change:
            return False
        
        if change.status != 'pending':
            logger.warning(f"Mudança {change_id} não está pendente")
            return False
        
        change.status = 'approved'
        self._save_changes_history()
        
        # Aplica mudança
        return self.apply_change(change_id)
    
    def reject_change(self, change_id: str, rejected_by: str, reason: str) -> bool:
        """Rejeita uma mudança pendente"""
        change = self._find_change(change_id)
        if not change:
            return False
        
        change.status = 'rejected'
        change.reason += f" | Rejeitado por {rejected_by}: {reason}"
        self._save_changes_history()
        
        logger.info(f"Mudança {change_id} rejeitada")
        return True
    
    def apply_change(self, change_id: str) -> bool:
        """Aplica uma mudança aprovada"""
        change = self._find_change(change_id)
        if not change:
            return False
        
        if change.status not in ['approved', 'pending']:
            logger.warning(f"Mudança {change_id} não pode ser aplicada (status: {change.status})")
            return False
        
        rule = self.rules.get(change.rule_id)
        if not rule:
            logger.error(f"Regra {change.rule_id} não encontrada")
            return False
        
        # Aplica mudança
        rule.current_value = change.new_value
        rule.last_modified = datetime.datetime.now()
        rule.modified_by = change.changed_by
        
        change.status = 'applied'
        
        # Salva alterações
        self._save_rules()
        self._save_changes_history()
        
        logger.info(f"Mudança {change_id} aplicada com sucesso")
        return True
    
    def _find_change(self, change_id: str) -> Optional[ConfigurationChange]:
        """Encontra uma mudança pelo ID"""
        for change in self.changes_history:
            if change.change_id == change_id:
                return change
        return None
    
    def get_pending_changes(self) -> List[ConfigurationChange]:
        """Retorna mudanças pendentes de aprovação"""
        return [change for change in self.changes_history if change.status == 'pending']
    
    def get_changes_history(self, rule_id: Optional[str] = None, 
                          limit: int = 50) -> List[ConfigurationChange]:
        """Retorna histórico de mudanças"""
        changes = self.changes_history
        
        if rule_id:
            changes = [change for change in changes if change.rule_id == rule_id]
        
        # Ordena por data (mais recente primeiro)
        changes.sort(key=lambda x: x.changed_at, reverse=True)
        
        return changes[:limit]
    
    def reset_rule_to_default(self, rule_id: str, reset_by: str) -> bool:
        """Reseta uma regra para o valor padrão"""
        rule = self.rules.get(rule_id)
        if not rule:
            return False
        
        return self.propose_change(
            rule_id=rule_id,
            new_value=rule.default_value,
            changed_by=reset_by,
            reason="Reset para valor padrão"
        ) is not None
    
    def export_configuration(self) -> Dict[str, Any]:
        """Exporta configuração atual"""
        return {
            "export_timestamp": datetime.datetime.now().isoformat(),
            "rules": {rule_id: asdict(rule) for rule_id, rule in self.rules.items()},
            "recent_changes": [
                asdict(change) for change in self.get_changes_history(limit=20)
            ]
        }
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Retorna resumo da configuração"""
        categories = {}
        for rule in self.rules.values():
            if rule.category not in categories:
                categories[rule.category] = {
                    "total_rules": 0,
                    "modified_rules": 0,
                    "critical_rules": 0
                }
            
            categories[rule.category]["total_rules"] += 1
            
            if rule.last_modified:
                categories[rule.category]["modified_rules"] += 1
            
            if rule.impact_level == "critical":
                categories[rule.category]["critical_rules"] += 1
        
        return {
            "total_rules": len(self.rules),
            "categories": categories,
            "pending_changes": len(self.get_pending_changes()),
            "recent_changes": len([
                c for c in self.changes_history 
                if c.changed_at > datetime.datetime.now() - datetime.timedelta(days=7)
            ])
        }

def test_advanced_configuration_system():
    """
    Testa o sistema avançado de configuração
    """
    print("🚀 Testando Sistema Avançado de Configuração...")
    
    # Inicializa sistema
    config_system = AdvancedConfigurationSystem()
    
    # Lista todas as regras
    all_rules = config_system.get_all_rules()
    print(f"✅ Total de regras carregadas: {len(all_rules)}")
    
    # Simula impacto de mudança
    simulation = config_system.simulate_impact("fraud_high_risk_threshold", 0.9)
    print(f"✅ Simulação de impacto - Confiança: {simulation.confidence_level:.2f}")
    
    # Propõe mudança
    change_id = config_system.propose_change(
        rule_id="fraud_medium_risk_threshold",
        new_value=0.6,
        changed_by="admin",
        reason="Ajuste para reduzir falsos positivos"
    )
    print(f"✅ Mudança proposta: {change_id}")
    
    # Verifica mudanças pendentes
    pending = config_system.get_pending_changes()
    print(f"✅ Mudanças pendentes: {len(pending)}")
    
    # Gera resumo
    summary = config_system.get_configuration_summary()
    print(f"✅ Resumo gerado - Categorias: {len(summary['categories'])}")
    
    print("🎉 Teste do Sistema Avançado de Configuração concluído!")
    
    return config_system, change_id

if __name__ == "__main__":
    test_advanced_configuration_system()
