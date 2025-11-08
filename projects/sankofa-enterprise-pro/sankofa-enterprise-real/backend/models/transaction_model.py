from dataclasses import dataclass

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

