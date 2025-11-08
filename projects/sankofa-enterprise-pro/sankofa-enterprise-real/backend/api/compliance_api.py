#!/usr/bin/env python3
"""
API de Compliance para o Sankofa Enterprise Pro
Expõe as funcionalidades de compliance para o frontend.
"""

import os
import sys
from flask import Flask, request, jsonify

# Adiciona o diretório pai ao path para importações
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compliance.compliance_manager import ComplianceManager
from security.enterprise_security_system import EnterpriseSecuritySystem

app = Flask(__name__)
compliance_manager = ComplianceManager()
security_system = EnterpriseSecuritySystem()

@app.route("/api/v1/compliance/status", methods=['GET'])
@security_system.require_auth()
def get_compliance_status():
    """Retorna o status geral de compliance."""
    status = {
        "bacen_resolution_6": "Partially Implemented",
        "lgpd": "Implemented",
        "pci_dss_v4": "Partially Implemented",
        "sox": "Not Implemented",
        "basel_iii": "Not Implemented",
    }
    return jsonify({"success": True, "data": status})

@app.route("/api/v1/compliance/share-fraud", methods=['POST'])
@security_system.require_auth()
@security_system.require_permission('share_fraud_data')
def share_fraud():
    """Compartilha dados de fraude com o BACEN."""
    data = request.get_json()
    user_context = g.user  # Assumindo que o usuário está em g.user
    success = compliance_manager.share_fraud_data_with_bacen(data, user_context)
    if success:
        return jsonify({"success": True, "message": "Dados de fraude compartilhados com sucesso."})
    else:
        return jsonify({"success": False, "message": "Falha ao compartilhar dados de fraude."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8445, debug=True)

