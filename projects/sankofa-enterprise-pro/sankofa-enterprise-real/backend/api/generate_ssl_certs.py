#!/usr/bin/env python3
"""
Gerador de Certificados SSL para Sankofa Enterprise Pro
Cria certificados auto-assinados para desenvolvimento e testes HTTPS
"""

import os
import subprocess
from datetime import datetime, timedelta
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

def generate_ssl_certificates():
    """Gera certificados SSL auto-assinados para desenvolvimento"""
    
    print("🔐 Gerando certificados SSL para Sankofa Enterprise Pro...")
    
    # Gera chave privada
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    # Informações do certificado
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "BR"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "São Paulo"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "São Paulo"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Sankofa Enterprise Pro"),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Security Department"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    
    # Cria certificado
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.utcnow()
    ).not_valid_after(
        datetime.utcnow() + timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.DNSName("127.0.0.1"),
            x509.DNSName("sankofa.local"),
            x509.DNSName("api.sankofa.local"),
        ]),
        critical=False,
    ).add_extension(
        x509.BasicConstraints(ca=True, path_length=0),
        critical=True,
    ).add_extension(
        x509.KeyUsage(
            digital_signature=True,
            content_commitment=False,
            key_encipherment=True,
            data_encipherment=False,
            key_agreement=False,
            key_cert_sign=True,
            crl_sign=True,
            encipher_only=False,
            decipher_only=False,
        ),
        critical=True,
    ).sign(private_key, hashes.SHA256())
    
    # Salva chave privada
    with open("key.pem", "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    # Salva certificado
    with open("cert.pem", "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    print("✅ Certificados SSL gerados com sucesso!")
    print("   - Chave privada: key.pem")
    print("   - Certificado: cert.pem")
    print("   - Válido por: 365 dias")
    print("   - Domínios: localhost, 127.0.0.1, sankofa.local, api.sankofa.local")
    
    # Define permissões seguras
    os.chmod("key.pem", 0o600)
    os.chmod("cert.pem", 0o644)
    
    print("🔒 Permissões de segurança aplicadas aos certificados")

if __name__ == "__main__":
    generate_ssl_certificates()
