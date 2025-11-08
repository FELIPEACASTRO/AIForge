# 🚀 Guia de Instalação - Sankofa Enterprise Pro

**Versão**: 3.0 Final  
**Ambiente**: Produção Bancária  
**Última Atualização**: 21 de Setembro de 2025  

---

## 📋 Pré-requisitos

### Hardware Mínimo (Produção)
- **CPU**: 8 cores (Intel Xeon ou AMD EPYC)
- **RAM**: 32 GB
- **Storage**: 500 GB SSD NVMe
- **Rede**: 10 Gbps

### Hardware Recomendado (Alta Disponibilidade)
- **CPU**: 16 cores (Intel Xeon Gold ou AMD EPYC)
- **RAM**: 64 GB
- **Storage**: 1 TB SSD NVMe (RAID 1)
- **Rede**: 25 Gbps com redundância

### Software
- **OS**: Ubuntu 22.04 LTS ou RHEL 9
- **Docker**: 24.0+
- **Docker Compose**: 2.20+
- **Redis**: 7.0+
- **PostgreSQL**: 15+
- **Node.js**: 18+ (para build do frontend)
- **Python**: 3.11+

---

## 🔧 Instalação Rápida (Docker)

### 1. Preparação do Ambiente

```bash
# Atualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Instalar Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Reiniciar sessão para aplicar grupos
newgrp docker
```

### 2. Download e Configuração

```bash
# Extrair o pacote
unzip SANKOFA_ENTERPRISE_PRO_PRODUCTION_READY.zip
cd sankofa-enterprise-real

# Configurar permissões
chmod +x deploy.sh
chmod +x scripts/*.sh

# Configurar variáveis de ambiente
cp .env.example .env
```

### 3. Configuração de Segurança

```bash
# Gerar chave JWT segura
export SANKOFA_JWT_SECRET=$(openssl rand -hex 32)
echo "SANKOFA_JWT_SECRET=$SANKOFA_JWT_SECRET" >> .env

# Gerar senha do admin
export ADMIN_PASSWORD=$(openssl rand -base64 32)
echo "SANKOFA_ADMIN_PASSWORD=$ADMIN_PASSWORD" >> .env

# Configurar Redis
export REDIS_PASSWORD=$(openssl rand -base64 32)
echo "REDIS_PASSWORD=$REDIS_PASSWORD" >> .env

# Gerar certificados SSL
./scripts/generate_ssl_certs.sh
```

### 4. Inicialização

```bash
# Iniciar todos os serviços
docker-compose up -d

# Verificar status
docker-compose ps

# Verificar logs
docker-compose logs -f
```

### 5. Verificação da Instalação

```bash
# Testar backend
curl -k https://localhost:8445/health

# Testar frontend
curl http://localhost:5174

# Testar Redis
redis-cli -h localhost -p 6379 ping

# Testar análise de fraude
curl -X POST https://localhost:8445/api/fraud/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "id": "TEST_001",
    "valor": 1000.00,
    "tipo": "PIX",
    "canal": "MOBILE",
    "cpf": "123.456.789-00"
  }'
```

---

## 🏗️ Instalação Manual (Sem Docker)

### 1. Instalação do Backend

```bash
# Instalar Python e dependências
sudo apt install python3.11 python3.11-venv python3.11-dev
python3.11 -m venv venv
source venv/bin/activate

# Instalar dependências Python
cd backend
pip install -r requirements.txt

# Configurar variáveis
export FLASK_APP=api/main_integrated_api.py
export FLASK_ENV=production
export SANKOFA_JWT_SECRET=$(openssl rand -hex 32)

# Iniciar backend
python api/main_integrated_api.py
```

### 2. Instalação do Frontend

```bash
# Instalar Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Instalar dependências
cd frontend
npm install --legacy-peer-deps

# Build para produção
npm run build

# Servir com nginx ou servidor estático
npm run preview
```

### 3. Configuração do Redis

```bash
# Instalar Redis
sudo apt install redis-server

# Configurar Redis
sudo nano /etc/redis/redis.conf
# Adicionar: requirepass <sua-senha-redis>

# Reiniciar Redis
sudo systemctl restart redis-server
sudo systemctl enable redis-server
```

### 4. Configuração do PostgreSQL

```bash
# Instalar PostgreSQL
sudo apt install postgresql postgresql-contrib

# Criar banco de dados
sudo -u postgres createdb sankofa
sudo -u postgres createuser sankofa
sudo -u postgres psql -c "ALTER USER sankofa PASSWORD 'senha-forte';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE sankofa TO sankofa;"
```

---

## 🔒 Configuração de Segurança Avançada

### 1. Firewall (UFW)

```bash
# Configurar firewall
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Permitir apenas portas necessárias
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 5174/tcp  # Frontend (temporário)
sudo ufw allow 8445/tcp  # Backend API

# Verificar status
sudo ufw status verbose
```

### 2. SSL/TLS com Let's Encrypt

```bash
# Instalar Certbot
sudo apt install certbot python3-certbot-nginx

# Gerar certificados
sudo certbot --nginx -d seu-dominio.com

# Configurar renovação automática
sudo crontab -e
# Adicionar: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 3. Nginx como Proxy Reverso

```bash
# Instalar Nginx
sudo apt install nginx

# Configurar proxy reverso
sudo nano /etc/nginx/sites-available/sankofa
```

```nginx
server {
    listen 80;
    server_name seu-dominio.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name seu-dominio.com;

    ssl_certificate /etc/letsencrypt/live/seu-dominio.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/seu-dominio.com/privkey.pem;

    # Frontend
    location / {
        proxy_pass http://localhost:5174;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Backend API
    location /api/ {
        proxy_pass https://localhost:8445;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Ativar configuração
sudo ln -s /etc/nginx/sites-available/sankofa /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 📊 Configuração de Monitoramento

### 1. DataDog Agent

```bash
# Instalar DataDog Agent
DD_API_KEY=<sua-api-key> DD_SITE="datadoghq.com" bash -c "$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script.sh)"

# Configurar integração Python
sudo nano /etc/datadog-agent/conf.d/python.d/conf.yaml
```

```yaml
init_config:

instances:
  - name: sankofa-backend
    url: https://localhost:8445/metrics
    tags:
      - env:production
      - service:sankofa
```

### 2. Logs Centralizados

```bash
# Configurar rsyslog para DataDog
sudo nano /etc/rsyslog.d/22-datadog.conf
```

```
# Enviar logs para DataDog
*.* @@intake.logs.datadoghq.com:10516
```

### 3. Alertas Personalizados

Configure alertas no DataDog para:
- CPU > 80%
- Memória > 85%
- Latência > 50ms
- Taxa de erro > 1%
- Fraudes detectadas > 100/hora

---

## 🔄 Backup e Recuperação

### 1. Backup Automático

```bash
# Criar script de backup
sudo nano /usr/local/bin/backup_sankofa.sh
```

```bash
#!/bin/bash
BACKUP_DIR="/backup/sankofa"
DATE=$(date +%Y%m%d_%H%M%S)

# Criar diretório de backup
mkdir -p $BACKUP_DIR

# Backup do banco de dados
pg_dump -h localhost -U sankofa sankofa > $BACKUP_DIR/db_$DATE.sql

# Backup do Redis
redis-cli --rdb $BACKUP_DIR/redis_$DATE.rdb

# Backup dos arquivos de configuração
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /opt/sankofa/config

# Limpar backups antigos (manter 30 dias)
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
find $BACKUP_DIR -name "*.rdb" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

```bash
# Tornar executável
sudo chmod +x /usr/local/bin/backup_sankofa.sh

# Agendar backup a cada 6 horas
sudo crontab -e
# Adicionar: 0 */6 * * * /usr/local/bin/backup_sankofa.sh
```

### 2. Procedimento de Recuperação

```bash
# Parar serviços
docker-compose down

# Restaurar banco de dados
psql -h localhost -U sankofa sankofa < /backup/sankofa/db_YYYYMMDD_HHMMSS.sql

# Restaurar Redis
redis-cli --rdb /backup/sankofa/redis_YYYYMMDD_HHMMSS.rdb

# Restaurar configurações
tar -xzf /backup/sankofa/config_YYYYMMDD_HHMMSS.tar.gz -C /

# Reiniciar serviços
docker-compose up -d
```

---

## 🧪 Testes de Validação

### 1. Teste de Funcionalidade

```bash
# Executar suite de testes
cd tests
python -m pytest test_integration.py -v

# Teste de carga básico
python load_test.py --transactions=1000 --concurrent=10
```

### 2. Teste de Performance

```bash
# Teste de throughput
ab -n 10000 -c 100 https://localhost:8445/api/health

# Teste de análise de fraude
python tests/performance_test.py
```

### 3. Teste de Segurança

```bash
# Scan de vulnerabilidades
nmap -sS -O localhost

# Teste de penetração básico
nikto -h https://localhost:8445
```

---

## 🚨 Troubleshooting

### Problemas Comuns

**1. Backend não inicia**
```bash
# Verificar logs
docker-compose logs backend

# Verificar variáveis de ambiente
docker-compose exec backend env | grep SANKOFA

# Reiniciar serviço
docker-compose restart backend
```

**2. Frontend não carrega**
```bash
# Verificar build
docker-compose logs frontend

# Verificar proxy
curl -I http://localhost:5174

# Rebuild frontend
docker-compose build frontend --no-cache
```

**3. Redis não conecta**
```bash
# Verificar status
docker-compose exec redis redis-cli ping

# Verificar configuração
docker-compose exec redis redis-cli config get requirepass

# Reiniciar Redis
docker-compose restart redis
```

**4. Performance baixa**
```bash
# Verificar recursos
docker stats

# Verificar cache hit rate
redis-cli info stats | grep hit

# Verificar logs de performance
tail -f logs/performance.log
```

### Logs Importantes

- **Backend**: `/var/log/sankofa/backend.log`
- **Frontend**: `/var/log/sankofa/frontend.log`
- **Redis**: `/var/log/redis/redis-server.log`
- **Nginx**: `/var/log/nginx/access.log`

---

## 📞 Suporte

### Contatos de Emergência
- **Suporte 24/7**: +55 11 9999-9999
- **Email**: support@sankofa.ai
- **Slack**: #sankofa-support

### Documentação Adicional
- **API Docs**: https://docs.sankofa.ai/api
- **Admin Guide**: https://docs.sankofa.ai/admin
- **Troubleshooting**: https://docs.sankofa.ai/troubleshooting

---

## ✅ Checklist de Instalação

- [ ] Hardware atende aos requisitos mínimos
- [ ] Sistema operacional atualizado
- [ ] Docker e Docker Compose instalados
- [ ] Variáveis de ambiente configuradas
- [ ] Certificados SSL gerados
- [ ] Firewall configurado
- [ ] Backup automático configurado
- [ ] Monitoramento ativo
- [ ] Testes de validação executados
- [ ] Documentação revisada
- [ ] Equipe treinada

---

**🎉 Parabéns! O Sankofa Enterprise Pro está pronto para produção bancária!**

*Para suporte adicional, consulte a documentação completa ou entre em contato com nossa equipe técnica.*
