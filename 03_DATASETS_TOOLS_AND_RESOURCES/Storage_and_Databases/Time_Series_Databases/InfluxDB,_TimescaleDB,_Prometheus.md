# InfluxDB, TimescaleDB, Prometheus

## Description

**InfluxDB** é uma plataforma de dados de séries temporais de alto desempenho, otimizada para ingestão e consulta de dados de alta velocidade e alto volume, como IoT, DevOps e análise em tempo real. Sua proposta de valor única reside na sua arquitetura nativa de séries temporais (TSDB), que oferece compressão de dados superior e um ecossistema completo (Telegraf, Flux, Chronograf, Kapacitor - TICK Stack) para processamento e visualização de dados. É um sistema push-based, ideal para "firehoses" de dados. **TimescaleDB** é uma extensão de código aberto para PostgreSQL que o transforma em um banco de dados de séries temporais escalável. Sua principal proposta de valor é combinar a robustez, a confiabilidade e o ecossistema do PostgreSQL (SQL completo, transações ACID, ferramentas de BI) com o desempenho e a escalabilidade de um TSDB, utilizando "hypertables" para particionamento automático baseado em tempo. É a escolha ideal para quem precisa de dados relacionais e de séries temporais em um único banco de dados. **Prometheus** é um sistema de monitoramento e alerta de código aberto, projetado para o mundo nativo da nuvem. Sua proposta de valor única é ser um sistema pull-based, onde o servidor "raspa" (scrapes) métricas de endpoints HTTP configurados. É o padrão de fato para monitoramento de clusters Kubernetes e infraestruturas de microsserviços, focando em métricas operacionais em tempo real e alertas precisos.

## Statistics

**InfluxDB:** Mais de 1 bilhão de downloads via Docker; Mais de 1 milhão de instâncias open source ativas; Mais de 5 bilhões de downloads do Telegraf; Classificado como #1 TSDB pelo DB Engines (frequentemente). **TimescaleDB:** Mais de 20.4K estrelas no GitHub (para o projeto TimescaleDB); Classificação 4.7/5 no G2; Baseado no PostgreSQL, o banco de dados de código aberto mais popular. **Prometheus:** Projeto graduado da CNCF (após Kubernetes); Mais de 61.1K estrelas no GitHub; Padrão de fato para monitoramento em ambientes Kubernetes. **Comparação de Performance:** Benchmarks independentes frequentemente mostram o TimescaleDB superando o InfluxDB em consultas de agregação complexas (até 168% de melhor desempenho em um benchmark) devido à otimização do SQL, enquanto o InfluxDB se destaca na ingestão pura de alta taxa.

## Features

**InfluxDB:** Arquitetura nativa de séries temporais; Linguagem de consulta Flux (e InfluxQL); Suporte a alta cardinalidade; Compressão de dados Parquet; Plataforma completa (Ingestão, Armazenamento, Consulta, Visualização). **TimescaleDB:** Compatibilidade total com SQL e PostgreSQL; Hypertables (particionamento automático); Compressão de dados e downsampling nativos; Suporte a transações ACID; Consultas complexas que unem dados relacionais e de séries temporais. **Prometheus:** Modelo de dados dimensional (métricas e rótulos); Linguagem de consulta PromQL; Modelo pull-based (scraping); Descoberta de serviços (Kubernetes, Consul); Sistema de alerta (Alertmanager) desacoplado.

## Use Cases

**InfluxDB:** Internet das Coisas (IoT) com alta taxa de ingestão de sensores; Análise de desempenho esportivo em tempo real; Monitoramento de energia e redes inteligentes; Aplicações de Finanças (dados de mercado). **TimescaleDB:** Mercados Financeiros (negociações e dados de tick) que exigem transações ACID e consultas SQL complexas; IoT Industrial (IIoT) que precisa unir dados de séries temporais com metadados relacionais; Análise de Web e Aplicações (DAU por região, tipo de dispositivo). **Prometheus:** Monitoramento de infraestrutura e microsserviços (Kubernetes, Docker); Alerta e observabilidade em tempo real; SRE (Site Reliability Engineering) e DevOps; Coleta de métricas de aplicações (instrumentação).

## Integration

**InfluxDB (Python):** Utiliza o cliente oficial `influxdb-client` para escrever pontos de dados (com tags e campos) e consultar usando a linguagem Flux. É a forma padrão de integração para ingestão de alta velocidade. **TimescaleDB (Python/SQL):** Integra-se como qualquer PostgreSQL, usando bibliotecas como `psycopg2`. O código de exemplo mostra a criação de uma `hypertable` e uma consulta SQL otimizada com `time_bucket` para agregação de séries temporais. **Prometheus (Python Client):** A integração primária é expor métricas em um formato que o Prometheus possa raspar. O exemplo usa a biblioteca `prometheus_client` para definir e incrementar métricas (Contadores e Medidores) e iniciar um servidor HTTP para expô-las no endpoint `/metrics`.

## URL

InfluxDB: https://www.influxdata.com/ | TimescaleDB: https://www.timescale.com/ | Prometheus: https://prometheus.io/