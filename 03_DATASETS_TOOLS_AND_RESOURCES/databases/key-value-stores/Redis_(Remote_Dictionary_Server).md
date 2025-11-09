# Redis (Remote Dictionary Server)

## Description

Servidor de estrutura de dados de código aberto, em memória, usado como cache, banco de dados e message broker. Sua principal proposta de valor é a **velocidade extrema** (sub-milissegundo) e o suporte a **estruturas de dados ricas** (Strings, Hashes, Lists, Sets, Sorted Sets, Bitmaps, HyperLogLogs, Streams, etc.), permitindo soluções mais complexas do que um simples cache de chave-valor.

## Statistics

**Latência:** Sub-milissegundo. **Desempenho:** Milhões de operações por segundo. **Arquitetura:** Modelo single-threaded otimizado. **Comparação:** Mais lento que o Memcached para operações de String simples devido à sobrecarga de estruturas de dados.

## Features

Suporte a tipos de dados complexos (Listas, Sets, Hashes, etc.); Persistência (RDB e AOF); Alta Disponibilidade (Replicação e Cluster); Transações atômicas; Funcionalidade de Message Broker (Pub/Sub); Extensível com Módulos.

## Use Cases

Caching de sessão e página, filas de mensagens, contadores e medidores em tempo real, tabelas de classificação (leaderboards), gerenciamento de sessão de usuário, análise em tempo real.

## Integration

**Python (redis-py):**\n```python\nimport redis\nr = redis.Redis(host='localhost', port=6379, db=0)\nr.set('user:100:session', '{"login_time": "2025-11-08T10:00:00"}')\nr.lpush('recent_logins', 'user:100', 'user:101', 'user:102')\n```

## URL

https://redis.io/