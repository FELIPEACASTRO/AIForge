# PostgreSQL (PostgresML/pgvector)

## Description

PostgreSQL é um sistema de gerenciamento de banco de dados relacional de código aberto (ORDBMS) conhecido por sua robustez, extensibilidade e conformidade com padrões. Seu valor único para Machine Learning reside na sua capacidade de trazer o ML para dentro do banco de dados, eliminando a necessidade de mover grandes volumes de dados. Extensões como **PostgresML** permitem treinamento e inferência de modelos diretamente via SQL, e **pgvector** o transforma em um banco de dados vetorial de alto desempenho, essencial para aplicações de IA generativa e busca semântica.

## Statistics

**Desempenho:** PostgresML pode ser 8-40x mais rápido que microsserviços HTTP tradicionais baseados em Python para inferência, devido à eliminação da latência de rede e serialização de dados. **Adoção:** Em 2025, o PostgreSQL detém cerca de 16.85% do mercado de bancos de dados relacionais, com uma adoção crescente em projetos de IA, impulsionada por extensões como o pgvector. **Otimização:** Versões recentes (e.g., PostgreSQL 18) mostram uma redução de 40% no uso de memória para junções em grandes conjuntos de dados.

## Features

**Extensibilidade (PostgresML/pgvector):** Treinamento e inferência de modelos de ML (regressão, classificação, LLMs) diretamente via SQL. Suporte nativo a vetores de alta dimensão para busca de similaridade. **Conformidade ACID:** Garante a integridade e confiabilidade dos dados. **Suporte a JSON/JSONB:** Flexibilidade para dados semi-estruturados. **Riqueza de Tipos de Dados:** Suporte a tipos complexos e definidos pelo usuário.

## Use Cases

**Busca Semântica e RAG:** Utilizando `pgvector` para armazenar embeddings e realizar buscas de similaridade para sistemas de Geração Aumentada de Recuperação (RAG). **Detecção de Fraudes:** Treinamento e inferência de modelos em tempo real sobre dados transacionais. **Recomendação de Produtos:** Construção de sistemas de recomendação baseados em vetores de usuários e itens. **Análise de Séries Temporais:** Uso de extensões como TimescaleDB (compatível com Postgres) para dados de sensores e IoT.

## Integration

A integração com Python é tipicamente feita usando a biblioteca `psycopg2` ou `psycopg`. Para usar o PostgresML, a interação é via SQL.

```python
# Exemplo de conexão básica com Python (psycopg2)
import psycopg2

try:
    conn = psycopg2.connect(
        host="localhost",
        database="ml_db",
        user="user",
        password="password"
    )
    print("Conexão com PostgreSQL bem-sucedida.")
    conn.close()
except Exception as e:
    print(f"Erro ao conectar: {e}")

# Exemplo de uso do PostgresML (via SQL)
# SELECT * FROM pgml.train('minha_tabela', 'target_col', 'regressor');
```

## URL

https://www.postgresql.org/ | https://postgresml.org/ | https://github.com/pgvector/pgvector