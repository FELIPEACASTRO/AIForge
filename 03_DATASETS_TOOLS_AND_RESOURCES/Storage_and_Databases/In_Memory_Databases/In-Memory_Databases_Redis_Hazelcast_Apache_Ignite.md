# In-Memory Databases: Redis, Hazelcast, Apache Ignite

## Description

**Redis** é o banco de dados em memória mais rápido do mundo, atuando como um armazenamento de estrutura de dados NoSQL de código aberto. Sua proposta de valor única reside na sua velocidade ultrabaixa latência (sub-milissegundo) e na versatilidade de suas estruturas de dados (Strings, Hashes, Lists, Sets, Sorted Sets, etc.), tornando-o ideal para caching, filas de mensagens e sessões de usuário. **Hazelcast** é uma Grade de Dados em Memória (IMDG) e Plataforma de Computação em Memória, focada em fornecer escalabilidade elástica e alta disponibilidade para aplicações distribuídas. Sua proposta de valor é transformar a memória RAM de um cluster de servidores em um único pool de dados distribuído, permitindo processamento paralelo e transações distribuídas. **Apache Ignite** é uma Plataforma de Computação em Memória (IMCP) e um Banco de Dados Distribuído, que se destaca por oferecer recursos de HTAP (Processamento Transacional e Analítico Híbrido) e um armazenamento de dados em múltiplas camadas (memória e disco). Sua proposta de valor é atuar como uma camada de aceleração de dados para sistemas existentes (cache) ou como um banco de dados distribuído completo com suporte a SQL, transações ACID e persistência nativa.

## Statistics

**Redis:** Latência de sub-milissegundo (muitas vezes sub-microssegundo) para operações de leitura/escrita. Alta taxa de transferência (throughput), atingindo centenas de milhares de operações por segundo em hardware moderno. **Hazelcast:** Latência de leitura/escrita em microssegundos. Projetado para escalabilidade elástica, suportando clusters com terabytes de dados em memória e alta taxa de transferência em cenários multi-threaded. **Apache Ignite:** Latência de leitura/escrita em microssegundos. Oferece desempenho de HTAP (Processamento Transacional e Analítico Híbrido) em tempo real, com a capacidade de escalar para petabytes de dados usando a arquitetura de múltiplas camadas (memória e disco).

## Features

**Redis:** Estruturas de dados ricas e atômicas (Listas, Sets, Hashes, Streams, etc.), persistência opcional (RDB/AOF), replicação assíncrona, transações, Pub/Sub, e módulos para funcionalidades avançadas como pesquisa de vetores (Vector Search) e JSON. **Hazelcast:** Estruturas de dados distribuídas (Mapas, Filas, Tópicos, Semáforos), transações distribuídas, processamento de fluxo (Stream Processing) com o Hazelcast Jet, alta disponibilidade e tolerância a falhas, e suporte a clientes em diversas linguagens. **Apache Ignite:** Banco de dados distribuído com suporte a SQL (ANSI-99), transações ACID, persistência nativa em disco (opcional), Data Grid, Compute Grid (processamento distribuído), Service Grid, e suporte a Machine Learning distribuído.

## Use Cases

**Redis:** Caching de sessão (session caching), filas de mensagens (message queues), contadores e rate limiting, placares de líderes (leaderboards) em tempo real, e armazenamento de dados geoespaciais. **Hazelcast:** Caching de segundo nível para Hibernate, processamento de transações de alto volume (ex: pagamentos), gerenciamento de estado para microsserviços e processamento de eventos em tempo real. **Apache Ignite:** Camada de aceleração de dados (Data Acceleration Layer) para bancos de dados legados, plataforma de HTAP para análise e transações em tempo real, e armazenamento de dados para aplicações de Machine Learning distribuído.

## Integration

**Redis (Python - `redis-py`):**
```python
import redis

# Conexão com o servidor Redis
r = redis.Redis(decode_responses=True)

# SET (Definir um valor)
r.set('user:100:name', 'Alice')

# GET (Obter um valor)
name = r.get('user:100:name')
print(f"O nome do usuário 100 é {name}")

# INCR (Incrementar um contador)
r.incr('page_views')
```

**Hazelcast (Java - `Hazelcast Client`):**
```java
/*
import com.hazelcast.client.HazelcastClient;
import com.hazelcast.client.config.ClientConfig;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.map.IMap;

public class HazelcastIntegration {
    public static void main(String[] args) {
        ClientConfig clientConfig = new ClientConfig();
        clientConfig.setClusterName("dev");
        clientConfig.getNetworkConfig().addAddress("127.0.0.1:5701");

        HazelcastInstance client = HazelcastClient.newHazelcastClient(clientConfig);
        IMap<String, String> cities = client.getMap("cities");

        // PUT (Inserir um valor)
        cities.put("BR", "Brasília");

        // GET (Obter um valor)
        String capitalBR = cities.get("BR");
        System.out.println("A capital do Brasil é " + capitalBR);

        client.shutdown();
    }
}
*/
```

**Apache Ignite (Python - `pyignite Thin Client`):**
```python
from pyignite import Client

# Conexão com o servidor Ignite
client = Client()
client.connect('127.0.0.1', 10800)

# Obter ou criar um Cache
cache = client.get_or_create_cache('myCache')

# PUT (Inserir um valor)
cache.put(1, 'Ignite Value 1')

# GET (Obter um valor)
value1 = cache.get(1)
print(f"O valor da chave 1 é {value1}")

client.close()
```

## URL

Redis: https://redis.io/ | Hazelcast: https://hazelcast.com/ | Apache Ignite: https://ignite.apache.org/