# Neo4j

## Description

Neo4j é o principal banco de dados de grafo nativo, otimizado para armazenar e percorrer relacionamentos complexos. Sua proposta de valor única reside na sua arquitetura nativa de grafo e na linguagem de consulta Cypher, que permite consultas intuitivas e de alto desempenho em dados altamente conectados. É a base da Plataforma de Grafos Neo4j, que inclui ferramentas para análise e visualização.

## Statistics

Líder de mercado em bancos de dados de grafo. Suporta centenas de milhões a bilhões de nós e relacionamentos. O desempenho é medido em milissegundos para consultas de profundidade de grafo, superando bancos de dados relacionais em 100x a 1000x para dados conectados. Amplamente adotado por empresas da Fortune 500.

## Features

Banco de dados de grafo nativo. Linguagem de consulta Cypher (declarativa, inspirada em SQL). Transações ACID. Alta disponibilidade e escalabilidade horizontal (clustering). Ferramentas de visualização e análise (Bloom, AuraDB). Suporte a índices e restrições.

## Use Cases

Detecção de fraude (identificação de padrões ocultos). Mecanismos de recomendação (conexões entre usuários e itens). Grafos de conhecimento (organização de dados complexos). Gerenciamento de identidade e acesso. Otimização da cadeia de suprimentos.

## Integration

Driver oficial em várias linguagens (Python, Java, JavaScript, .NET). Exemplo de integração Python:\n```python\nfrom neo4j import GraphDatabase\n\ndriver = GraphDatabase.driver(\"bolt://localhost:7687\", auth=(\"neo4j\", \"password\"))\n\ndef add_friend(tx, name, friend_name):\n    tx.run(\"MERGE (a:Person {name: $name}) \"\n           \"MERGE (b:Person {name: $friend_name}) \"\n           \"MERGE (a)-[:KNOWS]->(b)\", name=name, friend_name=friend_name)\n\nwith driver.session() as session:\n    session.execute_write(add_friend, \"Alice\", \"Bob\")\n```

## URL

https://neo4j.com/