# Apache Hadoop Distributed File System (HDFS)

## Description

HDFS é um sistema de arquivos distribuído baseado em Java, projetado para armazenar grandes volumes de dados de forma escalável e confiável em clusters de hardware commodity. Sua proposta de valor única reside em ser a camada de gerenciamento de dados do ecossistema Apache Hadoop, otimizado para processamento de grandes dados (Big Data) com alta taxa de transferência (throughput) e tolerância a falhas. Ele adota o princípio de 'mover a computação para os dados' para minimizar o tráfego de rede [1].

## Statistics

Capacidade de armazenamento em petabytes; Taxa de transferência de dados (throughput) medida em MB/s por nó; Tempo médio de recuperação de falhas (MTTR); Número de nós no cluster (escalabilidade horizontal); Fator de replicação padrão (geralmente 3) [2].

## Features

Arquitetura Master/Slave (NameNode e DataNodes); Tolerância a falhas e alta disponibilidade (HA) via replicação de dados e Standby NameNode; Consciência de Rack (Rack awareness) para otimização de E/S de rede; Minimal data motion (mover a computação para os dados); Grande escalabilidade horizontal [1].

## Use Cases

Armazenamento de Big Data para processamento em lote (batch processing); Plataformas de Data Lake; Análise de dados em larga escala com MapReduce e Spark; Hospedagem de dados para sistemas como Apache Hive e HBase [3].

## Integration

A integração é tipicamente feita através de APIs Java, comandos de linha de comando (CLI) ou bibliotecas cliente para outras linguagens. Exemplo de acesso via CLI:\n\n```bash\nhadoop fs -mkdir /user/data\nhadoop fs -put localfile.txt /user/data/remotefile.txt\nhadoop fs -cat /user/data/remotefile.txt\n```\n\nIntegração com ecossistema Hadoop (YARN, MapReduce) e ferramentas de BI como Tableau [4].

## URL

https://hadoop.apache.org/