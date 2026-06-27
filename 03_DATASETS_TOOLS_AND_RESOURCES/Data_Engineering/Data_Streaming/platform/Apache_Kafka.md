# Apache Kafka

## Description

Plataforma de streaming de eventos distribuída, de código aberto, usada por milhares de empresas para pipelines de dados de alto desempenho, análises de streaming e integração de dados. Sua proposta de valor única reside na sua capacidade de fornecer um backbone de dados de baixa latência, altamente escalável e tolerante a falhas, atuando como um "sistema operacional" para dados em tempo real.

## Statistics

Utilizado por mais de 80% das empresas da Fortune 100. Métricas chave incluem: Taxa de Transferência (mensagens/segundo), Latência (tempo de ponta a ponta), Atraso do Consumidor (Consumer Lag), e Utilização de Recursos do Broker. O particionamento de tópicos é a unidade fundamental de escalabilidade.

## Features

Alta taxa de transferência e baixa latência; Durabilidade e tolerância a falhas (replicação de dados); Escalabilidade horizontal (clusters de brokers e particionamento de tópicos); Modelo de mensagens de publicação/assinatura; Processamento de fluxo (Kafka Streams e KSQL); Conectores (Kafka Connect) para integração com sistemas externos.

## Use Cases

Construção de pipelines de dados em tempo real; Rastreamento de atividades do site (Web Tracking); Monitoramento de métricas e logs de aplicações; Event Sourcing e CQRS; Processamento de fluxo (Stream Processing) para detecção de fraudes ou análise em tempo real.

## Integration

A integração é tipicamente feita através de clientes em diversas linguagens (Java, Python, Node.js, Go) ou via Kafka Connect.
**Exemplo de Produtor em Python (usando `confluent-kafka`):**
```python
from confluent_kafka import Producer

conf = {'bootstrap.servers': 'localhost:9092'}
producer = Producer(conf)

def delivery_report(err, msg):
    if err is not None:
        print(f'Falha na entrega da mensagem: {err}')
    else:
        print(f'Mensagem entregue ao tópico {msg.topic()} [{msg.partition()}]')

producer.produce('meu_topico', key='chave', value='minha mensagem', callback=delivery_report)
producer.flush()
```

## URL

https://kafka.apache.org/