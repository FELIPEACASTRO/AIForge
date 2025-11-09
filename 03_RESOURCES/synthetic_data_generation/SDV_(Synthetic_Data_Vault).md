# SDV (Synthetic Data Vault)

## Description

O SDV é um ecossistema de código aberto (Python library) para a geração de dados sintéticos tabulares. Sua proposta de valor é ser uma solução completa ("one-stop shop") para a criação de dados sintéticos, permitindo que os usuários substituam dados reais por dados sintéticos para maior proteção de privacidade ou usem dados sintéticos como aprimoramento. Ele suporta diferentes modalidades de dados: tabela única, relacional (múltiplas tabelas) e sequencial (séries temporais).

## Statistics

Dominância de Mercado (Downloads): 84,49% dos downloads de bibliotecas de código aberto. Total de Downloads (Ecosistema): sdv (2.71M), rdt (6.91M), ctgan (1.30M).

## Features

Modelagem de Dados para Tabela Única, Múltiplas Tabelas e Séries Temporais. Ecosistema Modular com bibliotecas como Copulas (estatística), CTGAN (Deep Learning), DeepEcho (séries temporais) e SDMetrics (avaliação de qualidade).

## Use Cases

Expansão de Acesso a dados com segurança. Teste de novos produtos. Detecção de lavagem de dinheiro (setor bancário). Melhoria na detecção de fraude em seguros residenciais (MAPFRE).

## Integration

A integração é feita via Python SDK. Exemplo de uso com GaussianCopula:\n```python\nfrom sdv.datasets.demo import download_demo\nfrom sdv.single_table import GaussianCopulaSynthesizer\n\nreal_data, metadata = download_demo('single_table', 'fake_hotel_guests')\nsynthesizer = GaussianCopulaSynthesizer(metadata)\nsynthesizer.fit(real_data)\nsynthetic_data = synthesizer.sample(num_rows=10)\n```

## URL

https://sdv.dev/