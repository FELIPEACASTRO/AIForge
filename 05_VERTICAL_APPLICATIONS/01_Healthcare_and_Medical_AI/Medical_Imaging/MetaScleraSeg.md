# MetaScleraSeg

## Description

O MetaScleraSeg é um framework de meta-learning projetado para a segmentação generalizada da esclera (a parte branca do olho). Ele aborda o desafio da falta de generalização em modelos de deep learning tradicionais para novos domínios de dados (como diferentes etnias, qualidades de imagem ou conjuntos de dados). O framework utiliza uma estratégia de meta-amostragem para simular variações de domínio e um modelo base UNet 3+ invariante a estilo, otimizado através de uma estratégia de otimização de dois níveis (bilevel optimization) para aprender conhecimento transferível entre domínios. Foi publicado em 2023.

## Statistics

- **Publicação:** Neural Computing and Applications (2023).
- **Citações:** 11 (em agosto de 2023, conforme arXiv).
- **Desempenho:** Demonstrou superioridade em comparação com modelos de linha de base (baselines) em protocolos de generalização de domínio (cross-dataset, cross-ethnicity, cross-quality).
- **Métrica Típica:** O F1-score (ou Dice Score) para segmentação de esclera em datasets como SBVPI geralmente atinge valores acima de 96% para métodos de ponta. O MetaScleraSeg supera estes baselines em cenários de domínio não visto.

## Features

- **Meta-Sampling:** Estratégia para simular variações de domínio (domain shifts) em cenários do mundo real.
- **Modelo Base Invariante a Estilo:** Utiliza uma arquitetura UNet 3+ modificada para garantir que o modelo se concentre em características essenciais, ignorando variações de estilo.
- **Otimização de Dois Níveis:** Emprega uma estratégia de meta-otimização para atualizar o modelo base, permitindo que ele generalize bem para domínios de destino não vistos.
- **Generalização Robusta:** Projetado para funcionar em protocolos de validação cruzada (cross-dataset, cross-ethnicity e cross-quality).

## Use Cases

- **Biometria Ocular:** Segmentação precisa da esclera para sistemas de reconhecimento de identidade.
- **Diagnóstico Ocular Generalizado:** Criação de modelos de IA que podem ser implantados em diferentes clínicas ou regiões geográficas, lidando com variações de equipamento e população (etnia).
- **Aprendizado Few-Shot:** Aplicação em cenários onde novos domínios de dados médicos têm poucas amostras rotuladas.

## Integration

O código oficial do projeto está disponível no GitHub, implementado em PyTorch.
**Exemplo de Execução (Teste):**
```bash
# Clonar o repositório
git clone https://github.com/lhqqq/MetaScleraSeg.git
cd MetaScleraSeg

# Instalar dependências (assumindo ambiente Python e PyTorch configurados)
# pip install -r requirements.txt (se houver)

# Executar o script de teste (requer o dataset e modelos pré-treinados)
python test.py
```
**Observação:** O dataset (CDSS) e os modelos pré-treinados são fornecidos via Baidu Drive, conforme o README do repositório.

## URL

https://github.com/lhqqq/MetaScleraSeg