# Papers With Code Datasets - Research datasets with benchmarks (Archive)

## Description

O Papers With Code (PWC) foi uma plataforma centralizada e orientada pela comunidade que catalogava artigos de pesquisa em Machine Learning (ML), seus códigos de implementação, conjuntos de dados associados e tabelas de resultados de benchmarks. Sua proposta de valor única residia em conectar diretamente a pesquisa acadêmica (papers) com a implementação prática (code), permitindo que pesquisadores e praticantes encontrassem rapidamente o estado da arte (State-of-the-Art - SOTA) para tarefas específicas de ML. A plataforma foi descontinuada e seu conteúdo foi arquivado e parcialmente migrado para o Hugging Face, que agora é o principal ponto de acesso para o seu acervo de dados. O foco principal da plataforma eram os **Datasets de Pesquisa com Benchmarks**, que permitiam a comparação direta do desempenho de modelos em tarefas específicas.

## Statistics

- **Status Atual**: Descontinuado (Sunsetted) em agosto de 2025, com o conteúdo arquivado e migrado para o Hugging Face.
- **Volume de Dados (Histórico)**: Em seu auge, a plataforma indexava mais de **5.600** conjuntos de dados de Machine Learning [1], mais de **950** tarefas únicas de ML, mais de **500** tabelas de avaliação SOTA e mais de **8.500** artigos com código [2].
- **Acervo no Hugging Face**: O arquivo público (`pwc-archive`) contém o *snapshot* final dos dados, incluindo mais de **300.000** links entre artigos e código.
- **Adoção**: Foi uma das principais referências para pesquisa em ML, sendo citada em diversos artigos acadêmicos para coleta de dados e análise de tendências [3].

## Features

- **Conexão Paper-Code-Dataset**: Ligação direta entre artigos científicos, seus códigos e os datasets utilizados, facilitando a reprodutibilidade da pesquisa.
- **Tabelas de Benchmarks SOTA**: Organização de resultados de modelos em tabelas de classificação (Leaderboards) para diversas tarefas de ML, permitindo a identificação rápida do desempenho SOTA.
- **Taxonomia de Tarefas de ML**: Estrutura hierárquica de tarefas, sub-tarefas e áreas de ML (como Visão Computacional, Processamento de Linguagem Natural, etc.) para navegação e descoberta.
- **Dados Abertos (Data Dumps)**: O conjunto de dados completo do PWC foi disponibilizado publicamente no GitHub e arquivado no Hugging Face, permitindo o uso offline e a análise em larga escala.
- **API de Acesso (Descontinuada)**: Existia um cliente Python para a API de leitura/escrita, que agora é mantido pela comunidade para o acesso aos dados arquivados.

## Use Cases

- **Pesquisa Acadêmica e Reprodutibilidade**: Encontrar rapidamente o código e os dados associados a um artigo de pesquisa, permitindo a reprodução de resultados e a validação de metodologias.
- **Análise de Estado da Arte (SOTA)**: Identificar o modelo de melhor desempenho para uma tarefa específica de Machine Learning (e.g., segmentação de imagem, tradução automática) através das tabelas de benchmarks.
- **Descoberta de Datasets**: Explorar e filtrar conjuntos de dados de ML por modalidade (visão, texto, áudio) ou por tarefa, sendo um recurso valioso para iniciar novos projetos.
- **Desenvolvimento de Agentes de IA**: Utilizar o acervo de dados (via Hugging Face ou GitHub) para treinar e avaliar modelos de linguagem grandes (LLMs) e agentes de IA com conhecimento sobre a taxonomia de tarefas e resultados de pesquisa em ML.

## Integration

A integração com o acervo de dados do Papers With Code é feita primariamente através do seu arquivo público no GitHub ou, de forma mais estruturada, através do **Hugging Face Datasets Hub**, que hospeda o `pwc-archive`.

**1. Acesso ao Arquivo de Dados no Hugging Face (Recomendado)**

O Hugging Face hospeda o arquivo de dados do PWC, que pode ser acessado usando a biblioteca `datasets` do Python.

```python
from datasets import load_dataset

# Carrega o dataset de links entre artigos e código
dataset = load_dataset("pwc-archive/links-between-paper-and-code")

# Exemplo de acesso aos dados
print(dataset['train'][0])
# {'paper_id': '...', 'code_url': '...', 'repo_type': '...'}

# Outros datasets do arquivo podem ser explorados no Hugging Face Hub
# Ex: pwc-archive/all-papers-with-abstracts
```

**2. Acesso Direto ao Arquivo de Dados (GitHub)**

O repositório `paperswithcode-data` no GitHub contém o *data dump* completo, que pode ser clonado e processado localmente.

```bash
git clone https://github.com/paperswithcode/paperswithcode-data.git
```

Os dados estão em formato JSON e podem ser lidos com Python:

```python
import json

with open('paperswithcode-data/datasets.json', 'r') as f:
    datasets_data = json.load(f)

# Processar os dados conforme necessário
```

## URL

https://huggingface.co/pwc-archive