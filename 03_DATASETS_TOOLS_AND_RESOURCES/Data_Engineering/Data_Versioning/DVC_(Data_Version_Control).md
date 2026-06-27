# DVC (Data Version Control)

## Description

O DVC (Data Version Control) é um sistema de controle de versão de código aberto projetado especificamente para projetos de Data Science e Machine Learning. Ele estende o Git, permitindo que os usuários versionem grandes arquivos de dados e modelos, armazenando-os em um cache fora do repositório Git (em armazenamentos remotos como S3, Azure, GCS, etc.) e mantendo apenas metadados leves no Git. Isso garante a reprodutibilidade completa de experimentos de ML, rastreando o código, os dados e os pipelines de processamento. Seu principal diferencial é a experiência de usuário semelhante ao Git, focada em simplicidade e integração com fluxos de trabalho existentes.

## Statistics

- **GitHub Stars**: ~15.1k
- **Forks**: ~1.3k
- **Linguagem Principal**: Python (100.0% do repositório principal)
- **Adoção**: Amplamente adotado na comunidade MLOps por sua simplicidade e integração com o Git.

## Features

- **Versionamento de Dados e Modelos**: Rastreia grandes arquivos de dados e modelos sem armazená-los no Git.
- **Pipelines de ML**: Define e executa pipelines de dados e ML como um `Makefile` para reprodutibilidade.
- **Rastreamento de Experimentos**: Gerencia e compara experimentos de ML (código, dados, parâmetros, métricas) localmente, sem a necessidade de um servidor.
- **Armazenamento Flexível**: Suporta S3, Azure, Google Cloud Storage, SSH e outros como armazenamento remoto.
- **Extensão VS Code**: Interface gráfica para rastreamento de experimentos e gerenciamento de dados.

## Use Cases

- **Reprodutibilidade de ML**: Garante que qualquer experimento de ML possa ser recriado com o código e a versão exata dos dados e modelos.
- **Versionamento de Modelos**: Rastreia modelos de ML como artefatos de dados, permitindo a transição fácil entre versões de modelos.
- **Colaboração em Data Science**: Facilita o compartilhamento de grandes conjuntos de dados e modelos entre membros da equipe usando armazenamento remoto.
- **Auditoria e Conformidade**: Fornece um registro imutável de qual conjunto de dados foi usado para treinar um modelo específico.

## Integration

O DVC é uma ferramenta de linha de comando com uma API Python. A integração é feita através de comandos CLI que se assemelham ao Git.

**Exemplo de Integração (CLI e Python):**

1.  **Inicialização e Adição de Dados:**
    ```bash
    # Inicializa o DVC no repositório Git
    dvc init
    # Adiciona um diretório de dados para rastreamento
    dvc add data/raw_data
    # Confirma o metadado no Git
    git add data/raw_data.dvc
    git commit -m "Adiciona dados brutos"
    ```

2.  **Definição de Pipeline (dvc.yaml):**
    ```yaml
    # Define um estágio de pipeline para processamento de dados
    stages:
      featurize:
        cmd: python featurize.py
        deps:
          - data/raw_data
        outs:
          - data/features
    ```

3.  **API Python para Acesso Programático:**
    ```python
    from dvc.api import DVCFileSystem

    # Acessa um arquivo de dados versionado em um commit específico
    fs = DVCFileSystem(repo='.', rev='HEAD')
    with fs.open('data/processed.csv', mode='r') as f:
        content = f.read()
    ```

## URL

https://dvc.org/