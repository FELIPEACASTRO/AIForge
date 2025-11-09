# Dota 2 Pro League Matches 2016-2025

## Description
Este é um dataset abrangente de partidas da Liga Profissional de Dota 2, cobrindo o período de 2016 a 2025. O conjunto de dados é atualizado semanalmente e é derivado da API do OpenDota. Ele é ideal para pesquisadores e entusiastas de ciência de dados interessados em análise de esportes eletrônicos (esports), aprendizado de máquina para previsão de resultados de partidas e análise de comportamento de jogadores e equipes profissionais. O dataset é composto por 12 arquivos CSV principais por mês/ano, além de pastas de constantes e imagens.

## Statistics
*   **Tamanho Total:** 43.75 GB
*   **Número de Arquivos:** 1567
*   **Número de Colunas:** 5237 (em todos os arquivos)
*   **Cobertura:** Partidas da Liga Profissional de Dota 2 de 2016 a novembro de 2025.
*   **Frequência de Atualização:** Semanal.

## Features
O dataset é composto por 12 arquivos CSV principais que fornecem dados detalhados sobre as partidas:
*   **main\_metadata.csv**: Metadados principais da partida.
*   **players.csv**: Detalhes sobre os heróis e ações dos jogadores.
*   **picks\_bans.csv**: Informações sobre a fase de seleção e banimento de heróis.
*   **objectives.csv**: Marcos e objetivos alcançados na partida (e.g., FIRSTBLOOD, BARRACKS down).
*   **teamfights.csv**: Detalhes sobre as lutas de equipe.
*   **radiant\_gold\_adv.csv** e **radiant\_exp\_adv.csv**: Vantagem de ouro e experiência por minuto para a equipe Radiant.
*   **chat.csv** e **all\_word\_counts.csv**: Dados de chat e contagem de tokens.
*   **cosmetics.csv**, **draft\_timings.csv** e **teams.csv**: Informações adicionais sobre cosméticos, tempos de draft e equipes.
Inclui também pastas de `Constants` (IDs e estatísticas de habilidades, heróis e itens) e `Images`.

## Use Cases
*   **Previsão de Resultados de Partidas:** Utilização de dados de draft, vantagens de ouro/experiência e estatísticas de jogadores para prever o vencedor de partidas.
*   **Análise de Estratégia de Esports:** Estudo de padrões de seleção e banimento de heróis (picks/bans), análise de lutas de equipe e impacto de objetivos no resultado.
*   **Modelagem de Comportamento de Jogadores:** Análise de desempenho individual e de equipe ao longo do tempo.
*   **Desenvolvimento de Agentes de IA:** Uso dos dados como base para treinar modelos de aprendizado de máquina e reforço para jogar Dota 2.

## Integration
O dataset está hospedado no Kaggle e pode ser baixado diretamente através da interface web do Kaggle. Para usuários de Python, a integração pode ser feita usando a API do Kaggle, que permite o download programático do conjunto de dados.

**Passos para download via API do Kaggle:**
1.  Instale a biblioteca Kaggle: `pip install kaggle`
2.  Configure suas credenciais da API do Kaggle.
3.  Baixe o dataset usando o comando: `kaggle datasets download -d bwandowando/dota-2-pro-league-matches-2023`
4.  Descompacte o arquivo baixado para acessar os dados CSV.

A fonte primária dos dados é a API do OpenDota. A documentação detalhada dos campos pode ser encontrada na documentação do OpenDota.

## URL
[https://www.kaggle.com/datasets/bwandowando/dota-2-pro-league-matches-2023](https://www.kaggle.com/datasets/bwandowando/dota-2-pro-league-matches-2023)
