# Lichess/standard-chess-games

## Description
O dataset **Lichess/standard-chess-games** é uma coleção massiva e atualizada mensalmente de mais de **7,1 bilhões de linhas** (jogos) de xadrez padrão classificados (rated) jogados na plataforma Lichess.org. Os dados são fornecidos em formato Parquet particionado por ano e mês, o que facilita a análise e o processamento de grandes volumes de dados. A coleção abrange jogos desde 2013 e é continuamente atualizada, incluindo dados de 2025 (em andamento). É uma fonte primária para pesquisa e desenvolvimento em inteligência artificial e ciência de dados no domínio do xadrez.

## Statistics
**Tamanho:** 4.94 TB (em arquivos Parquet).
**Amostras:** Mais de **7.136.017.970** (7.1 bilhões) de jogos de xadrez.
**Versões:** Atualizado mensalmente, abrangendo dados de **2013 até 2025** (o ano de 2025 está em andamento). O foco em 2023-2025 é garantido pela atualização contínua.

## Features
O dataset inclui campos detalhados para cada jogo, como: Evento, URL do jogo (Site), nomes dos Jogadores (Brancas/Pretas), Resultado, Títulos dos jogadores, **Elo Glicko2** (Brancas/Pretas) e variação de Elo, Data/Hora UTC, Código ECO, Abertura, Tipo de Terminação e Controle de Tempo. O campo `movetext` contém os movimentos no formato PGN. Uma característica notável é que cerca de 6% dos jogos contêm **avaliações de análise do Stockfish** (vantagem em centipawns ou mate), essenciais para o treinamento de modelos de IA.

## Use Cases
- **Treinamento de Modelos de IA:** Ideal para treinar modelos de IA de xadrez, como aqueles para previsão de próximos movimentos (Next Move Prediction) e avaliação de posições.
- **Pesquisa em Ciência de Dados:** Utilizado para análise de grandes volumes de dados, como estudos de psicologia e comportamento de jogadores, e tendências de abertura.
- **Desenvolvimento de Ferramentas:** Base para a criação de Opening Explorers, motores de análise e outras ferramentas de auxílio ao xadrez.

## Integration
O dataset é facilmente acessível através da biblioteca **Hugging Face Datasets**.
1.  **Instalação:** `pip install datasets`
2.  **Uso em Python:**
    ```python
    from datasets import load_dataset
    # Carrega o dataset completo
    dataset = load_dataset("Lichess/standard-chess-games", split="train")
    # Para carregar um subconjunto específico (ex: jogos de 2024)
    # dataset_2024 = load_dataset("Lichess/standard-chess-games", split="train", year=2024)
    ```
    Os arquivos PGN originais e completos podem ser baixados diretamente dos dumps de banco de dados do Lichess (database.lichess.org).

## URL
[https://huggingface.co/datasets/Lichess/standard-chess-games](https://huggingface.co/datasets/Lichess/standard-chess-games)
