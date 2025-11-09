# SC2EGSet: StarCraft II Esport Game State Dataset

## Description
O **SC2EGSet: StarCraft II Esport Game State Dataset** é um conjunto de dados abrangente e atualizado (Versão 2.0.1, Março de 2025) que fornece informações detalhadas sobre o estado do jogo e replays de torneios de e-sports do StarCraft II desde 2016. Este dataset é uma evolução do trabalho original da DeepMind e do SC2ReSet, sendo mantido por pesquisadores independentes. Ele foi projetado para facilitar a pesquisa em Inteligência Artificial (IA), Aprendizado de Máquina (ML), e estudos de interação humano-computador (HCI) e e-sports. Os dados são processados a partir de replays brutos, oferecendo uma visão rica e estruturada das decisões estratégicas e táticas dos jogadores profissionais.

## Statistics
**Versão:** 2.0.1 (Publicada em Março de 2025).
**Replays Processados:** Dados processados de 55 "replaypacks" de torneios.
**Arquivos Finais:** 17.895 arquivos de estado de jogo processados.
**Tamanho:** O arquivo de exemplo `2016_IEM_10_Taipei.zip` tem 12.6 GB. O dataset completo é significativamente maior, distribuído em múltiplos arquivos.
**Período:** Replays de torneios desde 2016.

## Features
Dados de estado de jogo (game-state) e replays de e-sports de alto nível. Inclui informações detalhadas como histograma de versão do jogo, datas das partidas, informações do servidor, raças escolhidas, duração da partida, unidades detectadas e histograma de raça vs. tempo de jogo. O dataset é compatível com as APIs PyTorch e PyTorch Lightning através da biblioteca `sc2_datasets`, facilitando o carregamento e a modelagem dos dados. Licenciado sob Creative Commons Attribution 4.0 International (CC BY 4.0).

## Use Cases
**Inteligência Artificial e Aprendizado por Reforço (RL):** Treinamento de agentes de IA para jogar StarCraft II, como o AlphaStar da DeepMind, e desenvolvimento de modelos de RL offline.
**Análise de E-sports:** Estudo de estratégias de jogadores profissionais, detecção de padrões de jogo e análise de desempenho.
**Pesquisa em HCI:** Investigação da tomada de decisão humana em ambientes complexos e em tempo real.
**Modelagem Preditiva:** Criação de modelos para prever resultados de partidas ou ações futuras dos jogadores.

## Integration
O dataset pode ser acessado e utilizado através da biblioteca Python `sc2_datasets`.
1.  **Instalação da Biblioteca:** `pip install sc2_datasets`
2.  **Uso com PyTorch/PyTorch Lightning:** A biblioteca fornece classes como `SC2EGSetDataset` (PyTorch) e `SC2EGSetDataModule` (PyTorch Lightning) que gerenciam o download, descompactação e acesso aos dados.
3.  **Download:** O download dos arquivos brutos (replays) pode ser feito diretamente pelo Zenodo. O arquivo `2016_IEM_10_Taipei.zip` tem 12.6 GB, e o dataset completo é composto por múltiplos arquivos, totalizando um volume significativo de dados. O acesso programático via API é o método recomendado.

## URL
[https://zenodo.org/records/15073637](https://zenodo.org/records/15073637)
