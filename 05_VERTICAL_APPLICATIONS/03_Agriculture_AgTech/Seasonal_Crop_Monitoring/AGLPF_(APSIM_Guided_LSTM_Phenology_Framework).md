# AGLPF (APSIM Guided LSTM Phenology Framework)

## Description

O AGLPF (APSIM Guided LSTM Phenology Framework) é uma estrutura de aprendizado contínuo baseada em aprendizado profundo guiado por física, desenvolvida para simular dinamicamente as mudanças na fenologia do milho. Ele combina a interpretabilidade de modelos baseados em processos (PBMs), como o APSIM, com a capacidade de extração de padrões de modelos de inteligência artificial (AIMs), especificamente uma rede LSTM (Long Short-Term Memory) com mecanismo de atenção. O framework é projetado para superar as limitações de ambos os tipos de modelos, permitindo uma simulação contínua e aprimoramento progressivo com novos dados. Inicialmente treinado com dados de saída do PBM, o AGLPF é capaz de se autoajustar com dados reais de fenologia incremental, melhorando continuamente sua precisão ao longo do tempo.

## Statistics

- **Precisão Inicial (Treinamento com APSIM):** Erro Quadrático Médio da Raiz (RMSE) médio de 0,8 dias para a fase vegetativa e de floração, 1,4 dias para a fase de enchimento de grãos e 2,0 dias para o ciclo de crescimento completo.
- **Melhoria com Aprendizado Contínuo:** O RMSE do ciclo de crescimento completo diminuiu de 27,8 dias para **5,5 dias** após o autoajuste com dados reais de fenologia incremental (0 a 12 anos).
- **Vantagem do Autoajuste:** O método de autoajuste superou o método de treinamento do zero em todas as fases fenológicas.
- **Citações:** Citado por 2 (em 2025, conforme o snippet inicial).
- **Publicação:** Agricultural and Forest Meteorology, Volume 373, 1 de Junho de 2025, 110562.

## Features

- **Aprendizado Contínuo (Continual Learning):** Capacidade de se autoajustar e melhorar o desempenho com a incorporação incremental de novos dados de fenologia real, sem esquecer o conhecimento prévio.
- **Aprendizado Profundo Guiado por Física (Physics-Guided Deep Learning):** Integração de conhecimento de modelos baseados em processos (APSIM) para garantir simulações interpretáveis e temporalmente contínuas.
- **Modelo Híbrido (Hybrid Model):** Combinação de PBM (APSIM) e AIM (LSTM com Atenção) para sinergia de interpretabilidade e poder preditivo.
- **Simulação Dinâmica de Fenologia:** Capacidade de simular dinamicamente as fases de crescimento da cultura (vegetativa, floração, enchimento de grãos).
- **Fácil Atualização e Interpretabilidade:** Estrutura projetada para ser facilmente atualizada com novos insights e fornecer saídas interpretáveis.

## Use Cases

- **Simulação Dinâmica de Fenologia de Culturas:** Aplicação primária na simulação das mudanças nas fases de crescimento do milho (maize) em grandes regiões, como o Cinturão de Milho Chinês.
- **Monitoramento Agrícola em Tempo Real:** Potencial para uso em sistemas de monitoramento em tempo real, onde a precisão do modelo pode ser continuamente aprimorada com a chegada de novos dados de campo.
- **Previsão de Colheita e Gestão de Riscos:** Melhoria na precisão da previsão de datas de colheita e avaliação de riscos relacionados ao clima e manejo, devido à maior precisão e interpretabilidade do modelo.
- **Adaptação a Mudanças Ambientais:** A capacidade de aprendizado contínuo permite que o modelo se adapte a variações sazonais e mudanças climáticas ao longo do tempo.

## Integration

O artigo descreve o AGLPF como uma estrutura conceitual e metodológica. A implementação envolve:
1. **PBM (APSIM):** Geração de um conjunto de dados de fenologia inicial.
2. **AIM (LSTM com Atenção):** Treinamento inicial do modelo de Aprendizado Profundo com o conjunto de dados do APSIM.
3. **Autoajuste (Self-Tuning):** Utilização de dados reais de fenologia incremental para refinar o modelo via aprendizado contínuo.

Embora o código fonte específico não esteja disponível no resumo, a integração requer a utilização de bibliotecas de aprendizado profundo (como PyTorch ou TensorFlow) para implementar a rede LSTM com atenção e a integração com um modelo de simulação de culturas (como o APSIM) para a geração de dados e a orientação física. O conceito de "Physics-Guided Deep Learning" sugere a inclusão de restrições ou variáveis do modelo físico na função de perda do modelo de aprendizado profundo.

## URL

https://www.sciencedirect.com/science/article/abs/pii/S0168192325001820