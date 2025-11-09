# DQN Replay Dataset (Atari 2600 Games RL)

## Description
O **DQN Replay Dataset** é um conjunto de dados de Aprendizagem por Reforço (RL) *offline* baseado em jogos do Atari 2600. Ele foi coletado treinando um agente Deep Q-Network (DQN) em 60 jogos diferentes do Atari 2600, com ações "pegajosas" (sticky actions) ativadas, por 200 milhões de *frames* por jogo. O dataset armazena todas as tuplas de experiência (*observação, ação, recompensa, próxima observação*) encontradas durante o treinamento. É um *benchmark* crucial para a pesquisa em RL *offline* (também conhecida como *Batch RL*), permitindo que algoritmos sejam avaliados sem a necessidade de interação adicional com o ambiente. O dataset é associado ao *paper* "An Optimistic Perspective on Offline Reinforcement Learning" (ICML 2020) [1].

## Statistics
- **Jogos:** 60 jogos do Atari 2600.
- **Frames de Treinamento:** 200 milhões de *frames* por jogo.
- **Tuplas de Experiência:** Aproximadamente 50 milhões de tuplas de experiência (*observation, action, reward, next observation*) por *run* de 200 milhões de *frames*.
- **Versões:** O processo de coleta foi repetido 5 vezes para cada jogo, resultando em 5 *runs* de dados por jogo.
- **Tamanho Estimado:** Cada *run* de dados é aproximadamente 3.5 vezes maior que o ImageNet (o que sugere um tamanho massivo, embora o tamanho exato em GB não seja fornecido, a escala é de terabytes) [3].

## Features
- **Diversidade de Jogos:** Contém dados de 60 jogos diferentes do Atari 2600.
- **Experiência Completa de Replay:** Armazena toda a experiência de *replay* do agente DQN, totalizando aproximadamente 50 milhões de tuplas de experiência por *run* de 200 milhões de *frames*.
- **Configuração Padrão:** Coletado sob o protocolo padrão de 200 milhões de *frames* e com *sticky actions* (probabilidade de 25% de repetir a ação anterior).
- **Benchmark de RL Offline:** Serve como um *benchmark* de alta qualidade para o desenvolvimento e teste de algoritmos de Aprendizagem por Reforço *Offline* (Batch RL).
- **Acesso via GCP:** Os dados são hospedados em um *bucket* público do Google Cloud Platform (GCP).

## Use Cases
- **Aprendizagem por Reforço Offline (Batch RL):** Treinamento e avaliação de algoritmos de RL que aprendem a partir de um conjunto de dados fixo, sem interação adicional com o ambiente.
- **Avaliação de Generalização:** Testar a capacidade dos algoritmos de RL *offline* de generalizar para políticas de alta qualidade a partir de dados subótimos ou de políticas anteriores.
- **Pesquisa em DQN:** Análise do comportamento e da experiência de *replay* de agentes DQN.
- **Comparação de Algoritmos:** Serve como um *benchmark* para comparar o desempenho de diferentes algoritmos de RL *offline* (como REM, CQL, etc.) [1].

## Integration
O dataset está hospedado no *bucket* público do Google Cloud Platform (GCP) `gs://atari-replay-datasets`. O acesso e o *download* são realizados utilizando a ferramenta de linha de comando `gsutil`.

**Instruções de Download:**
1. **Instalar `gsutil`:** Siga as instruções de instalação do `gsutil` [2].
2. **Download Completo:** Para baixar o dataset completo (todos os 60 jogos):
   ```bash
   gsutil -m cp -R gs://atari-replay-datasets/dqn ./
   ```
3. **Download por Jogo:** Para baixar o dataset de um jogo específico (substitua `[NOME_DO_JOGO]` pelo nome do jogo, e.g., `Pong`):
   ```bash
   gsutil -m cp -R gs://atari-replay-datasets/dqn/[NOME_DO_JOGO] ./
   ```
**Observação Importante:** O dataset foi gerado usando uma versão legada das ROMs do Atari (`atari-py<=0.2.5`). Para evitar incompatibilidades, é recomendado usar `atari-py<=0.2.5` e `gym<=0.19.0` ou seguir as instruções no *site* oficial para o uso com versões mais recentes do `ale-py` [1]. O dataset também está disponível no formato `tfds` (TensorFlow Datasets) como parte do **RL Unplugged** [1].

## URL
[https://offline-rl.github.io/](https://offline-rl.github.io/)
