# Meta-World (Multi-task RL)

## Description
Meta-World é um *benchmark* simulado de código aberto para o desenvolvimento e avaliação de algoritmos de aprendizado por reforço multi-tarefa (*Multi-task RL*) e meta-aprendizado por reforço (*Meta-RL*). O *benchmark* consiste em 50 ambientes distintos de manipulação robótica em um simulador (MuJoCo), projetados para serem diversos, mas com estrutura compartilhada. O objetivo principal é permitir o desenvolvimento de algoritmos que possam generalizar e acelerar a aquisição de novas habilidades, avaliando a capacidade de aprender um conjunto de habilidades simultaneamente ou de se adaptar rapidamente a novas tarefas.

## Statistics
- **Número de Tarefas:** 50 tarefas distintas de manipulação robótica.
- **Versões:** A versão mais recente é a **v3.0.0** (lançada em Junho de 2025, conforme o repositório GitHub).
- **Benchmarks:** 6 modos de avaliação (MT1, MT10, MT50, ML1, ML10, ML45).
- **Contagem de Amostras/Tamanho:** Não há um tamanho de dataset fixo em termos de gigabytes, pois é um *benchmark* de ambientes simulados. O número de "amostras" (interações) é gerado dinamicamente durante o treinamento e avaliação, podendo chegar a milhões de *timesteps* por tarefa.

## Features
- **50 Tarefas Distintas:** Inclui tarefas como alcançar, empurrar, pegar e colocar, abrir e fechar portas/gavetas, e pressionar botões, todas em um ambiente de manipulação robótica.
- **Múltiplos Benchmarks:** Oferece seis modos de avaliação com diferentes níveis de dificuldade:
    - **Multi-Task (MT1, MT10, MT50):** Para aprendizado simultâneo de 1, 10 ou 50 tarefas.
    - **Meta-Learning (ML1, ML10, ML45):** Para avaliação da capacidade de adaptação a variações de objetivos (ML1) ou a novas tarefas (ML10, ML45) com poucos exemplos.
- **Ambiente de Simulação:** Utiliza o simulador MuJoCo para as interações robóticas.
- **API Compatível com Gymnasium:** A API segue o padrão Gymnasium (anteriormente Gym), facilitando a integração com frameworks de RL existentes.

## Use Cases
- **Avaliação de Algoritmos de Meta-Aprendizado por Reforço (Meta-RL):** Testar a capacidade de agentes de aprender a aprender e se adaptar rapidamente a novas tarefas com poucos exemplos.
- **Avaliação de Algoritmos de Aprendizado por Reforço Multi-Tarefa (Multi-task RL):** Desenvolver políticas que possam resolver um conjunto diversificado de tarefas simultaneamente.
- **Pesquisa em Generalização e Transferência de Conhecimento:** Estudar como o conhecimento adquirido em um conjunto de tarefas pode ser transferido para acelerar o aprendizado em tarefas não vistas.
- **Desenvolvimento de Agentes de Manipulação Robótica:** Servir como um ambiente de teste padronizado para agentes de RL focados em habilidades de manipulação fina.

## Integration
O Meta-World é distribuído como um pacote Python e pode ser instalado via `pip`.

**Instalação:**
A instalação mais recente e recomendada é feita diretamente do repositório GitHub da Farama Foundation:
```bash
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
```

**Uso Básico (Exemplo MT1):**
O *benchmark* é acessado através da API `gymnasium.make`.
```python
import gymnasium as gym
import metaworld

# Cria um ambiente MT1 (tarefa única) para 'reach-v3'
env = gym.make("Meta-World/MT1", env_name="reach-v3")

observation, info = env.reset()
for _ in range(500):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```
Para benchmarks multi-tarefa (MT10, MT50) ou meta-aprendizado (ML10, ML45), é necessário usar `gym.make_vec` para criar ambientes vetoriais.

## URL
[https://meta-world.github.io/](https://meta-world.github.io/)
