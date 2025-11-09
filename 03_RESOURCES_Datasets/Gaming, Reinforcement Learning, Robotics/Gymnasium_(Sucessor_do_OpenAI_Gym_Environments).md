# Gymnasium (Sucessor do OpenAI Gym Environments)

## Description
Gymnasium é o sucessor mantido da biblioteca **OpenAI Gym**, estabelecendo-se como o padrão de fato para o desenvolvimento e comparação de algoritmos de Aprendizagem por Reforço (RL). Ele fornece uma API padronizada e uma coleção diversificada de ambientes de referência, como Classic Control, Box2D, MuJoCo e Atari, permitindo que pesquisadores e desenvolvedores testem e avaliem seus agentes de RL em simulações controladas. O projeto é ativamente mantido pela Farama Foundation, garantindo compatibilidade com as versões mais recentes do Python e as melhores práticas de RL.

## Statistics
*   **Versão Atual:** v1.2.2 (Novembro de 2025).
*   **Downloads:** Mais de 18 milhões de downloads totais, com mais de 1 milhão de downloads mensais em 2025 (referente ao paper de 2024).
*   **Ambientes:** Centenas de ambientes de referência distribuídos em pacotes principais e extensões (e.g., `gymnasium-robotics`, `gymnasium-atari`).
*   **Tamanho do "Dataset":** Não é um dataset estático, mas uma coleção de ambientes de simulação. O tamanho do pacote base é pequeno, mas as dependências (como MuJoCo) podem ser grandes.
*   **Suporte Python:** 3.9, 3.10, 3.11, 3.12, 3.13.

## Features
*   **API Padronizada:** Interface unificada para todos os ambientes de RL, facilitando a troca de agentes entre diferentes tarefas.
*   **Diversidade de Ambientes:** Inclui categorias como Classic Control (e.g., CartPole), Box2D (e.g., LunarLander), MuJoCo (simulações de robótica) e Atari (jogos clássicos).
*   **Suporte a Wrappers:** Permite a modificação e aumento fácil dos ambientes (e.g., normalização de observações, adição de limites de tempo).
*   **Compatibilidade:** Suporte ativo para Python 3.9+ e integração com as principais bibliotecas de RL.
*   **Foco em Manutenção:** Soluciona problemas de manutenção e inconsistências presentes na versão original do OpenAI Gym.

## Use Cases
*   **Desenvolvimento de Agentes de RL:** Plataforma primária para a criação, treinamento e teste de novos algoritmos de Aprendizagem por Reforço.
*   **Benchmarking:** Uso padrão para comparar o desempenho de diferentes algoritmos de RL em tarefas conhecidas.
*   **Robótica e Controle:** Simulação de tarefas de controle e manipulação robótica (via ambientes MuJoCo e extensões).
*   **Educação e Pesquisa:** Ferramenta essencial para o ensino e a pesquisa em inteligência artificial e aprendizado de máquina.
*   **Testes de Robustez:** Avaliação da capacidade de generalização de agentes em diferentes configurações de ambiente.

## Integration
A instalação é feita através do gerenciador de pacotes `pip`. Ambientes específicos podem exigir dependências adicionais (e.g., `\[classic-control\]`, `\[box2d\]`, `\[mujoco\]`).

**Instalação Básica:**
```bash
pip install gymnasium
```

**Instalação com Ambientes Comuns:**
```bash
pip install "gymnasium[classic-control, box2d]"
```

**Exemplo de Uso (Python):**
```python
import gymnasium as gym

# Cria o ambiente CartPole
env = gym.make("CartPole-v1", render_mode="human")

# Reinicia o ambiente
observation, info = env.reset(seed=42)

for _ in range(1000):
    # Seleciona uma ação aleatória
    action = env.action_space.sample()

    # Executa a ação
    observation, reward, terminated, truncated, info = env.step(action)

    # Verifica se o episódio terminou ou foi truncado
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## URL
[https://gymnasium.farama.org/](https://gymnasium.farama.org/)
