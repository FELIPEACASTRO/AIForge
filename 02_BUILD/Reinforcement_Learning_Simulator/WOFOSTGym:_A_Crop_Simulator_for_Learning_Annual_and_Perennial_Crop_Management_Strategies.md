# WOFOSTGym: A Crop Simulator for Learning Annual and Perennial Crop Management Strategies

## Description

WOFOSTGym é um novo ambiente de simulação de culturas, baseado no modelo WOFOST (WOrld FOod STudies) e construído sobre a estrutura Gymnasium (anteriormente OpenAI Gym). Ele foi projetado para treinar agentes de Aprendizagem por Reforço (RL) na otimização de decisões de agrogereciamento para culturas anuais e perenes, em configurações de fazenda única e multifazenda. O simulador aborda a lacuna de ambientes de RL que modelam culturas perenes e múltiplos cultivos anuais, oferecendo um ambiente complexo com observabilidade parcial, dinâmicas não-Markovianas e feedback atrasado.

## Statistics

Publicado em 2025 (arXiv:2502.19308). Citado por 4 (dados de novembro de 2025). O modelo subjacente WOFOST é amplamente validado na literatura agronômica.

## Features

Suporte para 23 culturas anuais e 2 culturas perenes. Interface RL padrão (Gymnasium). Suporte nativo para algoritmos de RL Online (PPO, SAC, DQN) e geração de dados para RL Offline e Transfer Learning. Permite a otimização de estratégias de irrigação, fertilização e outras decisões de manejo.

## Use Cases

Otimização de decisões de agrogereciamento (irrigação, fertilização, plantio) para maximizar o rendimento e o retorno econômico, minimizando o impacto ambiental. Pesquisa e desenvolvimento de agentes de RL para agricultura de precisão.

## Integration

A integração é feita através da estrutura Gymnasium. Exemplo de treinamento de agente PPO via linha de comando:\n```bash\npython3 train_agent.py --agent-type PPO --save-folder logs/ppo/\n```\nPermite a configuração de hiperparâmetros via linha de comando (ex: `--PPO.gamma 0.85`).

## URL

https://github.com/Intelligent-Reliable-Autonomous-Systems/WOFOSTGym