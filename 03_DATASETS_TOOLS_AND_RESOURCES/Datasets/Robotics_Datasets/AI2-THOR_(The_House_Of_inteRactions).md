# AI2-THOR (The House Of inteRactions)

## Description
AI2-THOR (The House Of inteRactions) é um framework e ambiente de simulação 3D fotorrealista, desenvolvido pelo Allen Institute for AI (AI2), para pesquisa em Inteligência Artificial Visual e Embarcada (Embodied AI). Ele fornece um ambiente interativo onde agentes de IA podem navegar e interagir com objetos em cenas internas (cozinhas, quartos, banheiros, salas de estar) para realizar tarefas complexas. O ambiente é construído com Unity 3D, o que possibilita simulação física para objetos e cenas, incluindo estados visuais de objetos (abrir/fechar, ligar/desligar, quente/frio). O projeto evoluiu para incluir ambientes mais complexos como o ProcTHOR, que utiliza geração procedural para criar um número massivo de ambientes.

## Statistics
- **iTHOR:** 120 salas (cozinhas, quartos, banheiros, salas de estar), mais de 2000 objetos únicos.
- **RoboTHOR:** 89 apartamentos com mais de 600 objetos, com contrapartes físicas e simuladas para 14 apartamentos.
- **ProcTHOR (Versão mais recente):** 10.000 casas geradas proceduralmente, oferecendo um volume massivo de dados para treinamento.
- **Tamanho do Binário:** O ambiente 3D (Unity) é baixado na primeira execução e tem aproximadamente 500MB.
- **Versões:** A documentação mais recente disponível publicamente é para a versão 2.1.0, mas o projeto é ativamente mantido no GitHub, com atualizações e novos *frameworks* como ProcTHOR (anunciado em 2022).

## Features
- **Simulação Fotorrealista:** Cenas 3D de alta qualidade baseadas em Unity 3D.
- **Interação Física:** Suporte a simulação física para objetos, permitindo que agentes interajam com eles (empurrar, pegar, abrir, etc.).
- **Estados de Objetos:** Objetos com estados visuais mutáveis (e.g., torradeira ligada/desligada, porta aberta/fechada).
- **Múltiplos Agentes:** Suporte para múltiplos agentes na mesma cena e diferentes tipos de agentes (humanoides, drones).
- **Sub-ambientes:** Inclui iTHOR (navegação e interação), RoboTHOR (transferência simulação-para-real com robôs LoCoBot) e ManipulaTHOR (manipulação de objetos com braço robótico).
- **Escalabilidade (ProcTHOR):** A versão mais recente, ProcTHOR, permite a geração procedural de 10.000 casas, oferecendo um volume massivo de dados para treinamento de modelos de IA Embarcada.

## Use Cases
- **Inteligência Artificial Embarcada (Embodied AI):** Treinamento e avaliação de agentes que precisam interagir com o mundo físico.
- **Navegação Visual:** Desenvolvimento de modelos para navegação em ambientes internos complexos.
- **Manipulação de Objetos:** Pesquisa em tarefas de manipulação fina e planejamento de longo prazo com braços robóticos (ManipulaTHOR).
- **Transferência Simulação-para-Real (Sim2Real):** Uso do RoboTHOR para testar a generalização de modelos treinados em simulação para robôs físicos (LoCoBot).
- **Aprendizado por Reforço (RL):** Plataforma para o desenvolvimento de agentes de RL em tarefas que exigem raciocínio e interação complexa.
- **Resolução de Tarefas Multietapas:** Criação de agentes capazes de seguir instruções de linguagem natural para completar sequências de ações (e.g., cozinhar, arrumar um quarto).

## Integration
O AI2-THOR é instalado como uma biblioteca Python via `pip`.
1. **Instalação:** Recomenda-se um ambiente virtual Python (3.5+).
   ```bash
   pip install ai2thor
   ```
2. **Uso Básico (Python):** O ambiente 3D é baixado automaticamente (aprox. 500MB) na primeira execução.
   ```python
   import ai2thor.controller
   
   controller = ai2thor.controller.Controller()
   # Inicia o ambiente 3D
   controller.start() 
   
   # Exemplo de ação: mover o agente para frente
   event = controller.step(action="MoveAhead")
   
   # Para a simulação
   controller.stop()
   ```
3. **Documentação:** A documentação oficial fornece detalhes sobre as ações disponíveis (`MoveAhead`, `RotateLeft`, `PickupObject`, etc.) e a estrutura de metadados retornada.
4. **Requisitos:** Requer um servidor X com OpenGL (para usuários Linux) e uma placa gráfica com suporte a DX9 (shader model 3.0) ou DX11. O uso de renderização *headless* é suportado para clusters de computação.

## URL
[https://ai2thor.allenai.org/](https://ai2thor.allenai.org/)
