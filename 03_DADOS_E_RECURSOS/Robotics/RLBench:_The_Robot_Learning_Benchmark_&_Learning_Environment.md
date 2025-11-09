# RLBench: The Robot Learning Benchmark & Learning Environment

## Description
RLBench é um ambicioso benchmark e ambiente de aprendizado em larga escala, projetado para facilitar a pesquisa em diversas áreas de manipulação robótica guiada por visão. O ambiente apresenta **100 tarefas únicas** e projetadas manualmente, variando em dificuldade, desde tarefas simples de alcançar um alvo até manipulações mais complexas. É um recurso fundamental para o desenvolvimento e avaliação de algoritmos de **Aprendizado por Reforço (RL)**, **Aprendizado por Imitação (IL)**, **Aprendizado Multi-Tarefa** e **Aprendizado Few-Shot** em robótica. O ambiente utiliza o simulador CoppeliaSim para fornecer um ambiente de simulação realista e flexível.

## Statistics
O benchmark é composto por **100 tarefas únicas** de manipulação robótica. O dataset de demonstrações é gerado sob demanda, mas subconjuntos pré-gerados existem. Por exemplo, o subconjunto "rlbench-18-tasks" (18 tarefas) com 100 demonstrações por tarefa tem um tamanho total de aproximadamente **~116GB**. A versão mais recente do software mencionada no repositório GitHub é a **v1.2.0**, lançada em 18 de fevereiro de 2022. O artigo original é de 2019/2020.

## Features
O RLBench se destaca por sua coleção de **100 tarefas de manipulação** com dificuldade variada. Suporta diversos modos de ação para o braço robótico (como JointVelocity) e o gripper (como Discrete). É otimizado para pesquisa em **aprendizado few-shot** e **meta-aprendizado** devido à sua ampla gama de tarefas. Inclui suporte para **Imitation Learning** com a capacidade de carregar demonstrações pré-salvas e facilita experimentos de **Sim-to-Real** através da funcionalidade de randomização de domínio visual. O ambiente também possui integração com o popular ecossistema **Gym** (RLBench Gym). As observações de imagem padrão são de 128x128.

## Use Cases
O RLBench é amplamente utilizado para: 1) **Avaliação de Algoritmos de RL** em tarefas de manipulação robótica complexas. 2) **Pesquisa em Aprendizado por Imitação e Multi-Tarefa**, aproveitando o grande número de tarefas e demonstrações. 3) **Estudos de Aprendizado Few-Shot e Meta-Aprendizado**, testando a capacidade de generalização dos modelos. 4) **Experimentos de Transferência Sim-to-Real**, utilizando a randomização de domínio para aumentar a robustez dos modelos.

## Integration
O RLBench é construído sobre o simulador **CoppeliaSim v4.1.0** e a biblioteca **PyRep**. A integração requer a instalação do CoppeliaSim e a configuração de variáveis de ambiente. O pacote Python pode ser instalado via pip diretamente do repositório GitHub: `pip install git+https://github.com/stepjam/RLBench.git`. Para utilizar demonstrações pré-salvas (necessárias para Aprendizado por Imitação), o caminho do dataset deve ser especificado ao inicializar o ambiente: `env = Environment(action_mode, DATASET)`. O dataset de demonstrações deve ser gerado ou baixado separadamente.

## URL
[https://sites.google.com/view/rlbench](https://sites.google.com/view/rlbench)
