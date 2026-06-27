# MIME (Multiple Interactions Made Easy)

## Description
O MIME (Multiple Interactions Made Easy) é um dos maiores e mais diversos datasets de demonstrações robóticas disponíveis. Ele foi criado para impulsionar a pesquisa em aprendizado por imitação (Imitation Learning) em robótica, especialmente em tarefas de manipulação complexas e multi-tarefas. O dataset contém demonstrações de 20 tarefas robóticas distintas, que variam desde ações simples como empurrar até tarefas mais difíceis como empilhar objetos domésticos e abrir garrafas. Cada ponto de dado inclui demonstrações humanas (HD) e demonstrações robóticas (RD) coletadas com um robô Baxter. As demonstrações humanas são capturadas com um Kinect montado na cabeça (RGBD), enquanto as demonstrações robóticas incluem dados de um Kinect montado no alto (RGBD), dois sensores cinéticos suaves montados nos punhos (RGBD) e dados de ângulo de junta do Baxter. O dataset foi introduzido em 2018.

## Statistics
- **Amostras:** 8260 demonstrações humano-robô.
- **Tarefas:** 20 tarefas robóticas distintas.
- **Versão:** Versão única, introduzida no artigo CoRL 2018.
- **Tamanho:** O tamanho total do dataset não foi explicitamente fornecido na página oficial ou no artigo, mas a natureza dos dados (vídeos RGBD e dados cinestésicos) sugere um tamanho considerável (provavelmente na ordem de centenas de GB ou TB).

## Features
- **Diversidade de Tarefas:** Contém demonstrações para 20 tarefas robóticas diferentes, abrangendo uma ampla gama de manipulações.
- **Demonstrações Humanas e Robóticas:** Inclui tanto demonstrações de humanos realizando as tarefas quanto demonstrações do robô Baxter.
- **Dados Multimodais:** Cada ponto de dado é rico em informações, incluindo dados RGBD de múltiplas câmeras (cabeça humana, visão superior do robô, punhos do robô) e dados cinestésicos (ângulos de junta do robô Baxter).
- **Foco em Aprendizado por Imitação Multi-tarefa:** Projetado para permitir o treinamento de políticas de imitação que podem generalizar para múltiplas tarefas.

## Use Cases
- **Aprendizado por Imitação Multi-tarefa:** Treinamento de agentes robóticos para realizar uma ampla variedade de tarefas a partir de demonstrações.
- **Aprendizado de Representação Visual:** Uso dos dados RGBD para aprender representações visuais robustas para manipulação robótica.
- **Previsão de Trajetória:** Avaliação de modelos que preveem trajetórias robóticas com base em demonstrações.
- **Aprendizado de Habilidades Robóticas:** Desenvolvimento de políticas de controle para robôs como o Baxter.

## Integration
O dataset pode ser baixado através de um link do Dropbox fornecido na página oficial do projeto. O download pode ser feito do dataset completo ou por tarefa individual.
**Link de Download Principal:** `https://www.dropbox.com/scl/fo/cgvsmxayv6qqw1ynvxrfr/AFnncxm2YUHp86AI1mm0rSI?rlkey=iqeeazs9bfnx283zt4a3vg4cc&e=1&dl=0`
**Instruções de Uso:** O dataset é tipicamente usado para treinar modelos de aprendizado por imitação. Os dados cinestésicos (ângulos de junta) e os dados visuais (RGBD) são usados para mapear observações para ações do robô. O artigo original ("Multiple Interactions Made Easy (MIME): Large Scale Demonstrations Data for Imitation") deve ser consultado para detalhes sobre a estrutura dos dados e métodos de processamento.

## URL
[https://sites.google.com/view/mimedataset](https://sites.google.com/view/mimedataset)
