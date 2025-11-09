# Prompts de Controle de Robótica (Robotics Control Prompts)

## Description
A técnica de **Prompts de Controle de Robótica** (Robotics Control Prompts) refere-se ao uso de **Large Language Models (LLMs)** e **Vision-Language Models (VLMs)** para gerar planos de alto nível, código ou sequências de ações para robôs, permitindo que humanos interajam com sistemas robóticos usando linguagem natural [1] [2]. Em vez de programar cada movimento, o usuário fornece uma instrução em linguagem natural (o *prompt*), e o LLM atua como um planejador de alto nível, decompondo a tarefa em subtarefas e, frequentemente, gerando código executável (como Python) ou comandos de baixo nível que o robô pode processar [3]. Sistemas avançados, como o **ELLMER** e o **ARRC**, separam o planejamento de alto nível do LLM do controle de baixo nível do robô, garantindo segurança e reatividade. O prompt é crucial para definir o contexto, as ferramentas disponíveis (APIs do robô) e as restrições de segurança, transformando o LLM em um agente de raciocínio e planejamento para a robótica [4].

## Examples
```
**1. Prompt de Planejamento de Tarefa (Code-as-Policies):**

```
Você é um planejador de robôs. Sua tarefa é gerar um código Python que utilize as funções 'move_to(x, y, z)', 'grasp(object_name)', e 'release()' para completar a tarefa.

Instrução: "Pegue o bloco vermelho da mesa e coloque-o no cesto azul."

Ferramentas disponíveis:
- get_coordinates(object_name): Retorna as coordenadas (x, y, z) de um objeto.
- move_to(x, y, z): Move o efetor final para a posição.
- grasp(object_name): Fecha a garra no objeto.
- release(): Abre a garra.

Gere apenas o código Python.
```

**2. Prompt de Raciocínio com Ambiguidade (ELLMER):**

```
Contexto: O usuário disse "Estou cansado, por favor, me prepare uma bebida quente e decore o prato com um animal."
Estado Atual: O robô está na posição inicial. A câmera detecta uma caneca, um pote de café instantâneo e uma chaleira.

Tarefa: Decomponha a instrução do usuário em uma sequência lógica de 5 a 7 passos de alto nível. Considere que "bebida quente" implica café e "decorar o prato" é uma tarefa secundária.
```

**3. Prompt de Correção de Erro (Feedback Loop):**

```
Plano Original: move_to(x_caneca, y_caneca, z_caneca)
Feedback do Sensor: Falha na preensão. O sensor de força indica que a garra deslizou.
Instrução: Adapte o plano para tentar novamente a preensão da caneca. Aumente a força de preensão em 10% e tente um ângulo de aproximação ligeiramente diferente (z+5mm). Gere o novo código Python.
```

**4. Prompt de Geração Aumentada por Recuperação (RAG/ARRC):**

```
Você é o sistema de controle ARRC. Use o conhecimento recuperado (abaixo) para gerar um plano de ação JSON seguro para a tarefa.

Conhecimento Recuperado:
- Heurística de Segurança: 'Nunca exceder 50% da velocidade máxima ao manusear líquidos quentes.'
- Template de Tarefa: 'Pick-and-Place: [aproximar, prender, levantar, mover, soltar, retrair].'

Instrução: "Mova a garrafa de água da posição A para a posição B."

Gere o plano de ação no formato JSON.
```

**5. Prompt de Consulta de Estado (Debugging):**

```
Instrução: "Descreva o estado atual do ambiente e do robô em termos de objetos detectados, suas coordenadas e a posição da garra."

Resposta Esperada: Uma lista formatada de objetos e a posição do efetor final.
```
```

## Best Practices
**1. Estruturação do Prompt (Code-as-Policies):** Sempre solicite ao LLM que gere código executável (e.g., Python) em vez de apenas texto descritivo. Defina claramente as funções e ferramentas robóticas disponíveis (APIs) no prompt do sistema. **2. Separação de Responsabilidades (ELLMER/ARRC):** Use o LLM apenas para planejamento de alto nível e raciocínio. O controle de baixo nível (cinemática, segurança, detecção de colisão) deve ser tratado por módulos de controle locais e robustos do robô. **3. Geração Aumentada por Recuperação (RAG):** Para tarefas complexas ou específicas, utilize RAG para fornecer ao LLM conhecimento contextualizado, como manuais de operação, heurísticas de segurança ou exemplos de tarefas anteriores, melhorando a validade e a adaptabilidade do plano [2]. **4. Feedback e Iteração:** Inclua mecanismos de feedback visual (VLM) e de força/tato para que o LLM possa iterar e corrigir o plano em tempo real se o estado do mundo não corresponder ao esperado. **5. Restrições de Segurança:** Incorpore restrições de segurança (limites de espaço de trabalho, velocidade, força) no prompt e no sistema de execução para evitar movimentos perigosos ou falhas catastróficas.

## Use Cases
**1. Manipulação e Montagem:** Robôs que executam tarefas complexas de montagem ou manipulação de objetos em armazéns ou linhas de produção, recebendo instruções em linguagem natural (e.g., "Monte a cadeira usando as peças A, B e C"). **2. Robótica de Serviço:** Robôs em ambientes domésticos ou de escritório que respondem a comandos ambíguos ou de alto nível (e.g., "Limpe a mesa e me traga um copo d'água") [3]. **3. Planejamento de Missão:** Robôs autônomos (drones, rovers) que planejam rotas e sequências de ações em ambientes não estruturados com base em objetivos de missão em linguagem natural (e.g., "Explore a área sul e colete amostras de solo"). **4. Robótica Colaborativa (Cobots):** Agentes robóticos que trabalham ao lado de humanos e precisam adaptar seus planos rapidamente com base em comandos verbais ou gestos. **5. Geração de Políticas de Controle:** Uso de LLMs para gerar políticas de controle (sequências de ações) que podem ser usadas para treinar modelos de *Reinforcement Learning* (Aprendizado por Reforço) de forma mais eficiente.

## Pitfalls
**1. Alucinações e Insegurança:** O LLM pode gerar planos ou códigos que são fisicamente impossíveis, inseguros ou que violam restrições de segurança. **Mitigação:** Implementar "guardas de segurança" (guarded execution) que validam o código ou plano antes da execução e controlam o movimento de baixo nível [2]. **2. Falta de Aterramento (Grounding):** O LLM pode não conseguir mapear conceitos abstratos da linguagem natural para o estado real do mundo (e.g., não saber onde está o "bloco vermelho"). **Mitigação:** Usar VLMs e sistemas de percepção para fornecer ao LLM coordenadas métricas e feedback visual em tempo real. **3. Dependência Excessiva:** Confiar no LLM para o controle de baixo nível resulta em latência e falta de reatividade a eventos em tempo real (e.g., uma colisão inesperada). **Mitigação:** Usar o LLM apenas para planejamento de alto nível e deixar o controle reativo e de baixa latência para o sistema de controle local do robô. **4. Complexidade do Prompt:** Prompts longos e excessivamente detalhados podem confundir o LLM ou atingir o limite de contexto. **Mitigação:** Usar a técnica de *chain-of-thought* para decompor a tarefa e fornecer as ferramentas (APIs) de forma clara e concisa no prompt do sistema. **5. Dificuldade na Depuração:** O código ou plano gerado pelo LLM pode ser difícil de depurar, pois a lógica é gerada dinamicamente. **Mitigação:** Exigir que o LLM gere código bem comentado e usar um sistema de *logging* robusto para rastrear a execução do plano.

## URL
[https://arxiv.org/html/2510.05547v1](https://arxiv.org/html/2510.05547v1)
