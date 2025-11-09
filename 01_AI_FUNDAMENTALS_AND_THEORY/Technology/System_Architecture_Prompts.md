# System Architecture Prompts

## Description
"System Architecture Prompts" (Prompts de Arquitetura de Sistema) representa uma mudança de foco de um simples exercício de *prompting* para uma disciplina de **design de sistema** ao construir aplicações robustas baseadas em Large Language Models (LLMs) [1]. Em vez de ser uma técnica de prompt isolada, é uma abordagem que integra o LLM como um componente central em uma arquitetura de software mais ampla, onde o prompt é o ponto de controle para guiar o raciocínio do modelo. O principal desafio é construir um sistema que possa direcionar a natureza probabilística e, por vezes, imprevisível dos LLMs para resultados confiáveis e precisos [1].

Os prompts de arquitetura são frequentemente utilizados em conjunto com padrões como **Retrieval-Augmented Generation (RAG)**, onde o prompt é *aumentado* com contexto factual recuperado de uma base de conhecimento externa, e em arquiteturas de **Agentes (Agents)**, onde o LLM usa o prompt para decidir qual ferramenta ou API chamar para executar uma tarefa [1]. O objetivo é transformar o LLM de um "sabe-tudo" falível em um "motor de raciocínio" que opera sobre um conjunto de fatos fornecidos [1].

## Examples
```
1. **Planejamento de Escalabilidade (RAG-Augmented):** "Com base nestas métricas de sistema [inserir métricas de tráfego e latência], identifique possíveis gargalos no banco de dados e na camada de microsserviços. Proponha três soluções de *sharding* e compare-as com base no custo e na complexidade de implementação. Use o contexto fornecido para justificar sua escolha."
2. **Design de Arquitetura de Software:** "Atue como um Arquiteto de Soluções Sênior. Projete uma arquitetura de microsserviços *serverless* para um sistema de processamento de pedidos de e-commerce que espera um pico de 10.000 pedidos por minuto. O design deve incluir: serviços principais, *message queues*, banco de dados (escolha entre NoSQL ou SQL e justifique), e um diagrama de alto nível em formato Mermaid. Considere a resiliência e a observabilidade."
3. **Análise de Segurança:** "Analise o seguinte trecho de código Python [inserir trecho de código] para vulnerabilidades de injeção de SQL e XSS. Além disso, sugira um mecanismo de criptografia de ponta a ponta para a comunicação entre o serviço de autenticação e o serviço de pagamento, detalhando o protocolo (TLS 1.3, AES-256)."
4. **Recomendação de Padrões de Design:** "Estamos desenvolvendo um sistema de notificação em tempo real que precisa enviar mensagens para milhões de usuários via e-mail, SMS e *push notification*. Descreva o padrão de design mais adequado (ex: *Observer*, *Pub/Sub*, *Event Sourcing*) para gerenciar a distribuição de mensagens e justifique por que ele é superior aos outros para este caso de uso."
5. **Assistência ao Design de Modelo de Dados:** "Crie um esquema de dados otimizado para um banco de dados relacional (PostgreSQL) para um sistema de gerenciamento de inventário. As entidades principais são: Produto, Armazém, Fornecedor e Transação de Estoque. Inclua as chaves primárias, chaves estrangeiras e os índices necessários para otimizar consultas de baixa latência sobre o nível de estoque em um armazém específico."
6. **Análise Comparativa de Tecnologia:** "Forneça uma análise comparativa entre Kubernetes (K8s) e AWS ECS para orquestração de contêineres, considerando os seguintes critérios: custo operacional, curva de aprendizado da equipe, facilidade de integração com CI/CD e capacidade de escalabilidade horizontal. Conclua com uma recomendação para uma startup com foco em velocidade de desenvolvimento."
7. **Geração de Documentação Técnica:** "Gere um *outline* detalhado para a documentação técnica de um novo serviço de *Machine Learning Inference*. O *outline* deve incluir seções para: Visão Geral da Arquitetura, Diagrama de Fluxo de Dados, Requisitos de Infraestrutura (CPU/GPU), Estratégia de *Rollback* e Plano de Monitoramento (métricas de latência e erro)."
```

## Best Practices
- **Seja Específico e Contextual:** Forneça o máximo de detalhes possível sobre o sistema, métricas, parâmetros e restrições. O prompt deve ser um reflexo da documentação de requisitos.
- **Utilize Marcadores de Posição:** Use colchetes (`[]`) para indicar claramente onde o LLM deve inserir informações específicas (código, métricas, descrições), facilitando a automação.
- **Habilite a Interação:** Adicione uma instrução como "Faça perguntas se precisar de mais informações" para permitir que o LLM solicite o contexto ausente, melhorando a qualidade da resposta.
- **Integre com RAG:** Para garantir a precisão e o conhecimento atualizado, utilize o prompt no estágio de **Contextual Prompting** de um pipeline RAG, fornecendo ao LLM o contexto factual recuperado de sua base de conhecimento proprietária [1].
- **Decomponha o Problema:** Para tarefas complexas, use técnicas de orquestração como **Chain-of-Thought (CoT)** para forçar o LLM a decompor o problema em etapas lógicas antes de fornecer a solução final.

## Use Cases
- **Design de Arquitetura:** Geração de propostas de arquitetura (microsserviços, monolito, *serverless*), incluindo diagramas de alto nível e justificativas de escolha de tecnologia.
- **Análise de Vulnerabilidade:** Revisão de código e design de sistema para identificar falhas de segurança e sugerir mecanismos de proteção (criptografia, autenticação).
- **Otimização de Desempenho e Escalabilidade:** Identificação de gargalos, sugestão de estratégias de *sharding*, balanceamento de carga e otimização de consultas de banco de dados.
- **Modelagem de Dados:** Criação de esquemas de dados otimizados para diferentes tipos de bancos de dados (SQL, NoSQL) com base em requisitos de consulta e volume de dados.
- **Geração de Documentação:** Criação de *outlines*, templates e rascunhos de documentação técnica, planos de recuperação de desastres (DRP) e planos de continuidade de negócios (BCP).
- **Avaliação de Tecnologia:** Análise comparativa de *stacks* tecnológicas (ex: *frameworks*, provedores de nuvem, linguagens de programação) com base em critérios definidos (custo, performance, suporte).
- **Garantia de Conformidade:** Verificação de que o design da arquitetura atende a regulamentações específicas da indústria (ex: GDPR, LGPD, HIPAA).

## Pitfalls
- **Falta de Detalhes:** Prompts vagos levam a respostas genéricas e inúteis. A arquitetura de software é complexa e exige especificações claras (métricas, *stack* tecnológica, restrições de custo/tempo).
- **Confiança Cega (*Over-reliance*):** Assumir que a saída do LLM é infalível, especialmente em questões críticas como segurança, custo e conformidade regulatória. A saída do LLM deve ser tratada como uma sugestão de um consultor e sempre validada por um arquiteto humano.
- **Ignorar o Contexto Proprietário:** Não fornecer o contexto específico do sistema (dados internos, APIs legadas) no prompt. O LLM não tem conhecimento dos detalhes internos da sua organização.
- **Ausência de RAG/Grounding:** Tentar obter respostas factuais ou específicas do domínio sem aumentar o prompt com dados recuperados, resultando em **alucinações** ou informações desatualizadas sobre o seu sistema.
- **Prompts Monolíticos:** Tentar resolver um problema de arquitetura complexo em um único prompt. A orquestração e a decomposição do problema em prompts menores e encadeados são essenciais.

## URL
[https://medium.com/@vi.ha.engr/the-architects-guide-to-llm-system-design-from-prompt-to-production-8be21ebac8bc](https://medium.com/@vi.ha.engr/the-architects-guide-to-llm-system-design-from-prompt-to-production-8be21ebac8bc)
