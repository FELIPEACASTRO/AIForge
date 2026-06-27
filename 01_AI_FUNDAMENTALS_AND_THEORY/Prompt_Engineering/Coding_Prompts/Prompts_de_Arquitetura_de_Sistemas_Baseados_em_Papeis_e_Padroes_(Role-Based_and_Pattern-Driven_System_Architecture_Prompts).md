# Prompts de Arquitetura de Sistemas Baseados em Papéis e Padrões (Role-Based and Pattern-Driven System Architecture Prompts)

## Description

A Engenharia de Prompts para Design de Arquitetura de Sistemas é uma disciplina emergente que utiliza Large Language Models (LLMs) para auxiliar arquitetos de software em diversas tarefas, desde a geração de designs iniciais a partir de requisitos até a análise e revisão de arquiteturas existentes. A técnica se baseia na definição clara de um **papel** para o LLM (ex: "Arquiteto de Nuvem Sênior") e na aplicação de **padrões de prompt** (ex: Chain-of-Thought, Role-Based Templates) para guiar o modelo através de um processo estruturado de design. Isso permite que os LLMs atuem como "coaches" ou assistentes, aumentando a produtividade e a qualidade das decisões arquiteturais.

## Statistics

- **Adoção:** O uso de LLMs em tarefas de arquitetura de software está em forte crescimento, com um aumento acentuado de publicações acadêmicas em 2024 e 2025 (Fonte: *Software Architecture Meets LLMs: A Systematic Literature Review*, 2025).
- **Automação:** 71% dos trabalhos acadêmicos utilizam LLMs de forma automatizada ou semi-automatizada para tarefas de arquitetura.
- **Eficácia:** Estudos mostram que LLMs frequentemente **superam as linhas de base** (baselines) em tarefas como classificação de decisões de design e recuperação de links de rastreabilidade.
- **Modelos Comuns:** GPT-4, GPT-3.5 e BERT são os modelos mais utilizados em pesquisas na área.

## Features

- **Geração de Design:** Criação de designs de arquitetura a partir de requisitos funcionais e não-funcionais.
- **Classificação e Detecção:** Identificação de padrões de design, táticas arquiteturais e decisões de design em código ou documentação.
- **Revisão de Código/Arquitetura:** Análise de diagramas e código com base em políticas e padrões definidos (ex: `Code review with policy`).
- **Assistência a Decisões:** Ajuda na seleção, avaliação e captura de decisões arquiteturais (ADRs - Architecture Decision Records).
- **Transformação:** Conversão de esboços (whiteboard/sketch) em diagramas digitais formais (requer LLMs multimodais).

## Use Cases

- **Geração de Arquitetura a partir de Requisitos:** Criar um design inicial de microsserviços a partir de uma lista de histórias de usuário.
- **Análise de Dívida Técnica:** Priorizar e explicar problemas identificados em análises formais de código para públicos não-técnicos.
- **Coaching de Arquitetura:** Diálogo contínuo para aplicar padrões como "branching by abstraction" em funções serverless.
- **Scaffolding de Testes:** Geração de testes de integração críticos a partir de esquemas de API (ex: OpenAPI/Swagger).

## Integration

### Melhores Práticas:
1. **Definição de Papel (Role Definition):** Comece o prompt definindo o papel do LLM (ex: "Aja como um Arquiteto de Soluções Sênior com 15 anos de experiência em sistemas distribuídos e AWS").
2. **Contexto e Restrições:** Forneça o máximo de contexto possível, incluindo requisitos não-funcionais (escalabilidade, segurança, custo) e restrições tecnológicas (linguagens, provedores de nuvem).
3. **Estrutura de Saída:** Peça a saída em um formato estruturado (ex: Markdown, PlantUML, JSON) e defina as seções esperadas (ex: Diagrama de Componentes, Justificativas, Riscos).
4. **Iteração e Refinamento (Chain-of-Thought):** Use prompts de acompanhamento para refinar o design, solicitando justificativas, análise de riscos ou alternativas (ex: "Agora, analise os riscos de segurança do componente de autenticação e proponha 3 mitigações.").

### Exemplo de Prompt (Role-Based Template):

```
**Papel:** Você é um Arquiteto de Software Sênior especializado em arquiteturas orientadas a eventos e serverless.

**Tarefa:** Projete a arquitetura de alto nível para um novo serviço de processamento de pedidos (Order Processing Service).

**Requisitos:**
1. **Funcionais:** Receber pedidos, validar estoque, notificar o sistema de faturamento.
2. **Não-Funcionais:** Alta disponibilidade (99.99%), escalabilidade para 1000 pedidos/segundo, custo otimizado (serverless preferencialmente).
3. **Restrições:** Deve usar AWS, Python para funções de processamento, e um banco de dados NoSQL para o catálogo de pedidos.

**Instruções de Saída:**
1. **Diagrama de Componentes:** Descreva os componentes principais (ex: API Gateway, Lambda, SQS, DynamoDB) e suas interações.
2. **Justificativa Tecnológica:** Explique por que a arquitetura serverless/orientada a eventos é a melhor escolha.
3. **Análise de Risco:** Identifique o principal gargalo de escalabilidade e a estratégia de mitigação.
4. **Formato:** Use Markdown para a descrição e um formato de lista para os componentes.
```

## URL

https://github.com/mikaelvesavuori/chatgpt-architecture-coach