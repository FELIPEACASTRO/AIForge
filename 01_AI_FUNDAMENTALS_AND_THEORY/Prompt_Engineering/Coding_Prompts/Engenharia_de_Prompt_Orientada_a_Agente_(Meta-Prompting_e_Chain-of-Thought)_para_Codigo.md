# Engenharia de Prompt Orientada a Agente (Meta-Prompting e Chain-of-Thought) para Código

## Description

Técnicas avançadas de engenharia de prompt que utilizam o LLM para atuar como um **Agente de Refatoração Sênior**, seguindo um fluxo de trabalho estruturado e incremental. O **Meta-Prompting** define o papel, o objetivo, o escopo, as regras de proteção (guardrails) e o fluxo de trabalho detalhado. O **Chain-of-Thought (CoT) Stepwise** garante que o processo seja auditável e controlável, exigindo confirmação do usuário a cada passo.

## Statistics

- **Taxa de Sucesso em Migração:** Um estudo de caso (Medium, ELCA IT) mostrou que LLMs refatoraram com sucesso **65–70% dos métodos** sem intervenção manual em uma migração de Java Spring, resultando em economia significativa de tempo.
- **Redução de Erros:** O uso de CoT Stepwise (Medium, Reynald) é uma melhor prática para **prevenir erros compostos** (compounding errors) em tarefas complexas como refatoração, mantendo o controle do desenvolvedor.
- **Benchmarking:** Pesquisas recentes (arXiv 2025) focam em **benchmarking de LLMs para Revisão de Código**, buscando métricas mais robustas que considerem o contexto completo do projeto, em vez de apenas unidades de código isoladas.

## Features

- **Análise Holística:** Capacidade de analisar o *workspace* completo (mono/multi-repo) e respeitar configurações existentes (package.json, tsconfig, eslintrc).
- **Refatoração Estruturada:** Divide a refatoração em fases (Segura/Mecânica, Moderada, Alto Risco) para gerenciamento de risco.
- **Guardrails de Segurança:** Impõe regras como "Manter APIs públicas estáveis", "Mudanças incrementais e revisáveis" e "Nunca reduzir a cobertura de testes".
- **Diagnóstico Detalhado:** Gera um inventário do repositório e uma tabela de "Descobertas" (Findings) com Severidade, Problema, Evidência, Impacto e Esforço.
- **Fluxo de Trabalho Controlado:** Utiliza o CoT Stepwise para garantir que o desenvolvedor possa revisar e aprovar cada etapa antes da execução.

## Use Cases

- **Modernização de Código Legado:** Refatorar grandes bases de código para melhorar a estrutura, modularidade e conformidade com as melhores práticas modernas.
- **Migração de Frameworks/Linguagens:** Auxiliar na conversão de código entre versões de frameworks ou linguagens, como visto no estudo de caso de Java Spring.
- **Revisão de Código Automatizada:** Atuar como um revisor sênior, identificando *code smells*, falhas de segurança e inconsistências de estilo antes da revisão humana.
- **Aplicação de Padrões de Projeto:** Garantir a aplicação consistente de padrões de projeto (e.g., Inversão de Dependência, Arquitetura Limpa) em todo o repositório.

## Integration

### Melhor Prática 1: Chain-of-Thought (CoT) Stepwise para Refatoração
**Objetivo:** Manter o controle e evitar erros compostos em refatorações complexas.
**Prompt Exemplo:**
```
"Ajude-me a refatorar o código no arquivo 'service.js'. Prossiga um passo de cada vez. Não avance para o próximo passo até que eu forneça a palavra-chave 'próximo'."
```
**Resposta Esperada do LLM (Exemplo):**
```
"Passo 1: Analisar a estrutura atual do código. Identifiquei os seguintes problemas: 1. Funções não relacionadas no mesmo arquivo. 2. Duplicação de código no tratamento de erros. 3. Funções com alta complexidade ciclomática. Digite 'próximo' para eu prosseguir com o primeiro passo de refatoração."
```

### Melhor Prática 2: Meta-Prompting (Template de Agente de Refatoração)
**Objetivo:** Fornecer contexto e restrições abrangentes para uma refatoração de repositório.
**Prompt Template (Adaptado):**
```
"Você é um engenheiro de software sênior atuando como um agente de refatoração de repositório.

OBJETIVO: Analisar todo o espaço de trabalho e propor/refatorar mudanças que tornem a base de código mais estruturada, consistente e de fácil manutenção, seguindo as melhores práticas reconhecidas e as convenções existentes do projeto. Mantenha o comportamento público estável.

ESCOPO:
- Trate isso como um mono- ou multi-repo (detectado automaticamente).
- Respeite ferramentas/configurações existentes (package.json, tsconfig, eslintrc, etc.).
- Exclua caminhos gerados/terceiros (node_modules, .git, dist, etc.).

REGRAS DE PROTEÇÃO (GUARDRAILS):
- Mantenha APIs públicas e contratos externos estáveis.
- Mantenha as alterações incrementais e revisáveis: prefira *diffs* pequenos e focados.
- Adicione ou ajuste testes quando as refatorações não forem triviais; nunca reduza a cobertura de testes intencionalmente.

FLUXO DE TRABALHO:
1. Inventário: Construa um mapa do repositório e resuma as convenções.
2. Descobertas: Produza uma lista de problemas com ID, Severidade, Problema, Evidência e Impacto.
3. Plano de Refatoração: Proponha fases (Segura, Moderada, Alto Risco).
4. Execução: Apresente *diffs* concretos para as 3-5 principais mudanças de alto impacto e baixo risco.

Comece escaneando o espaço de trabalho e produzindo o Mapa do Repositório e a Tabela de Descobertas."
```

## URL

https://imrecsige.dev/snippets/llm-prompt-for-refactoring-your-codebase-using-best-practices/