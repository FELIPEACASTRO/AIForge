# Engenharia de Prompt para Avaliação e Validação de LLMs

## Description

A Engenharia de Prompt para Avaliação e Validação de LLMs (Large Language Models) é uma disciplina focada em medir a eficácia de um prompt específico em gerar a resposta desejada, em contraste com a avaliação do modelo em si. Ela utiliza prompts de avaliação para julgar a qualidade, relevância, correção e segurança das saídas do LLM. O objetivo é iterar e otimizar prompts para casos de uso específicos, garantindo que o LLM atenda aos critérios de desempenho e segurança definidos. As técnicas envolvem a criação de "assertions" (afirmações) e "metrics" (métricas) que podem ser avaliadas de forma determinística ou assistida por outro LLM (Model-Graded Evaluation).

## Statistics

**Métricas Chave:** A avaliação de prompts se concentra em métricas como **Taxa de Aprovação (Pass Rate)** em testes de regressão, **Latência** (tempo de resposta do LLM) e **Custo** por chamada de API. Frameworks como Promptfoo e PromptLayer rastreiam essas métricas para otimização. **Exemplos de Assertions (Promptfoo):** O Promptfoo suporta mais de 20 tipos de assertions, incluindo `is-json` (para garantir a saída JSON), `regex` (para validação de formato) e `llm-rubric` (para avaliação assistida por modelo). **Tendência:** A tendência de 2024-2025 é a migração de avaliações puramente humanas para **Model-Graded Evaluation (MGE)**, onde um LLM atua como avaliador, reduzindo custos e acelerando a iteração do prompt. [1] [2]

## Features

**Métricas Determinísticas:** Validação de formato (JSON, XML, SQL), correspondência exata, expressões regulares, verificação de conteúdo (contém/não contém), e validação de chamadas de função/ferramentas. **Métricas Assistidas por LLM:** Uso de um LLM avaliador para julgar a qualidade da resposta com base em rubricas (e.g., G-Eval, Pi Scorer), avaliando aspectos como relevância, coerência, tom e fidelidade ao contexto. **Frameworks:** Utilização de ferramentas como Promptfoo e PromptLayer para testes em lote, comparação de modelos, rastreamento de custos e latência, e criação de métricas derivadas (como F1-Score). **Avaliação Humana:** Uso de escalas de classificação (Likert) ou sistemas binários (Pass/Fail) para julgamento qualitativo.

## Use Cases

**Testes de Regressão:** Garantir que novos prompts ou modelos mantenham a qualidade das saídas esperadas. **Validação de Formato:** Assegurar que as saídas do LLM estejam sempre em um formato específico (e.g., JSON para APIs, SQL para consultas a banco de dados). **Detecção de Alucinações e Segurança:** Utilizar prompts de avaliação para verificar a fidelidade ao contexto (Context Faithfulness) e a ausência de conteúdo prejudicial (Guardrails). **Otimização de Custos e Latência:** Comparar o desempenho de diferentes modelos ou prompts em termos de custo e velocidade de resposta para selecionar a opção mais eficiente para produção. **Desenvolvimento de RAG (Retrieval-Augmented Generation):** Avaliar a relevância e a recuperação correta de documentos pelo sistema RAG.

## Integration

**Exemplo de Prompt de Avaliação (Model-Graded Rubric):**

```
Você é um avaliador de LLM. Sua tarefa é julgar a resposta fornecida a um prompt, com base na rubrica abaixo.

**Prompt Original:** [Insira o prompt do usuário aqui]
**Resposta do LLM:** [Insira a resposta do LLM aqui]

**Rubrica de Avaliação:**
1. **Relevância (0-5):** A resposta aborda diretamente o tópico do prompt?
2. **Correção Factual (0-5):** A resposta contém informações factualmente precisas?
3. **Tom (0-5):** O tom da resposta é profissional e adequado ao contexto?

**Instrução:** Forneça sua avaliação em formato JSON, incluindo uma pontuação para cada critério e um breve comentário.
```

**Melhores Práticas:**
1. **Definir Critérios Claros:** Estabelecer métricas de sucesso acionáveis (e.g., clareza, precisão, ausência de alucinações).
2. **Usar Testes de Regressão:** Manter um conjunto de testes de avaliação (regression set) para garantir que as atualizações do modelo ou do prompt não degradem o desempenho.
3. **Automatizar com Frameworks:** Utilizar ferramentas como Promptfoo para automatizar testes e comparações em escala, integrando assertions determinísticas e model-graded evals.
4. **Priorizar Pass/Fail:** Para avaliação humana, um sistema binário (Passa/Falha) é frequentemente mais claro e menos propenso a subjetividade do que escalas numéricas detalhadas.

## URL

https://www.promptfoo.dev/docs/configuration/expected-outputs/