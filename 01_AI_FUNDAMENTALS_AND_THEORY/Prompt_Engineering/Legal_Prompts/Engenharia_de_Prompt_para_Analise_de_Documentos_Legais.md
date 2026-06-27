# Engenharia de Prompt para Análise de Documentos Legais

## Description

A Engenharia de Prompt para Análise de Documentos Legais é uma técnica especializada que utiliza modelos de linguagem grande (LLMs) para otimizar tarefas jurídicas complexas, como revisão de contratos, pesquisa de jurisprudência, análise de risco e sumários de conformidade. A técnica se baseia em estruturar prompts de forma precisa, atribuindo um **papel específico** ao LLM (ex: "Advogado Sênior em Fusões e Aquisições"), fornecendo **contexto detalhado** e exigindo um **formato de resposta estruturado** (ex: formato IRAC - Issue, Rule, Application, Conclusion). O objetivo é mitigar riscos de alucinação e garantir a precisão e a relevância das saídas em um domínio onde a exatidão é crítica. A técnica evoluiu para o uso de "Super-Prompts" que encapsulam todas as instruções, contexto e requisitos de formato em uma única solicitação abrangente.

## Statistics

A adoção de IA no setor jurídico está em rápido crescimento, com **80%** dos profissionais esperando um impacto transformacional em 5 anos [2]. **54%** dos profissionais jurídicos já utilizam IA para redação de correspondência [3]. As métricas de desempenho para LLMs no domínio legal focam em **Precisão (Accuracy)** na extração de informações (F1-Score), **Latência** (tempo de resposta) e, crucialmente, a taxa de **Alucinação** (geração de informações factualmente incorretas) [4]. Estudos de caso demonstram que o uso de prompts estruturados (como o Super-Prompt) e o RAG (Retrieval-Augmented Generation) são essenciais para manter a precisão e a rastreabilidade das fontes em tarefas de pesquisa e análise jurídica [5].

## Features

Atribuição de Papel (Role-Playing) para especialização do LLM; Estrutura de Raciocínio IRAC (Issue, Rule, Application, Conclusion) para rastreabilidade da lógica; Framework de Super-Prompt para solicitações complexas e multifacetadas; Exigência de Citações e Referências para validação; Instruções de Formato de Saída Detalhadas (sumário executivo, cabeçalhos, negrito) para clareza.

## Use Cases

Revisão e extração de cláusulas contratuais (ex: LoL, Indenização, Rescisão); Sumarização de regulamentações complexas (ex: NY DFS 500, LGPD); Análise de risco em documentos de terceiros (ex: DPAs de fornecedores); Pesquisa jurídica e identificação de precedentes relevantes; Elaboração de painéis de KPI (Key Performance Indicator) para operações jurídicas internas; Classificação e categorização de documentos jurídicos.

## Integration

**Framework de Super-Prompt (Exemplo de Integração):**

1.  **Papel e Contexto:** "Você é um advogado de conformidade sênior especializado em regulamentações de privacidade de dados (LGPD/GDPR)."
2.  **Tarefa:** "Analise o [DOCUMENTO ANEXADO] e extraia todas as cláusulas relacionadas à transferência internacional de dados e às obrigações de notificação de violação."
3.  **Requisitos de Raciocínio:** "Para cada cláusula, forneça uma análise de risco (Baixo, Médio, Alto) e cite o número da seção exata. Se não houver cláusula, declare explicitamente."
4.  **Formato de Saída:** "Comece com um sumário executivo de 3 pontos. Use o formato IRAC para a análise de risco de cada cláusula. Use negrito para as palavras-chave."

**Exemplo de Prompt para Análise de Cláusula:**

```
[PAPEL]: Você é um advogado interno com 10 anos de experiência em contratos de SaaS.
[DOCUMENTO]: [Insira o texto da Cláusula 7 - Limitação de Responsabilidade]
[TAREFA]: Analise a cláusula de Limitação de Responsabilidade (LoL) para determinar se ela é favorável ao cliente ou ao fornecedor.
[REQUISITOS]:
1. Identifique o teto de responsabilidade (ex: 12 meses de taxas).
2. Liste as exclusões de responsabilidade (ex: danos indiretos, violação de dados).
3. Forneça uma recomendação de negociação para torná-la mais favorável ao cliente.
[FORMATO]: Responda em formato de tabela com as colunas: Aspecto, Análise, Recomendação.
```

## URL

https://www.lsuite.co/blog/mastering-ai-legal-prompts