# Engenharia de Prompt para Conformidade e Regulamentação (Compliance & Regulatory Prompts)

## Description

A Engenharia de Prompt para Conformidade e Regulamentação é uma técnica especializada que utiliza Modelos de Linguagem de Grande Escala (LLMs) com instruções altamente estruturadas para automatizar e aprimorar tarefas críticas de Governança, Risco e Conformidade (GRC). O foco principal é garantir que as saídas dos LLMs sejam precisas, transparentes e aderentes aos requisitos legais e regulatórios. Isso inclui a geração de documentação de modelos de risco, a realização de avaliações de risco, a criação de cenários de estresse e a identificação de viés em modelos de Inteligência Artificial (IA) [1] [2]. A aplicação eficaz da engenharia de prompt neste domínio é crucial para mitigar riscos regulatórios, reduzir a carga de trabalho manual e demonstrar um compromisso robusto com a conformidade, conforme exigido por órgãos reguladores como o Departamento de Justiça dos EUA (DOJ) [3].

## Statistics

- **Aumento de Desempenho:** Em um estudo de caso, o desempenho em tarefas de conformidade aumentou de 80% para 95%–100% após a aplicação de técnicas de engenharia de prompt, RAG (Retrieval-Augmented Generation) e *fine-tuning* [4].
- **Foco na Consistência:** A eficácia é medida pela consistência e confiabilidade da saída do LLM, sendo a engenharia de prompt uma ferramenta crítica para testar a conformidade dos modelos com diretrizes específicas [5].
- **Métricas de Avaliação:** As métricas de avaliação de LLMs em conformidade incluem precisão, redução de alucinações, transparência e detecção de viés, sendo o *prompt design* essencial para otimizar esses resultados [6].
- **Tendência de Mercado:** A engenharia de prompt é considerada uma habilidade essencial para gerentes de risco de modelo (MRM) e profissionais de GRC, com o mercado buscando soluções que integrem LLMs para automação de conformidade [1].

## Features

- **Automação de Documentação:** Geração de resumos e relatórios técnicos a partir de código-fonte ou documentos complexos [1].
- **Avaliação de Risco e Cenários:** Criação de cenários hipotéticos e plausíveis para testes de estresse e avaliação de risco de modelos [1].
- **Revisão de Código de Validação:** Análise de código (Python, R, SQL) para identificar erros lógicos, *overfitting* ou *data leakage* [1].
- **Testes de Viés e Justiça (Fairness):** Identificação de preocupações éticas e regulatórias em modelos de IA e sugestão de mitigações [1].
- **Monitoramento Regulatório:** Sumarização de atualizações regulatórias recentes e identificação de riscos emergentes na indústria [3].
- **Geração de Conteúdo de Treinamento:** Criação de exemplos de treinamento baseados em cenários para dilemas éticos e conformidade [3].

## Use Cases

- **Model Risk Management (MRM):** Auxílio na supervisão e validação de modelos de IA, garantindo que estejam em conformidade com regulamentações como SR 11-7 [1].
- **Auditoria e Due Diligence:** Geração de checklists e procedimentos para auditorias internas e *due diligence* de terceiros, especialmente em relação a riscos de suborno e corrupção (FCPA, UK Bribery Act) [3].
- **Gerenciamento de Políticas:** Criação e atualização de políticas internas, como políticas de *whistleblower* (denúncia), alinhadas com as exigências legais [3].
- **Análise de Causa Raiz (RCA):** Estruturação de processos de RCA para falhas de conformidade, demonstrando um compromisso com a melhoria contínua aos reguladores [3].
- **Processamento de Documentos Regulatórios:** Extração, sumarização e análise de grandes volumes de documentos legais e regulatórios para identificar requisitos de conformidade [2].

## Integration

A integração de prompts de conformidade requer a definição clara do papel do LLM (por exemplo, "Você é um analista de risco sênior") e a inclusão de contexto regulatório específico.

**Exemplos de Prompts e Melhores Práticas:**

| Categoria de Tarefa | Exemplo de Prompt (Português) | Melhor Prática (Inglês) |
| :--- | :--- | :--- |
| **Documentação de Modelo** | "Resuma este script Python que implementa um modelo de *gradient boosting*, incluindo suas características de entrada, etapas de pré-processamento e métricas de avaliação do modelo. Explique-o em linguagem simples, adequada para um relatório de validação de modelo." [1] | **Role-Playing & Context:** Assign the LLM a persona (e.g., "Senior Risk Analyst") and provide the full context (code, document, or regulation) [1]. |
| **Risco Emergente** | "Identifique riscos de conformidade emergentes em nosso setor (ex: serviços financeiros) relacionados à Lei de IA da UE (EU AI Act)." [3] | **Specificity:** Specify the industry, the regulation, and the type of risk (e.g., operational, legal, reputational) [3]. |
| **Teste de Estresse** | "Gere cinco cenários de crise econômica que poderiam afetar um modelo de risco de inadimplência de hipotecas. Inclua mudanças no desemprego, taxas de juros e preços de imóveis." [1] | **Constraint-Based Generation:** Use constraints (e.g., "five scenarios," "must include X, Y, and Z variables") to ensure relevance and plausibilidade [1]. |
| **Viés e Justiça** | "Dado um modelo de pontuação de crédito que usa código postal, renda e tipo de emprego, quais preocupações de justiça podem surgir? Sugira recursos alternativos para reduzir o potencial viés." [1] | **Ethical Guardrails:** Explicitly instruct the LLM to analyze the output against ethical and regulatory fairness standards (e.g., disparate impact) [1]. |
| **Relatório Regulatório** | "Elabore um modelo de relatório de conformidade para regulamentações ambientais (ex: ESG) aplicáveis a uma empresa de manufatura no Brasil." [2] | **Template Generation:** Request a structured output (e.g., "Draft a checklist," "Generate a sample template") to ensure a usable format [2]. |

**Melhores Práticas Adicionais:**
*   **RAG (Retrieval-Augmented Generation):** Utilize RAG para fornecer ao LLM documentos regulatórios internos ou externos atualizados, garantindo que as respostas sejam baseadas em fontes autorizadas [4].
*   **Prompt Chain:** Use uma sequência de prompts (encadeamento) para decompor tarefas complexas de conformidade (ex: primeiro, resumir a lei; segundo, aplicar a lei a um caso de uso) [4].
*   **Validação Humana:** Sempre use a saída do LLM como um rascunho ou *insight*, e não como a decisão final. A validação humana por um especialista em conformidade é obrigatória [1].

## URL

https://empoweredsystems.com/blog/prompt-engineering-for-model-risk-managers-a-powerful-ally-for-ai-model-oversight/