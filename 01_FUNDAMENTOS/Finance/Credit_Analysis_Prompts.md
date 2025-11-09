# Credit Analysis Prompts

## Description
**Credit Analysis Prompts** (Prompts de Análise de Crédito) são instruções estruturadas e detalhadas, frequentemente incorporando técnicas avançadas de Engenharia de Prompt como o **Labeled Guide Prompting (LGP)** [3], utilizadas para guiar Large Language Models (LLMs) na execução de tarefas complexas de avaliação de risco de crédito. O objetivo central é converter dados financeiros estruturados (como balanços, demonstrações de resultados e scores de crédito) em uma descrição em linguagem natural que o LLM possa processar, permitindo-lhe realizar desde a classificação binária de risco (Alto/Baixo) [1] até a geração de relatórios de risco completos, justificados e interpretáveis [3]. Esta abordagem foca em aproveitar as capacidades de raciocínio e geração de texto dos LLMs para aumentar a transparência e a confiabilidade do processo de decisão de crédito, alinhando-o com os requisitos regulatórios e de auditoria [2].

## Examples
```
### Exemplo 1: Classificação de Risco com Raciocínio Explícito (CoT)
**Role:** Você é um analista de risco de crédito sênior.
**Instrução:** Analise os dados do requerente e determine se o risco de inadimplência é **ALTO** ou **BAIXO**.
**Dados do Requerente:**
- Score de Crédito (FICO): 680
- Renda Anual: R$ 120.000
- Dívida Total/Renda (DTI): 35%
- Histórico de Pagamentos Atrasados (últimos 2 anos): 2
**Processo de Raciocínio (Passo a Passo):**
1. Avalie o Score de Crédito (680).
2. Avalie o DTI (35%).
3. Avalie o Histórico de Pagamentos.
4. Conclua o risco final e justifique.
**Saída Requerida:** Apenas a palavra **ALTO** ou **BAIXO**.

### Exemplo 2: Geração de Relatório de Risco Estruturado (LGP)
**Role:** Você é um especialista em análise de risco de crédito corporativo.
**Instrução:** Gere um relatório de risco detalhado para a empresa "TechCorp S.A.", utilizando a técnica Labeled Guide Prompting para garantir a completude.
**Dados da Empresa:**
- Receita Líquida (2024): R$ 50M
- Lucro Líquido (2024): R$ 5M
- Índice de Liquidez Corrente: 1.2
- Setor: Tecnologia (Alto Crescimento, Alta Volatilidade)
**Itens a Serem Abordados (LGP):**
- **[ANÁLISE_QUANTITATIVA]:** Avaliação dos indicadores financeiros e score de crédito.
- **[ANÁLISE_QUALITATIVA]:** Avaliação do setor, gestão e cenário macroeconômico.
- **[RISCO_FINAL]:** Classificação de risco (A, B, C, D) e limite de crédito sugerido.
- **[JUSTIFICATIVA_REGULATÓRIA]:** Explicação de como a decisão se alinha com a Circular X do Banco Central.

### Exemplo 3: Interpretabilidade (XAI) e Fatores de Influência
**Role:** Você é um modelo de IA de risco de crédito focado em interpretabilidade (XAI).
**Instrução:** Com base na decisão de risco **ALTO** para o requerente, identifique e descreva os 3 fatores que mais contribuíram para essa classificação, como se fossem "SHAP values" em linguagem natural.
**Decisão de Risco:** ALTO
**Dados do Requerente:** [Inserir dados completos]
**Saída Requerida:**
1. **Fator Principal:** [Descrição do fator e seu impacto]
2. **Segundo Fator:** [Descrição do fator e seu impacto]
3. **Terceiro Fator:** [Descrição do fator e seu impacto]

### Exemplo 4: Análise de Documentos Não Estruturados
**Role:** Você é um extrator de dados de risco de crédito.
**Instrução:** Leia o trecho do contrato social e extraia as cláusulas que representam um risco potencial para a concessão de crédito.
**Trecho do Contrato Social:** [Inserir trecho]
**Saída Requerida (JSON):**
```json
{
  "clausulas_de_risco": [
    {"clausula": "Cláusula 4.1", "risco_associado": "Restrição de Venda de Ativos"},
    {"clausula": "Cláusula 7.3", "risco_associado": "Subordinação de Dívida"}
  ]
}
```

### Exemplo 5: Simulação de Cenários (Stress Test)
**Role:** Você é um modelador de risco de crédito.
**Instrução:** Simule o impacto de um aumento de 5 pontos percentuais na taxa de juros (Cenário de Estresse) no DTI e na probabilidade de inadimplência do requerente.
**Dados do Requerente:** [Inserir dados completos, incluindo valor do empréstimo e taxa atual]
**Saída Requerida:**
- **DTI no Cenário Base:** [Valor]%
- **DTI no Cenário de Estresse:** [Valor]%
- **Probabilidade de Inadimplência (Estresse):** [Valor]%
- **Recomendação:** [Manter/Revisar/Negar]
```

## Best Practices
| Prática | Descrição | Fonte |
| :--- | :--- | :--- |
| **Labeled Guide Prompting (LGP)** | Decompor a tarefa em sub-tarefas rotuladas (ex: `[ANÁLISE_QUANTITATIVA]`) para garantir que o LLM aborde todas as dimensões do problema (o quê, porquê, como), promovendo raciocínio abdutivo e completude [3]. | [3] |
| **Raciocínio em Cadeia (CoT)** | Exigir uma análise passo a passo e justificativas humanas para as decisões de risco. Isso aumenta a transparência, facilita a auditoria e reduz a taxa de erro em consultas complexas em cerca de 20% [2]. | [1], [2] |
| **Controle de Saída Estrito** | Especificar um formato de saída estrito (JSON, XML, ou texto rotulado) para facilitar a integração automatizada com sistemas de crédito e garantir o controle de qualidade. | [3] |
| **Contexto Regulatório** | Incluir no prompt a necessidade de aderência a frameworks regulatórios (ex: Basileia III) para garantir a conformidade da análise e a precisão do modelo [2]. | [2] |
| **Few-Shot Learning** | Fornecer exemplos anotados de análises de crédito bem-sucedidas e não-sucedidas para refinar o comportamento do modelo e aumentar a precisão. | [3] |

## Referências
[1] Chen, Q. (2025). Explore the Use of Prompt-Based LLM for Credit Risk Classification. *Journal of Computer and Communications*, 13, 33-46.
[2] Joshi, S. (2025). Leveraging Prompt Engineering to Enhance Financial Market Integrity and Risk Management. *World Journal of Advanced Research and Reviews*, 25(01), 1775-1785.
[3] Teixeira, A. C., et al. (2023). Enhancing Credit Risk Reports Generation using LLMs: An Integration of Bayesian Networks and Labeled Guide Prompting. *4th ACM International Conference on AI in Finance*.

## Use Cases
1. **Classificação Automatizada de Risco:** Determinar rapidamente a probabilidade de inadimplência (default) de um tomador (pessoa física ou jurídica) para triagem inicial [1].
2. **Geração de Relatórios de Risco:** Criar relatórios detalhados e justificados para analistas e comitês de crédito, com maior confiabilidade e aceitação por analistas humanos [3].
3. **Interpretabilidade (XAI):** Gerar explicações claras sobre os fatores que mais influenciam a decisão de risco, convertendo métricas complexas (como SHAP values) em linguagem natural.
4. **Análise de Dados Não Estruturados:** Processar documentos como demonstrações financeiras, contratos, notícias e e-mails para extrair features de risco relevantes.
5. **Simulação de Estresse (Stress Testing):** Avaliar o impacto de cenários macroeconômicos adversos (aumento de juros, recessão) na carteira de crédito.

## Pitfalls
1. **Alucinações Financeiras:** O risco de o LLM gerar dados ou análises factualmente incorretas é alto e crítico no setor financeiro, exigindo validação rigorosa da saída.
2. **Viés e Injustiça:** O modelo pode perpetuar ou amplificar vieses presentes nos dados de treinamento, levando a decisões de crédito discriminatórias ou injustas.
3. **Falta de Interpretabilidade (Black Box):** Sem a exigência de raciocínio explícito (CoT), a decisão do LLM pode ser opaca, dificultando a auditoria e a aceitação regulatória.
4. **Complexidade do Prompt:** Prompts excessivamente longos ou complexos podem confundir o modelo ou exceder o limite de tokens, resultando em saídas truncadas ou irrelevantes.
5. **Desalinhamento Regulatório:** A falha em incorporar o contexto regulatório no prompt pode levar a análises não conformes, expondo a instituição a riscos legais.

## URL
[https://dl.acm.org/doi/fullHtml/10.1145/3604237.3626902](https://dl.acm.org/doi/fullHtml/10.1145/3604237.3626902)
