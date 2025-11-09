# Oncology Prompts

## Description
A engenharia de prompts em oncologia refere-se à arte e ciência de formular entradas (prompts) eficazes para **Large Language Models (LLMs)**, com o objetivo de otimizar seu desempenho em tarefas clínicas, de pesquisa e administrativas relacionadas ao câncer. Este campo é crucial para a adoção segura e eficaz da Inteligência Artificial (IA) na prática oncológica, conforme destacado pela **ESMO Guidance on the Use of Large Language Models in Clinical Practice (ELCAP)** [1].

Os LLMs são empregados para aprimorar, e não substituir, os fluxos de trabalho clínicos e a tomada de decisões, atuando como ferramentas de assistência. A eficácia dos "Oncology Prompts" depende da clareza da instrução, do contexto fornecido (dados do paciente, diretrizes clínicas) e da técnica de engenharia de prompt utilizada (e.g., Few-shot, RAG, CoT) [2].

## Examples
```
### Exemplo 1: Suporte à Decisão de Tratamento Adjuvante (CoT + RAG)

**Objetivo:** Sugerir o tratamento adjuvante para Câncer de Mama, seguindo a diretriz ESMO fornecida.

**Prompt:**
\`\`\`
Você é um assistente de IA especializado em oncologia, com foco em suporte à decisão clínica. Sua tarefa é analisar o perfil da paciente e a diretriz clínica fornecida para sugerir o tratamento adjuvante mais apropriado, utilizando o raciocínio passo a passo (Chain-of-Thought).

**DIRETRIZ CLÍNICA (RAG Contexto):**
[INSERIR AQUI TRECHO DA DIRETRIZ ESMO SOBRE CÂNCER DE MAMA INICIAL, POR EXEMPLO: "Para pacientes com câncer de mama inicial, RH+/HER2-, com 1-3 linfonodos positivos e escore Oncotype DX de 26 ou superior, a quimioterapia seguida de terapia endócrina é recomendada."]

**DADOS DA PACIENTE:**
- Idade: 55 anos
- Diagnóstico: Carcinoma ductal invasivo, estágio IIA
- Status do Receptor: RH+/HER2-
- Linfonodos: 2/15 positivos
- Escore de Risco: Oncotype DX = 31

**INSTRUÇÕES:**
1.  **Análise (CoT):** Compare os DADOS DA PACIENTE com a DIRETRIZ CLÍNICA. Identifique os critérios que se alinham e os que não se alinham.
2.  **Conclusão:** Com base na análise, qual é a recomendação de tratamento adjuvante?
3.  **Justificativa:** Cite a parte específica da DIRETRIZ CLÍNICA que suporta sua recomendação.
\`\`\`

### Exemplo 2: Extração Estruturada de Relatório de Patologia (Few-shot)

**Objetivo:** Extrair informações-chave de um relatório de patologia não estruturado para preenchimento de um Registro Médico Eletrônico (EMR).

**Prompt:**
\`\`\`
Você é um extrator de dados clínicos. Sua saída DEVE ser um objeto JSON.

**EXEMPLO DE FEW-SHOT (Entrada):**
"O espécime de biópsia de próstata revelou adenocarcinoma acinar. Gleason 4+3=7. Margens negativas. Invasão perineural presente. Estadiamento pT2c."

**EXEMPLO DE FEW-SHOT (Saída):**
\`\`\`json
{
  "Diagnóstico Primário": "Adenocarcinoma acinar da próstata",
  "Escore de Gleason": "7 (4+3)",
  "Status da Margem": "Negativa",
  "Invasão Perineural": "Presente",
  "Estadiamento Patológico": "pT2c"
}
\`\`\`

**RELATÓRIO DE PATOLOGIA (Entrada):**
"O exame histopatológico da peça cirúrgica de pulmão confirma a presença de Adenocarcinoma. O tumor mede 3,5 cm. Margens cirúrgicas livres (distância de 1,2 cm). Não foi observada invasão vascular ou linfática. Estadiamento patológico T2a N0 M0."

**INSTRUÇÕES:**
Extraia as informações do RELATÓRIO DE PATOLOGIA e retorne o resultado no formato JSON, seguindo a estrutura do EXEMPLO DE FEW-SHOT.
\`\`\`

### Exemplo 3: Comunicação com o Paciente (Zero-shot + Persona)

**Objetivo:** Explicar o que é "imunoterapia" para um paciente recém-diagnosticado, usando linguagem simples e empática.

**Prompt:**
\`\`\`
**PERSONA:** Você é um enfermeiro oncológico com 10 anos de experiência, conhecido por sua capacidade de explicar conceitos médicos complexos de forma clara e tranquilizadora.

**PÚBLICO:** Paciente de 68 anos, recém-diagnosticado com melanoma, com escolaridade de nível médio.

**INSTRUÇÃO:** Explique o que é a **imunoterapia** para o câncer. Use analogias simples (zero-shot) e evite jargões médicos. O tom deve ser de apoio e informativo.
\`\`\`

### Exemplo 4: Triagem de Ensaios Clínicos (RAG)

**Objetivo:** Determinar se um paciente é elegível para um ensaio clínico específico.

**Prompt:**
\`\`\`
Você é um especialista em elegibilidade de ensaios clínicos. Sua tarefa é comparar os dados do paciente com os critérios de inclusão e exclusão do Ensaio Clínico fornecido.

**CRITÉRIOS DO ENSAIO CLÍNICO (RAG Contexto):**
- **Inclusão:** Pacientes com Carcinoma de Células Renais (CCR) metastático, que receberam no máximo uma linha de terapia anterior. Idade ≥ 18 anos. ECOG Performance Status de 0 ou 1.
- **Exclusão:** Histórico de metástases cerebrais sintomáticas. Doença autoimune ativa.

**DADOS DO PACIENTE:**
- Diagnóstico: CCR metastático
- Terapias Anteriores: Sunitinibe (1 linha)
- Idade: 62 anos
- ECOG Status: 1
- Histórico: Metástases cerebrais assintomáticas e tratadas há 3 anos.

**INSTRUÇÕES:**
1.  Liste os critérios de Inclusão e Exclusão.
2.  Para cada critério, indique se o paciente atende ("SIM" ou "NÃO").
3.  **Conclusão:** O paciente é elegível para o ensaio? Justifique a resposta.
\`\`\`

### Exemplo 5: Geração de Resumo de Consulta (Structured Output)

**Objetivo:** Gerar um resumo estruturado da consulta para o prontuário.

**Prompt:**
\`\`\`
**FUNÇÃO:** Gerar um resumo de consulta estruturado.

**ENTRADA DE ÁUDIO (Transcrição):**
"O paciente, Sr. João, 72 anos, veio para acompanhamento. Ele está no ciclo 4 de quimioterapia para câncer de cólon. Relata náuseas leves, controladas com ondansetrona. O exame físico está estável. Solicitamos novos exames de CEA e TC de abdômen e pelve para reavaliação. Próxima consulta em 3 semanas."

**INSTRUÇÕES:**
Gere um resumo no formato Markdown, com as seguintes seções:
1.  **Dados do Paciente:** (Nome, Idade, Diagnóstico)
2.  **Status do Tratamento:** (Ciclo atual, Regime)
3.  **Sintomas/Toxicidade:** (Descrição e manejo)
4.  **Plano:** (Exames solicitados, Próxima consulta)
\`\`\`

### Exemplo 6: Análise de Risco de Viés em Pesquisa (Prompt Permissivo)

**Objetivo:** Triar artigos de pesquisa para um estudo de revisão sistemática, aplicando um critério de exclusão "soft".

**Prompt:**
\`\`\`
Você é um revisor de literatura para uma revisão sistemática sobre LLMs em oncologia.

**CRITÉRIO DE EXCLUSÃO PERMISSIVO:** Exclua o artigo SOMENTE se ele não mencionar explicitamente o uso de Large Language Models (LLMs) ou modelos de linguagem em seu resumo.

**RESUMO DO ARTIGO (Entrada):**
"Avaliamos a eficácia de um novo algoritmo de aprendizado de máquina para prever a resposta à radioterapia em pacientes com câncer de pulmão. O modelo foi treinado em dados de imagem e texto de prontuários eletrônicos. Os resultados mostram alta precisão na estratificação de risco."

**INSTRUÇÕES:**
1.  **Análise:** O resumo menciona explicitamente LLMs ou modelos de linguagem?
2.  **Decisão:** O artigo deve ser incluído ou excluído? Justifique a decisão com base no CRITÉRIO DE EXCLUSÃO PERMISSIVO.
\`\`\`
```

## Best Practices
As melhores práticas para "Oncology Prompts" são fortemente guiadas pelos princípios de segurança, transparência e responsabilidade, conforme estabelecido pelo ELCAP [1].

| Princípio | Descrição | Técnica de Prompt Engineering Relacionada |
| :--- | :--- | :--- |
| **Responsabilidade Humana Explícita** | Para ferramentas voltadas para profissionais de saúde (Tipo 2), a responsabilidade final pelas decisões clínicas deve permanecer com o oncologista. O prompt deve solicitar a fonte e a justificativa para a saída. | **Chain-of-Thought (CoT):** Solicitar ao LLM que apresente o raciocínio passo a passo antes da conclusão final. |
| **Fundamentação em Dados (RAG)** | Integrar a recuperação de informações de fontes confiáveis (diretrizes clínicas, EHRs, literatura validada) ao prompt. Isso mitiga alucinações e garante que a resposta seja baseada em evidências. | **Retrieval-Augmented Generation (RAG):** Inserir trechos de documentos clínicos relevantes no prompt antes da pergunta. |
| **Validação e Transparência** | Para tarefas de suporte à decisão, o prompt deve ser formalmente validado e suas limitações transparentes. Usar prompts **Few-shot** com exemplos de casos clínicos conhecidos e validados. | **Few-shot Prompting:** Fornecer exemplos de pares de entrada/saída para tarefas específicas (e.g., classificação de estágio, sugestão de tratamento). |
| **Proteção de Dados** | Em aplicações voltadas para o paciente (Tipo 1) e sistemas institucionais (Tipo 3), o prompt deve ser formulado para proteger a privacidade, evitando a inclusão de Informações de Saúde Protegidas (PHI) desnecessárias. | **Prompt de Filtragem/Anonimização:** Instruir o LLM a anonimizar ou resumir dados sensíveis antes de processá-los. |
| **Monitoramento Contínuo** | Em sistemas institucionais, os prompts devem ser projetados para facilitar o monitoramento contínuo de viés e desempenho, garantindo que o sistema seja revalidado quando os processos ou fontes de dados mudarem. | **Prompt Estruturado:** Usar um formato de prompt rígido (e.g., JSON, XML) para garantir entradas consistentes e saídas facilmente auditáveis. |

## Use Cases
O uso de "Oncology Prompts" abrange diversas áreas da oncologia [1] [2]:

1.  **Suporte à Decisão Clínica:** Geração de sugestões de tratamento adjuvante ou neoadjuvante com base em diretrizes e dados do paciente (e.g., câncer de mama inicial).
2.  **Extração e Resumo de Dados:** Extração de informações de câncer de textos não estruturados (notas clínicas, relatórios patológicos) para preencher Registros Médicos Eletrônicos (EMRs) ou para pesquisa.
3.  **Comunicação e Educação do Paciente:** Criação de chatbots para responder a perguntas de pacientes sobre sua condição, tratamento e suporte a sintomas, operando sob supervisão clínica.
4.  **Triagem de Literatura e Pesquisa:** Uso de prompts para triar artigos biomédicos, identificar estudos relevantes ou avaliar a sinergia de linhagens celulares em pesquisa de medicamentos.
5.  **Correspondência de Ensaios Clínicos:** Automatização da correspondência de perfis de pacientes com critérios de elegibilidade de ensaios clínicos.
6.  **Sistemas de Question-Answering (QA):** Melhorar a usabilidade de diretrizes clínicas complexas, transformando-as em sistemas de QA acionáveis.

## Pitfalls
A adoção de LLMs em oncologia apresenta riscos significativos que a engenharia de prompts deve mitigar [1]:

*   **Alucinações e Imprecisão:** O risco de o LLM gerar informações clinicamente incorretas ou inventadas é alto, especialmente se o prompt for vago ou se basear em dados de entrada incompletos.
*   **Viés e Iniquidade:** O LLM pode perpetuar ou amplificar vieses presentes nos dados de treinamento, levando a disparidades no cuidado ou sugestões inadequadas para subpopulações de pacientes.
*   **Falta de Transparência:** Se o prompt não exigir o raciocínio (CoT) ou a fonte (RAG), a saída do LLM pode ser uma "caixa preta", dificultando a auditoria e a responsabilidade clínica.
*   **Confiança Excessiva:** Profissionais de saúde podem confiar excessivamente nas sugestões do LLM, negligenciando a responsabilidade humana explícita e o julgamento clínico.
*   **Violação de Privacidade:** Prompts mal formulados podem inadvertidamente expor Informações de Saúde Protegidas (PHI) ou levar a vazamentos de dados em sistemas institucionais.

## URL
[https://www.esmo.org/society-updates/esmo-publishes-first-guidance-on-the-safe-use-of-large-language-models-in-oncology-practice](https://www.esmo.org/society-updates/esmo-publishes-first-guidance-on-the-safe-use-of-large-language-models-in-oncology-practice)
