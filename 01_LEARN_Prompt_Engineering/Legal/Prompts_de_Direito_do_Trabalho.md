# Prompts de Direito do Trabalho

## Description
Prompts de Direito do Trabalho são instruções especializadas e estruturadas, projetadas para serem usadas com modelos de Linguagem Grande (LLMs) e outras ferramentas de IA generativa, com o objetivo de auxiliar profissionais de Recursos Humanos (RH), advogados internos e consultores jurídicos em questões relacionadas à legislação trabalhista. Esses prompts são formulados para gerar rascunhos de documentos legais, analisar riscos de conformidade, resumir jurisprudência, comparar classificações de trabalhadores (empregado vs. contratado independente) e criar políticas internas, como acordos de confidencialidade e políticas de férias remuneradas. A eficácia desses prompts reside na sua capacidade de fornecer contexto legal específico (jurisdição, tipo de lei) e definir a persona e o formato de saída desejados, transformando a IA em um "estagiário jurídico" altamente eficiente para tarefas de primeira linha.

## Examples
```
1.  **Análise de Risco de Demissão:**
    `Você é um advogado trabalhista com 10 anos de experiência. Analise o seguinte cenário de demissão por justa causa [INSERIR DETALHES DO CASO: histórico de desempenho, advertências, motivo da demissão]. Liste as 5 principais considerações legais e os riscos de litígio sob a lei do estado de [ESTADO/JURISDIÇÃO]. Apresente a resposta em formato de tabela com colunas para 'Risco', 'Base Legal' e 'Ação Recomendada'.`
2.  **Comparação de Classificação de Trabalhadores:**
    `Explique as principais diferenças legais entre um 'empregado' e um 'contratado independente' sob a lei federal dos EUA (e a lei de [ESTADO]). Crie uma lista de verificação de 10 pontos para o RH usar na determinação da classificação correta, focando nos critérios de controle e independência. O tom deve ser informativo e didático.`
3.  **Rascunho de Política de Férias Remuneradas (PTO):**
    `Rascunhe uma política de folgas remuneradas (PTO) para uma força de trabalho remota baseada nos EUA, considerando os requisitos mínimos de acumulação e uso de PTO nos estados de Califórnia, Nova York e Texas. Inclua cláusulas sobre o pagamento de PTO não utilizado no término do contrato, conforme exigido por lei. O formato deve ser um rascunho de documento de política interna.`
4.  **Rascunho de Acordo de Confidencialidade:**
    `Crie um rascunho de acordo básico de confidencialidade de funcionário (NDA) para uso nos EUA. O acordo deve incluir cláusulas de não divulgação, devolução de informações da empresa no término do contrato e obrigações pós-término. O tom deve ser formal e juridicamente preciso.`
5.  **Guia de Conformidade para Contratação/Demissão:**
    `Gere um guia de conformidade curto para a equipe de RH ao contratar e demitir funcionários no estado de [ESTADO/JURISDIÇÃO]. O guia deve cobrir a documentação legal necessária, considerações sobre salários e horas, e práticas para evitar discriminação. O formato deve ser uma lista de verificação (checklist) numerada.`
6.  **Análise de Questões de Trabalho Remoto:**
    `Escreva um resumo das principais questões legais a serem consideradas antes de implementar uma política de trabalho híbrido ou remoto em vários estados ([ESTADO 1], [ESTADO 2]). Concentre-se em impostos estaduais, leis de compensação de trabalhadores e requisitos de registro de horas para funcionários não isentos. O formato deve ser um memorando interno conciso.`
7.  **Linguagem de Carta de Rescisão:**
    `Gere linguagem de modelo para uma carta de rescisão de contrato de trabalho (sem justa causa) que evite criar promessas implícitas ou admissões de responsabilidade. Inclua uma seção sobre a finalização dos benefícios e a elegibilidade para COBRA. O tom deve ser neutro e factual.`
```

## Best Practices
*   **Defina a Persona e o Público:** Comece o prompt definindo a função da IA (ex: "Você é um advogado trabalhista experiente") e o público-alvo da saída (ex: "Escreva em um nível que um gerente de RH possa entender").
*   **Especifique a Jurisdição:** O Direito do Trabalho é altamente dependente da jurisdição. Sempre inclua o estado, país ou lei específica (ex: "sob a lei da Califórnia", "em conformidade com a CLT brasileira").
*   **Forneça Contexto e Dados:** Inclua o máximo de contexto relevante (ex: "o funcionário recebeu 3 advertências formais nos últimos 6 meses") para evitar "alucinações" e garantir a precisão legal.
*   **Exija o Formato:** Especifique o formato de saída (ex: "tabela", "lista de verificação", "rascunho de e-mail") para garantir que o resultado seja imediatamente útil.
*   **Validação Humana (Mandatória):** **Nunca** use a saída da IA como aconselhamento jurídico final ou documento sem uma revisão e validação minuciosa por um profissional jurídico qualificado. A IA deve ser tratada como um "primeiro rascunho" ou "estagiário".

## Use Cases
*   **Conformidade e Auditoria:** Criar listas de verificação de conformidade para contratação, demissão, salários e horas, e leis de licença (ex: FMLA, leis estaduais).
*   **Rascunho de Documentos:** Gerar rascunhos iniciais de políticas de RH (ex: código de conduta, política de trabalho remoto, política de PTO), acordos de confidencialidade (NDAs) e cartas de rescisão.
*   **Análise de Risco:** Avaliar os riscos legais de decisões de pessoal, como demissões, reclassificações de cargos ou implementação de novas tecnologias de monitoramento de funcionários.
*   **Treinamento e Comunicação:** Criar materiais de treinamento concisos para gerentes e funcionários sobre tópicos como assédio, discriminação e uso ético de IA no local de trabalho.
*   **Pesquisa Jurídica:** Resumir as implicações de novas leis ou decisões judiciais (jurisprudência) para as políticas internas da empresa.

## Pitfalls
*   **Alucinações Legais:** A IA pode gerar informações jurídicas falsas, mas convincentes (alucinações), especialmente em jurisdições menos comuns ou em leis muito recentes. Isso pode levar a erros de conformidade graves.
*   **Violação de Confidencialidade e Sigilo:** Inserir dados confidenciais de funcionários ou informações privilegiadas (como detalhes de um caso de litígio) no prompt pode violar o sigilo profissional e as políticas de privacidade de dados.
*   **Viés Algorítmico:** A IA pode perpetuar ou amplificar vieses existentes nos dados de treinamento, resultando em políticas ou análises que levam a resultados discriminatórios (ex: em processos de triagem de currículos ou análise de desempenho).
*   **Generalização Excessiva:** A IA pode não levar em conta as nuances das leis estaduais ou locais (ex: diferenças nas leis de não concorrência ou transparência salarial), especialmente nos EUA, onde as leis trabalhistas variam muito.
*   **Falta de Fonte:** Muitas ferramentas de IA generativa não fornecem citações ou fontes para suas respostas jurídicas, dificultando a verificação da precisão e da atualidade da informação.

## URL
[https://tenthings.blog/2025/03/31/ten-things-practical-generative-ai-prompts-for-in-house-lawyers/](https://tenthings.blog/2025/03/31/ten-things-practical-generative-ai-prompts-for-in-house-lawyers/)
