# Trial Preparation Prompts (Prompts de Preparação para Julgamento)

## Description
**Prompts de Preparação para Julgamento** são um conjunto de instruções estruturadas e detalhadas fornecidas a modelos de linguagem avançados (LLMs) para auxiliar profissionais do direito em tarefas críticas da fase pré-julgamento e julgamento. Esta técnica se enquadra na subcategoria **Legal** e visa otimizar a análise de casos, a formulação de estratégias, a preparação de testemunhas, a elaboração de roteiros de inquirição e a identificação de precedentes relevantes. A eficácia reside na capacidade de simular cenários, sintetizar grandes volumes de documentos (como transcrições e descobertas) e gerar argumentos e contra-argumentos de forma rápida e estruturada, liberando o tempo do advogado para o raciocínio estratégico e a atuação em tribunal. A chave é fornecer à IA um papel (persona), o contexto do caso e o formato de saída desejado.

## Examples
```
**1. Análise de Transcrição de Testemunha:**
"**Papel:** Você é um advogado de acusação. **Tarefa:** Analise a transcrição da declaração de [Nome da Testemunha] fornecida abaixo. **Instrução:** Identifique todas as inconsistências, contradições com [Documento X] e pontos fracos que podem ser explorados no cross-examination. Liste em uma tabela com as colunas: 'Ponto Fraco', 'Referência na Transcrição' e 'Pergunta Sugerida para Exploração'."

**2. Roteiro de Inquirição Direta:**
"**Papel:** Você é um advogado especialista em direito de família. **Caso:** Divórcio litigioso envolvendo custódia. **Tarefa:** Crie um roteiro de perguntas para a inquirição direta da minha cliente, [Nome da Cliente]. **Objetivo:** Estabelecer a estabilidade emocional e a capacidade de prover um ambiente seguro. **Formato:** Divida em seções lógicas (Antecedentes, Rotina, Incidentes Chave) e use linguagem clara e não sugestiva."

**3. Simulação de Argumento de Abertura:**
"**Papel:** Você é um juiz cético. **Tarefa:** Avalie o rascunho do meu argumento de abertura (texto fornecido abaixo). **Instrução:** Identifique os três pontos mais fracos e sugira possíveis objeções que o advogado adversário faria. Em seguida, reescreva a introdução para ser mais persuasiva e concisa, focando na narrativa de [Fato Central]."

**4. Identificação de Precedentes Contrários:**
"**Tarefa:** Pesquise a jurisprudência mais recente (últimos 2 anos) na jurisdição de [Tribunal/Estado] sobre [Questão Legal Específica, ex: 'admissibilidade de evidência digital sem cadeia de custódia completa']. **Instrução:** Liste os 5 precedentes mais desfavoráveis ao meu argumento (que defende a admissibilidade) e resuma a *ratio decidendi* de cada um. Use o formato de lista numerada."

**5. Preparação para Cross-Examination (Testemunha Perita):**
"**Papel:** Você é um advogado de defesa. **Testemunha:** [Nome do Perito], perito em [Área de Expertise]. **Relatório:** [Resumo das conclusões do relatório]. **Tarefa:** Gere 10 perguntas 'matadoras' para o cross-examination, focando em questionar a metodologia, a falta de dados e a parcialidade do perito. As perguntas devem ser curtas e projetadas para obter respostas 'sim' ou 'não'."

**6. Resumo de Descoberta (Discovery):**
"**Tarefa:** Analise os seguintes documentos de descoberta (e-mails e memorandos) relacionados ao caso [Nome do Caso]. **Instrução:** Crie um resumo executivo de uma página, destacando as 5 evidências mais incriminatórias contra a parte adversa e as 3 evidências mais prejudiciais à minha própria posição. Use negrito para as palavras-chave."
```

## Best Practices
**1. Defina o Papel (Persona):** Comece o prompt instruindo a IA a assumir o papel de um profissional jurídico específico (ex: "Você é um advogado de defesa criminal com 15 anos de experiência..."). Isso alinha o tom e o foco da resposta.
**2. Forneça Contexto Detalhado:** Inclua fatos cruciais do caso, jurisdição, tipo de tribunal, e as regras de evidência aplicáveis. Quanto mais dados de entrada (documentos, transcrições) você fornecer, melhor.
**3. Seja Específico no Formato:** Peça o resultado em um formato estruturado (ex: "Liste em tópicos", "Crie uma tabela com 3 colunas", "Formate como um roteiro de perguntas").
**4. Itere e Refine:** Use o output inicial da IA como rascunho. Peça refinamentos, como "Agora, reescreva a pergunta 5 para ser mais incisiva" ou "Identifique a jurisprudência contrária a este argumento".
**5. Validação Humana é Obrigatória:** Nunca use o output da IA diretamente em um processo judicial sem a revisão e validação de um profissional jurídico qualificado. A IA é uma ferramenta de assistência, não um substituto para o julgamento legal.

## Use Cases
**1. Formulação de Estratégia de Caso:** Analisar os fatos e a lei aplicável para identificar a teoria do caso mais forte e as vulnerabilidades da parte adversa.
**2. Preparação de Testemunhas:** Gerar listas de perguntas para inquirição direta e cruzada, focando em áreas de potencial conflito ou reforço de narrativa.
**3. Síntese de Descoberta (Discovery):** Resumir rapidamente milhares de documentos, e-mails ou transcrições para extrair evidências-chave e cronogramas de eventos.
**4. Elaboração de Argumentos:** Criar rascunhos de argumentos de abertura, fechamento e moções processuais, garantindo que todos os elementos legais sejam abordados.
**5. Simulação de Julgamento (Mock Trial):** Usar a IA para simular o papel de um juiz, júri ou advogado adversário, testando a força dos argumentos e antecipando objeções.
**6. Pesquisa de Jurisprudência:** Identificar precedentes favoráveis e desfavoráveis, bem como analisar a *ratio decidendi* de decisões recentes em questões específicas do caso.

## Pitfalls
**1. Alucinações e Falsos Precedentes:** A IA pode "inventar" casos, estatutos ou citações (conhecido como alucinação). **Risco:** Usar um precedente inexistente em tribunal, resultando em sanções ou perda de credibilidade.
**2. Falta de Contexto Local:** A IA pode não estar atualizada com as regras processuais locais, costumes do tribunal ou a jurisprudência mais recente de uma jurisdição específica. **Risco:** Estratégias ou documentos legalmente falhos.
**3. Violação de Confidencialidade:** Inserir informações confidenciais do cliente em modelos de IA não seguros ou públicos. **Risco:** Violação ética e de sigilo profissional.
**4. Dependência Excessiva:** Confiar na IA para o raciocínio jurídico complexo ou para a tomada de decisões estratégicas. **Risco:** Perda da capacidade de julgamento crítico do advogado e respostas genéricas que não se aplicam ao caso.
**5. Prompts Vagos:** Pedir um "roteiro de julgamento" sem especificar o papel, o objetivo, a jurisdição e os fatos relevantes. **Risco:** Receber um output inútil e desperdiçar tempo.

## URL
[https://callidusai.com/blog/top-ai-legal-prompts-lawyers-2025/](https://callidusai.com/blog/top-ai-legal-prompts-lawyers-2025/)
