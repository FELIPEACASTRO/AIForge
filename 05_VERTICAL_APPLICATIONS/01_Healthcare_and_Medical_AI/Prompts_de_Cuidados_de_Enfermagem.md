# Prompts de Cuidados de Enfermagem

## Description
Prompts de Cuidados de Enfermagem referem-se à aplicação estratégica da Engenharia de Prompt para guiar Modelos de Linguagem Grande (LLMs) na geração de resultados estruturados, precisos e contextualmente relevantes para a prática e educação em enfermagem. Esta técnica é crucial para transformar a IA em uma ferramenta de apoio clínico e educacional, permitindo que enfermeiros e estudantes criem rapidamente planos de cuidados detalhados (seguindo formatos como NANDA), fichas de farmacologia, resumos de comunicação clínica (SBAR) e materiais de estudo personalizados. O foco principal é fornecer à IA um papel, um cenário clínico específico e restrições de formato rigorosas para garantir que a saída seja clinicamente aceitável e academicamente rigorosa. O uso eficaz desses prompts visa reduzir o tempo gasto em tarefas administrativas e de documentação, permitindo que os profissionais se concentrem mais no cuidado direto ao paciente.

## Examples
```
**1. Prompt de Plano de Cuidados (NANDA):**
> "Você é um professor de enfermagem de nível BSN. Ajude-me a elaborar um plano de cuidados de nível RN para um cliente adulto com **[problema primário]** relacionado a **[etiologia]** conforme evidenciado por **[sinais/sintomas]**. Restrições: 1) Siga o formato NANDA, 2) Inclua dois objetivos SMART (com prazo), 3) Forneça quatro intervenções de enfermagem com **justificativas** e **fontes de evidência**, 4) Adicione critérios de avaliação que posso verificar em 24-48h. Saída em uma tabela de duas colunas (Plano / Notas para Documentação)."

**2. Prompt de Ficha de Farmacologia:**
> "Crie uma ficha de farmacologia concisa para **[droga/classe]** cobrindo: mecanismo de ação, indicações, efeitos adversos de alto rendimento, contraindicações, interações, considerações de enfermagem e ensino ao paciente. Inclua duas perguntas estilo NCLEX com justificativas. Formate como títulos e marcadores."

**3. Prompt de Comunicação SBAR:**
> "Converta esta situação em um SBAR conciso para uma passagem de plantão para o médico de plantão. Inclua um pedido/recomendação de uma linha no final. Situação: **[Descreva brevemente o cenário do paciente, ex: Paciente de 68 anos, pós-operatório de colecistectomia, com dor abdominal crescente e hipotensão]**."

**4. Prompt de Racionalização de Prioridade (NGN):**
> "Explique por que as ações corretas são prioridade para este caso em três etapas: 1) fisiopatologia do problema, 2) como a ação muda a fisiologia, 3) qual parâmetro prova o sucesso em 10-30 minutos. Caso: **[Descreva o caso clínico]**."

**5. Prompt de Comunicação Terapêutica:**
> "Forneça cinco respostas terapêuticas e duas não terapêuticas para esta declaração do cliente: 'Sinto que estou falhando como pai.' Rotule cada resposta e explique o porquê."

**6. Prompt de Geração de Cenário de Simulação:**
> "Gere um cenário de simulação de alta fidelidade para estudantes de enfermagem de nível ADN sobre **[tópico]**. Inclua: histórico do paciente, sinais vitais iniciais, resultados laboratoriais críticos, e três ações de enfermagem esperadas (prioridade)."
```

## Best Practices
**1. Clareza e Contexto (Regra de Ouro):** Sempre defina o seu papel (ex: "Você é um professor de enfermagem", "Você é um enfermeiro de UTI"), o nível de conhecimento esperado (ex: "nível BSN", "primeiro termo"), o cenário do paciente e o formato de saída desejado (ex: "tabela de duas colunas", "lista com marcadores").
**2. Estrutura e Formato:** Peça explicitamente por formatos estruturados como tabelas, listas com marcadores ou checklists. Isso facilita a revisão e a extração de informações.
**3. Restrições Específicas:** Inclua restrições acadêmicas ou clínicas, como "Siga o formato NANDA", "Inclua dois objetivos SMART" ou "Use a estrutura SBAR".
**4. Verificação Humana:** Nunca use a saída da IA diretamente no cuidado ao paciente ou em trabalhos acadêmicos sem uma verificação cruzada com fontes confiáveis (livros didáticos, guias de medicamentos, protocolos hospitalares).
**5. Ação e Parâmetro:** Ao pedir intervenções, peça para a IA listar a **Ação** e o **Parâmetro** que prova o sucesso da ação (ex: Ação: Administrar oxigênio. Parâmetro: SpO₂ > 92% em 10 minutos).

## Use Cases
**1. Educação em Enfermagem:**
*   Criação de planos de cuidados (PC) detalhados e prontos para rubrica para tarefas acadêmicas.
*   Geração de perguntas de prática estilo NCLEX (Next Generation NCLEX) com justificativas para estudo.
*   Desenvolvimento de fichas de medicamentos concisas e mnemônicos para farmacologia.
*   Transformação de notas de aula em guias de estudo estruturados e flashcards para recordação ativa.

**2. Prática Clínica e Documentação:**
*   Elaboração de resumos de passagem de plantão usando a estrutura SBAR (Situação, Histórico, Avaliação, Recomendação).
*   Geração de sugestões de comunicação terapêutica para interações difíceis com pacientes ou familiares.
*   Criação de protocolos de segurança e listas de verificação (checklists) para procedimentos específicos.
*   Auxílio na documentação, como a formulação de diagnósticos de enfermagem precisos e objetivos de cuidado.

**3. Pesquisa e Desenvolvimento Profissional:**
*   Explicação de fisiopatologias complexas em termos leigos para educação do paciente.
*   Comparação e contraste de diretrizes de tratamento ou classes de medicamentos.
*   Auxílio na redação de artigos, propostas de pesquisa ou documentos de política de saúde.

## Pitfalls
**1. Confiança Excessiva (Alucinações):** O maior risco é confiar cegamente na saída da IA. Os LLMs podem "alucinar" fatos, doses de medicamentos ou intervenções que parecem plausíveis, mas são clinicamente incorretas ou perigosas. **Sempre verifique.**
**2. Violação de Privacidade (PHI):** Nunca insira informações de saúde protegidas (PHI) reais nos prompts. O prompt deve ser desidentificado e generalizado (ex: "paciente de 68 anos com insuficiência cardíaca", em vez de "João da Silva, 68 anos, leito 302").
**3. Falta de Contexto:** Prompts vagos (ex: "O que é insuficiência cardíaca?") resultarão em respostas genéricas e inúteis. A saída deve ser específica para o seu papel e nível de estudo/prática.
**4. Viés e Inadequação Cultural:** A IA pode perpetuar vieses ou fornecer conselhos culturalmente insensíveis. Os enfermeiros devem revisar a saída para garantir que ela seja apropriada para o paciente e o ambiente.
**5. Não Especificar o Formato:** Não pedir um formato estruturado (tabela, lista) leva a longos blocos de texto difíceis de analisar e usar em um ambiente clínico ou acadêmico.

## URL
[https://goodnurse.com/article/199/ai-prompt-library-for-nursing-students-2025-care-plans-pharmacology-ngn-study-notes](https://goodnurse.com/article/199/ai-prompt-library-for-nursing-students-2025-care-plans-pharmacology-ngn-study-notes)
