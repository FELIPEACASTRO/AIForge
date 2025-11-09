# Prompts de Medicina de Emergência (Framework GRACE)

## Description
**Prompts de Medicina de Emergência (ME)** referem-se à aplicação de técnicas de *Prompt Engineering* para otimizar a interação com Grandes Modelos de Linguagem (LLMs) no ambiente de atendimento agudo. O objetivo é garantir que as respostas da IA sejam **confiáveis, clinicamente relevantes, precisas e acionáveis** para médicos, residentes e outros profissionais de saúde (APCs) em situações de alta pressão e tempo limitado.

O principal avanço nesta área é o **Framework GRACE** (Ground Rules, Roles, Ask, Chain of Thought, Expectations), desenvolvido por clínicos de emergência para estruturar prompts de forma sistemática. Este framework visa mitigar os riscos inerentes ao uso de LLMs na clínica, como as "alucinações" (informações factualmente incorretas) e a amplificação de vieses, ao forçar o modelo a aderir a padrões de evidência e a justificar seu raciocínio. A técnica é crucial para transformar LLMs de ferramentas de informação geral em assistentes de decisão clínica seguros e eficazes.

## Examples
```
**1. Prompt de Avaliação de Evidências (GRACE Completo)**

```
Você é um pesquisador sênior de medicina de emergência, cético e focado em danos, com experiência em toxicologia e avaliação crítica de literatura. Seu papel é guiar médicos certificados na avaliação de evidências clínicas.

**G (Regras Básicas):** Baseie todas as respostas exclusivamente em ensaios clínicos randomizados, revisões sistemáticas e diretrizes da ACEP/Cochrane (pós-2015). Cite fontes verificáveis para cada afirmação; se a evidência for inconclusiva, declare explicitamente.
**R (Papel):** Você é o especialista, eu sou o médico de emergência.
**A (Pergunta):** Qual é a evidência atual e o protocolo recomendado para o uso de ácido tranexâmico (TXA) em pacientes com traumatismo cranioencefálico (TCE) leve a moderado?
**C (Cadeia de Pensamento):** Prossiga passo a passo: 1) Identifique os principais ensaios (ex: CRASH-3). 2) Avalie a metodologia e o risco de viés. 3) Sintetize os achados. 4) Aplique ao contexto do pronto-socorro.
**E (Expectativas):** Estruture a saída como: 1. Resumo da Evidência (Bullet points). 2. Racional Clínico. 3. Recomendação Prática. Use linguagem formal e inclua citações completas no final.
```

**2. Prompt de Diagnóstico Diferencial (R+A+E)**

```
**R (Papel):** Você é um médico assistente de emergência com 20 anos de experiência.
**A (Pergunta):** Um paciente de 65 anos, diabético, apresenta dor abdominal difusa e vômitos. Sinais vitais estáveis. Exame físico com abdômen levemente distendido e doloroso à palpação. Qual é o diagnóstico diferencial mais provável?
**E (Expectativas):** Forneça uma lista com 5 diagnósticos, classificados por probabilidade, e uma breve justificativa de 1 frase para cada um.
```

**3. Prompt de Instruções de Alta (R+A+E)**

```
**R (Papel):** Você é um educador de pacientes.
**A (Pergunta):** Crie instruções de alta para um paciente de 30 anos diagnosticado com celulite não purulenta na perna, tratado com cefalexina.
**E (Expectativas):** Use linguagem de nível de 6ª série. Inclua: 1. Sinais de alerta para retornar ao PS. 2. Cuidados com a ferida. 3. Instruções sobre o antibiótico (dose, duração, efeitos colaterais). Formate como uma lista numerada.
```

**4. Prompt de Resumo de Caso para Handoff (A+C+E)**

```
**A (Pergunta):** Resuma o seguinte caso para um handoff rápido para a equipe de internação: [Insira o resumo do caso, incluindo HPI, Exame Físico, Labs/Imagens, e Plano].
**C (Cadeia de Pensamento):** Siga a estrutura SBAR (Situação, Background, Avaliação, Recomendação).
**E (Expectativas):** A saída deve ter no máximo 4 frases, uma para cada seção SBAR.
```

**5. Prompt de Simulação de Cenário (R+A+C)**

```
**R (Papel):** Você é o examinador do Conselho Americano de Medicina de Emergência (ABEM). Eu sou o candidato.
**A (Pergunta):** Apresento um paciente de 45 anos com dor torácica atípica e ECG com depressão de ST em V4-V6. Qual é o seu próximo passo imediato?
**C (Cadeia de Pensamento):** Justifique cada decisão com a fisiopatologia ou diretriz relevante antes de prosseguir para o próximo passo. Mantenha a simulação interativa.
```

**6. Prompt de Revisão de Tópico Rápida (G+A+E)**

```
**G (Regras Básicas):** Use apenas as diretrizes mais recentes do American Heart Association (AHA) e do American College of Cardiology (ACC).
**A (Pergunta):** Quais são as indicações e contraindicações absolutas para a trombólise em um paciente com acidente vascular cerebral isquêmico agudo que chega à janela de 3 horas?
**E (Expectativas):** Responda em uma tabela com duas colunas: "Indicações Absolutas" e "Contraindicações Absolutas".
```
```

## Best Practices
**Adotar o Framework GRACE (Ground Rules, Roles, Ask, Chain of Thought, Expectations):** Este é o principal conjunto de práticas recomendadas para prompts de ME.
*   **G (Ground Rules - Regras Básicas):** Defina explicitamente as restrições e os padrões de evidência (ex: "Baseie-se apenas em literatura revisada por pares. Forneça citações. Não invente fontes.").
*   **R (Roles - Papéis):** Atribua papéis específicos ao usuário e ao LLM (ex: "Você é um médico assistente experiente em ME, eu sou um residente.").
*   **A (Ask - Pergunta):** Seja explícito e focado na tarefa central (ex: "Forneça o diagnóstico diferencial para este caso...").
*   **C (Chain of Thought - Cadeia de Pensamento):** Peça ao LLM para "explicar seu raciocínio passo a passo" para expor o processo de tomada de decisão e reduzir o risco de "caixa preta".
*   **E (Expectations - Expectativas):** Defina o formato e o estilo de saída para usabilidade (ex: "Responda de forma concisa em uma lista com marcadores.").
**Priorizar Fontes Confiáveis:** Use LLMs que sejam "fundamentados" (grounded) em literatura médica revisada por pares e diretrizes clínicas, como o OpenEvidence, em vez de modelos de propósito geral para decisões críticas.
**Incluir Dados Clínicos Relevantes:** Para prompts de caso, inclua informações cruciais como idade, sexo, comorbidades, sinais vitais, achados de exames e medicamentos atuais para evitar "alucinações" por falta de dados.

## Use Cases
**Suporte à Decisão Clínica:**
*   **Diagnóstico Diferencial:** Geração de listas de DDX para casos complexos ou atípicos, com base em dados de apresentação do paciente.
*   **Recomendação de Tratamento:** Consulta de diretrizes e evidências para o manejo de condições agudas (ex: sepse, IAM, AVC).
*   **Interpretação de Exames:** Auxílio na interpretação de achados laboratoriais ou de imagem incomuns.
**Educação e Treinamento:**
*   **Simulação de Casos:** Criação de cenários clínicos interativos para treinamento de residentes e estudantes, simulando o papel de um examinador ou paciente.
*   **Revisão Rápida de Tópicos:** Síntese de informações complexas de toxicologia, farmacologia ou procedimentos em formatos de fácil digestão.
**Eficiência Operacional:**
*   **Instruções de Alta Personalizadas:** Geração de instruções de alta claras, em linguagem simples e adaptadas às condições e ao nível de alfabetização do paciente.
*   **Comunicação de Handoff (SBAR):** Resumo rápido e estruturado de casos para transição de cuidados entre equipes (ex: do PS para a UTI ou enfermaria).
*   **Documentação Clínica:** Criação de rascunhos de notas clínicas ou justificativas para procedimentos.

## Pitfalls
**Alucinações e Fontes Falsas:** O maior risco. LLMs de propósito geral podem inventar estudos, autores ou citações. **Mitigação:** Use a regra básica "Forneça citações verificáveis e não invente fontes."
**Vieses de Dados:** Os modelos podem perpetuar vieses de saúde existentes (ex: subestimar a dor em certas populações). **Mitigação:** Peça ao LLM para considerar a equidade e as disparidades de saúde em sua análise (ex: "Considere como o manejo pode diferir em populações com acesso limitado à saúde.").
**Falta de Contexto Clínico:** A IA não tem acesso ao paciente, ao monitor ou ao ambiente. **Mitigação:** Seja o mais detalhado possível no prompt, fornecendo todos os dados vitais e de exames relevantes.
**Respostas Não Acionáveis:** Respostas longas, densas ou acadêmicas demais para um ambiente de emergência. **Mitigação:** Use a regra de Expectativas (E) para exigir formatos concisos (ex: "Bottom line up front", lista com marcadores).
**Dependência Excessiva:** Usar a IA como substituto para o julgamento clínico. **Mitigação:** A IA deve ser usada como um assistente de raciocínio, e não como uma autoridade final. O Framework GRACE incentiva a revisão crítica do raciocínio da IA.

## URL
[https://www.acepnow.com/article/search-with-grace-artificial-intelligence-prompts-for-clinically-related-queries/](https://www.acepnow.com/article/search-with-grace-artificial-intelligence-prompts-for-clinically-related-queries/)
