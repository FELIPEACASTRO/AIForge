# Physical Therapy Prompts (Prompts para Fisioterapia)

## Description
A Engenharia de Prompt para Fisioterapia (Physical Therapy Prompts) refere-se à arte e ciência de criar entradas de texto otimizadas para modelos de Linguagem Grande (LLMs) e assistentes de IA, com o objetivo de obter resultados clinicamente relevantes, precisos e acionáveis no contexto da prática fisioterapêutica [1].

Essa técnica permite que fisioterapeutas utilizem a IA como um assistente clínico e administrativo, auxiliando em tarefas como:
1.  **Geração de Planos de Tratamento:** Criação de programas de reabilitação personalizados e baseados em fases [2].
2.  **Documentação Clínica:** Elaboração de notas SOAP, resumos de alta e relatórios de progresso [3].
3.  **Educação do Paciente:** Desenvolvimento de materiais didáticos e respostas a perguntas comuns de pacientes [1].
4.  **Apoio à Decisão Clínica:** Identificação de fatores prognósticos, testes diagnósticos e contraindicações [2].

A eficácia do prompt depende diretamente da **clareza, especificidade e contextualização** do pedido, sendo a estrutura **Role-Goal-Instruction (RGI)** (Papel-Objetivo-Instrução) uma das metodologias mais recomendadas para maximizar a utilidade clínica da resposta da IA [1].

## Examples
```
**1. Criação de Programa de Reabilitação (RGI)**
*   **Prompt:** "Você é um fisioterapeuta especialista em reabilitação de joelho. Seu objetivo é criar um programa de exercícios domiciliares para um paciente de 55 anos, sedentário, 6 semanas após uma artroplastia total de joelho. O programa deve focar em ganho de amplitude de movimento (ROM) e fortalecimento inicial. Apresente o plano em uma tabela com 3 colunas: Exercício, Frequência (séries x repetições), e Precauções. Inclua 5 exercícios."

**2. Análise de Fatores Prognósticos**
*   **Prompt:** "Quais são os 5 principais fatores prognósticos (positivos e negativos) para o sucesso do tratamento conservador da dor lombar crônica inespecífica, de acordo com as diretrizes clínicas de 2023-2025? Liste-os e forneça uma breve justificativa baseada em evidências para cada um."

**3. Geração de Nota SOAP (Structured Output)**
*   **Prompt:** "Atue como um assistente de documentação clínica. Gere uma nota SOAP para o seguinte cenário: Paciente: Atleta de 22 anos, entorse de tornozelo grau II há 48h. Subjetivo: Relata dor 7/10 (EVA) ao caminhar e edema. Objetivo: Edema moderado, teste de gaveta anterior positivo, ROM de dorsiflexão limitada em 50%. Avaliação: Entorse de tornozelo lateral aguda. Plano: Iniciar mobilização suave, exercícios isométricos e crioterapia. Formate a saída estritamente no formato SOAP, com títulos em negrito."

**4. Desenvolvimento de Material Educacional**
*   **Prompt:** "Crie um folheto educativo de uma página, com linguagem simples e acessível (nível de leitura de 6ª série), para um paciente com diagnóstico recente de Síndrome do Túnel do Carpo. O folheto deve explicar a condição, listar 3 dicas de ergonomia no trabalho e 3 exercícios de deslizamento neural. Use títulos e bullet points."

**5. Identificação de Contraindicações**
*   **Prompt:** "Quais são as contraindicações absolutas e relativas para o uso de terapia por ondas de choque (ESWT) em um paciente com tendinopatia de Aquiles? Apresente a resposta em duas listas separadas e cite a fonte da diretriz clínica (ex: NICE, Cochrane) que suporta a informação."
```

## Best Practices
**Seja Específico e Contextual:** Utilize a estrutura **Role-Goal-Instruction (RGI)** (Papel-Objetivo-Instrução) para fornecer contexto claro (ex: "Você é um fisioterapeuta especializado em reabilitação esportiva"). Inclua detalhes clínicos relevantes, como idade, nível de atividade, tempo de lesão e tratamentos prévios [1].

**Solicite Evidências e Formato:** Peça explicitamente que a IA baseie a resposta em **evidências** (ex: "baseado nas diretrizes da APTA 2024") e defina o formato de saída (ex: "apresente em formato de tabela com 3 colunas: Exercício, Repetições, Objetivo") [2].

**Refinamento Iterativo:** Comece com um prompt mais amplo e refine-o com perguntas de acompanhamento para obter detalhes mais específicos (ex: "Expanda o exercício X, detalhando a progressão para as semanas 4-6") [1].

**Mantenha a Confidencialidade:** **NUNCA** inclua informações de identificação do paciente (PHI) nos prompts. Use dados clínicos anonimizados e genéricos [1].

## Use Cases
**Apoio à Decisão Clínica:**
*   Sugestão de testes ortopédicos e sua sensibilidade/especificidade para condições específicas [2].
*   Análise de artigos de pesquisa para resumir a eficácia de uma nova modalidade de tratamento.

**Otimização da Documentação:**
*   Geração de rascunhos de notas de progresso (SOAP) ou resumos de alta, economizando tempo administrativo [3].
*   Conversão de notas de voz (transcritas) em documentação clínica estruturada.

**Personalização do Tratamento:**
*   Criação de planos de exercícios personalizados, ajustados para comorbidades (ex: diabetes, osteoporose) ou restrições ambientais (ex: exercícios que podem ser feitos em casa sem equipamento) [1].
*   Desenvolvimento de critérios de progressão e regressão de exercícios baseados na fase de cicatrização do tecido.

**Educação e Comunicação:**
*   Elaboração de respostas claras e concisas para perguntas frequentes dos pacientes (FAQs) sobre sua condição ou tratamento [1].
*   Criação de conteúdo para mídias sociais ou apresentações clínicas sobre tópicos de fisioterapia.

## Pitfalls
**Alucinações Clínicas:** A IA pode gerar informações que parecem plausíveis, mas são clinicamente incorretas ou desatualizadas (alucinações). **Sempre verifique as recomendações da IA** com o raciocínio clínico e diretrizes baseadas em evidências [1] [2].

**Falta de Especificidade:** Prompts vagos (ex: "Quais exercícios para o ombro?") resultam em respostas genéricas e de baixa utilidade clínica. A falta de detalhes sobre o paciente (idade, comorbidades, fase da lesão) é um erro comum [1].

**Violação de Confidencialidade (HIPAA/LGPD):** Incluir dados de identificação do paciente (nome, data de nascimento, endereço) em prompts de IA é uma grave violação de privacidade e ética profissional. A anonimização é obrigatória [1].

**Dependência Excessiva:** Usar a IA como substituto para o julgamento clínico. A IA é uma ferramenta de suporte, não um substituto para a avaliação e tomada de decisão do fisioterapeuta [2].

**Ignorar o Formato de Saída:** Não especificar o formato (tabela, lista, nota SOAP) pode levar a respostas desorganizadas e difíceis de integrar ao fluxo de trabalho clínico [3].

## URL
[https://www.physio-pedia.com/Physiopedia_AI_Assistant_Prompt_Writing_Guide](https://www.physio-pedia.com/Physiopedia_AI_Assistant_Prompt_Writing_Guide)
