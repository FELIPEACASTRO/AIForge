# Pediatrics Prompts (Prompts em Pediatria)

## Description
A Engenharia de Prompts em Pediatria é a aplicação estratégica de técnicas de comunicação com Modelos de Linguagem Grande (LLMs) para otimizar a prática clínica e administrativa no cuidado infantil. Envolve a criação de instruções detalhadas e contextuais para guiar a IA a gerar saídas precisas, seguras e adequadas às nuances do desenvolvimento e das patologias pediátricas. O foco principal é garantir a segurança do paciente, a comunicação eficaz com pais e cuidadores, e o suporte ao raciocínio clínico, sempre respeitando a necessidade de validação humana e o julgamento médico final [1] [2].

## Examples
```
**1. Anamnese Inicial Estruturada (Role-playing + Reasoning):**
"Você é um pediatra recebendo uma criança de 5 anos em primeira consulta. Liste 12 perguntas essenciais para anamnese inicial, dividindo em blocos: queixa principal, histórico de desenvolvimento, antecedentes, hábitos atuais e vacinação. Antes de listar as perguntas, explique em 3 frases o motivo da divisão em blocos e por que ela é importante para o raciocínio clínico."

**2. Comunicação com Pais (Tone Specification + Output Specification):**
"Assuma o papel de um pediatra. Redija uma mensagem de WhatsApp para os pais de um paciente de 8 anos com diagnóstico de Otite Média Aguda. A mensagem deve ser concisa (máximo 50 palavras), com tom empático e profissional, e deve incluir: o diagnóstico, a prescrição do antibiótico (Amoxicilina), e 3 sinais de alerta para retorno imediato ao pronto-socorro."

**3. Explicação de Doença para Criança (Role-playing + Customization):**
"Assuma o papel de um educador de saúde infantil. Explique o que é a Asma para uma criança de 6 anos. Use analogias simples (ex: tubos de ar que ficam apertados), linguagem de fácil compreensão e um tom encorajador. A explicação deve ter no máximo 5 frases curtas."

**4. Criação de Conteúdo Educacional (Output Specification + Context):**
"Crie um folheto informativo para pais sobre a introdução alimentar (BLW - Baby-Led Weaning) para bebês de 6 meses. O folheto deve ser formatado em 5 tópicos principais, com linguagem clara e acessível, e incluir uma lista de 5 alimentos seguros para começar."

**5. Resumo de Prontuário para Colega (Role-playing + Output Specification):**
"Você é um médico residente de pediatria. Resuma o prontuário do paciente [Nome, Idade, Diagnóstico Principal] para o plantonista que está chegando. O resumo deve ter no máximo 6 linhas e destacar: Queixa Principal, Conduta Adotada, Medicamentos em Uso e Próximos Passos (exames pendentes ou interconsultas)."

**6. Cálculo de Dose de Medicamento (Specific Instruction):**
"Calcule a dose de Amoxicilina para uma criança de 15 kg com Otite Média Aguda. Use a dose de 90 mg/kg/dia, dividida em duas tomadas. Forneça o resultado em mg por dose e o volume a ser administrado, considerando uma suspensão de 250 mg/5 ml."

**7. Simulação de Cenário Clínico (Reasoning Prompt + Role-playing):**
"Assuma o papel de um preceptor de residência. Crie um caso clínico simulado de um lactente de 3 meses com febre e irritabilidade. Inclua dados de exame físico e laboratoriais. Ao final, peça para o usuário listar 3 diagnósticos diferenciais e a conduta inicial, justificando o raciocínio."

**8. Elaboração de Plano de Follow-up (Chaining + Output Specification):**
"Com base no diagnóstico de TDAH em um adolescente de 14 anos, elabore um plano de acompanhamento multidisciplinar para os próximos 6 meses. O plano deve ser apresentado em uma tabela com 3 colunas: Mês, Especialista Envolvido (Pediatra, Psicólogo, Neurologista) e Objetivo da Consulta."

**9. Tradução de Termos Técnicos (Customization + Tone Specification):**
"Traduza o termo 'Enurese Noturna Monossintomática' para a linguagem de uma avó preocupada. Use um tom tranquilizador e explique o conceito de forma simples, focando que é uma condição comum e tratável."

**10. Geração de Conteúdo para Redes Sociais (Output Specification):**
"Crie 3 ideias de posts curtos (máximo 100 caracteres) para o Instagram de uma clínica pediátrica, focando na importância da vacinação contra a gripe em crianças. Use uma linguagem informal e chamativa, com foco em pais."
```

## Best Practices
**1. Seja Específico e Contextualizado (Role-playing e Contexto):** Sempre defina o papel da IA ("Você é um pediatra experiente") e forneça o contexto clínico completo (idade, peso, histórico, exames). A especificidade é crucial para obter respostas clinicamente relevantes e precisas [1] [2]. **2. Peça o Raciocínio (Reasoning Prompting):** Inclua a instrução para que a IA articule o processo de raciocínio por trás de sua resposta ("Explique o seu raciocínio passo a passo"). Isso aumenta a transparência, permite a validação humana e reduz a probabilidade de "alucinações" [1] [2]. **3. Forneça Exemplos (Few-shot Prompting):** Para tarefas que exigem um formato ou estilo específico (ex: resumo de prontuário, mensagem para pais), forneça um ou mais exemplos de entrada/saída para calibrar o modelo [1]. **4. Defina o Formato e o Tom (Output Specification e Tone Specification):** Especifique o formato de saída desejado (tabela, lista, texto corrido) e o tom (técnico para colegas, acessível e empático para pais) para garantir que o resultado seja útil para o público-alvo [1] [2]. **5. Itere e Valide:** O primeiro resultado nem sempre é o melhor. Ajuste o prompt e itere até obter a resposta ideal. **Nunca** use a saída da IA sem validação humana por um profissional de saúde qualificado [2].

## Use Cases
**1. Suporte à Decisão Clínica:** Auxílio na formulação de diagnósticos diferenciais, sugestão de protocolos de tratamento e cálculo de doses de medicamentos com base em peso e idade. **2. Comunicação com Pacientes e Pais:** Geração de mensagens, e-mails ou folhetos informativos em linguagem acessível e empática para explicar condições médicas complexas, planos de tratamento ou resultados de exames. **3. Otimização Administrativa:** Resumo de prontuários, transcrição estruturada de consultas, e elaboração de atestados ou laudos médicos. **4. Educação e Treinamento:** Criação de casos clínicos simulados para residentes e estudantes, e elaboração de conteúdo didático para pais e cuidadores [1] [2]. **5. Pesquisa e Revisão:** Busca e resumo de literatura médica atualizada sobre patologias pediátricas específicas.

## Pitfalls
**1. Vagueza e Generalidade:** Fornecer prompts genéricos sem especificar o contexto pediátrico (ex: "calcule a dose" sem o peso da criança) leva a respostas imprecisas ou perigosas [2]. **2. Viés e Alucinações:** A IA pode incorporar vieses de dados de treinamento, ou pior, gerar "alucinações" (informações falsas apresentadas como fatos) que são inaceitáveis em um ambiente clínico sensível como a pediatria [3]. **3. Falta de Contexto Clínico:** Não incluir informações cruciais como idade, peso, estágio de desenvolvimento ou histórico de vacinação no prompt impede que a IA considere as nuances do cuidado infantil [2]. **4. Confiança Excessiva (Over-reliance):** Usar a IA como substituto para o julgamento clínico ou para a tomada de decisões finais. A IA deve ser uma ferramenta de suporte, e não um substituto para o profissional de saúde [3]. **5. Ignorar a Necessidade de Validação:** A saída da IA, especialmente em cálculos de dose ou sugestões de diagnóstico, deve ser sempre validada por fontes médicas confiáveis e pelo médico responsável [1].

## URL
[https://www.childrenshospitals.org/news/childrens-hospitals-today/2024/12/5-tips-for-effective-ai-prompts](https://www.childrenshospitals.org/news/childrens-hospitals-today/2024/12/5-tips-for-effective-ai-prompts)
