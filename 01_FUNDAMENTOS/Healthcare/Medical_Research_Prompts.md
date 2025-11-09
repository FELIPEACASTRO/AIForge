# Medical Research Prompts

## Description
**Prompts para Pesquisa Médica** (ou *Medical Research Prompts*) referem-se à arte e ciência de elaborar instruções e entradas de texto otimizadas para **Modelos de Linguagem Grande (LLMs)**, como GPT-4 ou Claude, com o objetivo de obter saídas precisas, clinicamente relevantes e eticamente sólidas em contextos de saúde e pesquisa.

Esta técnica de *Prompt Engineering* é crucial porque a eficácia dos LLMs em ambientes clínicos e de pesquisa depende diretamente da qualidade do *prompt* de entrada. Em um campo onde a precisão é vital (por exemplo, diagnóstico, dosagem de medicamentos, síntese de evidências), *prompts* mal formulados podem levar a informações genéricas, imprecisas ou até perigosas.

O foco principal dos *Medical Research Prompts* é a **especificidade contextual e a adesão a práticas baseadas em evidências** [1]. Eles vão além da simples pergunta, incorporando elementos como: **Definição de Papel** (Role-Playing), **Inclusão de Contexto Clínico Detalhado**, **Restrição de Fonte de Evidência** e **Formato Estruturado de Saída** (PICO, SOAP, tabelas). Em essência, a engenharia de *prompts* na área médica transforma o LLM de um gerador de texto generalista em uma ferramenta de suporte à decisão e pesquisa altamente especializada, desde que o usuário forneça a estrutura e o contexto necessários [2].

**Referências:**
[1] Liu J, Liu F, Wang C, Liu S. Prompt Engineering in Clinical Practice: Tutorial for Clinicians. J Med Internet Res 2025;27(1):e72644.
[2] Meskó B. Prompt Engineering for Medical Professionals: Tutorial. J Med Internet Res 2023;25(1):e50638.

## Examples
```
**1. Suporte à Decisão Clínica (Baseado em Evidências)**
"**Aja como um consultor de medicina interna** com foco em diretrizes baseadas em evidências. **Cenário Clínico:** Paciente de 68 anos, sexo masculino, com diagnóstico recente de Insuficiência Cardíaca com Fração de Ejeção Reduzida (ICFER). O paciente está em uso de Sacubitril/Valsartana e Betabloqueador. **Pergunta:** Qual é a recomendação atual (Classe I) para a quarta classe de medicamentos (SGLT2i ou MRA) para ICFER, de acordo com as **diretrizes da AHA/ACC de 2022**? **Instruções de Saída:** 1. Liste o medicamento recomendado. 2. Descreva o mecanismo de ação relevante para ICFER. 3. Cite o principal ensaio clínico que suporta essa recomendação (Nome e Ano). 4. Apresente a resposta em formato de tabela."

**2. Formulação de Hipótese de Pesquisa (Formato PICO)**
"**Aja como um pesquisador sênior em epidemiologia.** **Objetivo:** Formular uma pergunta de pesquisa PICO (População, Intervenção, Comparação, Desfecho) bem definida. **Tópico:** O uso de telemonitoramento domiciliar em comparação com o acompanhamento clínico padrão para reduzir a taxa de reinternação em pacientes idosos com Doença Pulmonar Obstrutiva Crônica (DPOC). **Instruções de Saída:** 1. Identifique e defina cada componente PICO. 2. Formule a pergunta de pesquisa completa."

**3. Síntese de Literatura e Revisão Sistemática**
"**Aja como um revisor de literatura médica.** **Tarefa:** Analise os seguintes resumos de ensaios clínicos sobre o uso de [Nome do Medicamento] para [Condição]. **Resumos:** [INSERIR RESUMO 1], [INSERIR RESUMO 2], [INSERIR RESUMO 3]. **Instruções de Saída:** 1. Identifique os principais achados de eficácia e segurança em uma tabela comparativa. 2. Determine se os resultados são consistentes entre os estudos. 3. Escreva um parágrafo de conclusão sobre a força da evidência, destacando quaisquer vieses ou limitações."

**4. Criação de Material Educativo para Pacientes**
"**Aja como um educador de saúde.** **Público-alvo:** Paciente de 75 anos com baixo letramento em saúde, recém-diagnosticado com Fibrilação Atrial. **Objetivo:** Explicar o que é Fibrilação Atrial e por que o uso de anticoagulantes é crucial. **Instruções de Saída:** 1. Use linguagem simples, analogias e evite jargões médicos. 2. O tom deve ser tranquilizador e encorajador. 3. A explicação deve ter no máximo 200 palavras. 4. Inclua uma seção de 'O que fazer' com 3 pontos de ação claros."

**5. Otimização de Fluxo de Trabalho (Geração de Nota Clínica)**
"**Aja como um residente de clínica médica.** **Tarefa:** Gere uma nota clínica no formato SOAP (Subjetivo, Objetivo, Avaliação, Plano) com base nas seguintes informações. **Informações:** *Subjetivo:* Paciente relata dor torácica atípica intermitente há 2 dias, sem irradiação, aliviada com repouso. Nega dispneia ou palpitações. *Objetivo:* Exame físico sem alterações. ECG: Ritmo sinusal, sem alterações de ST-T. Troponina I: Negativa. *Avaliação:* Dor torácica atípica, provável causa musculoesquelética. Baixa probabilidade de Síndrome Coronariana Aguda (SCA). *Plano:* Alta hospitalar. Orientar retorno imediato em caso de dor típica. Prescrever anti-inflamatório. Agendar reavaliação em 7 dias. **Instruções de Saída:** 1. Formate a nota estritamente no formato SOAP. 2. Garanta que a seção 'Avaliação' justifique a baixa probabilidade de SCA."

**6. Análise de Dados (Interpretação de Resultados)**
"**Aja como um bioestatístico.** **Tarefa:** Interprete os resultados do seguinte estudo de fase III. **Resultados Chave:** *Desfecho Primário (Mortalidade por todas as causas):* Grupo Intervenção (n=500): 10% de mortalidade. Grupo Controle (n=500): 15% de mortalidade. *Hazard Ratio (HR):* 0.65 (IC 95%: 0.45-0.94). Valor de p: 0.02. **Instruções de Saída:** 1. Explique o significado do Hazard Ratio (HR) de 0.65. 2. Comente sobre a significância estatística (Valor de p) e a precisão (IC 95%). 3. Escreva uma conclusão clínica concisa sobre a eficácia da intervenção."
```

## Best Practices
**1. Expliciticidade e Especificidade:** Prompts devem ser claros, precisos e concisos. Incorporar variáveis clínicas específicas do paciente (idade, estágio da doença, comorbidades) e diretrizes aplicáveis. **2. Relevância Contextual:** Fornecer o contexto clínico completo, incluindo o papel do usuário (ex: "Você é um médico") e informações cruciais do paciente (histórico, medicamentos). **3. Refinamento Iterativo:** Começar com um prompt simples e refiná-lo progressivamente com base nas saídas do LLM, adicionando restrições e especificações. **4. Práticas Baseadas em Evidências:** Direcionar o LLM a usar fontes de informação confiáveis e atualizadas, especificando a fonte (ex: "Com base nas diretrizes da AHA/ACC de 2023"). **5. Considerações Éticas:** Evitar incluir informações de identificação pessoal (PII) e solicitar respostas neutras e baseadas em evidências, reconhecendo as limitações.

## Use Cases
**Geração de Resumos de Alta Qualidade:** Resumir prontuários, artigos de pesquisa ou diretrizes complexas. **Suporte à Decisão Clínica:** Auxiliar no diagnóstico diferencial, seleção de tratamento e manejo de doenças raras. **Educação do Paciente:** Gerar materiais educativos personalizados e de fácil compreensão. **Otimização de Fluxo de Trabalho:** Criar checklists, modelos de notas clínicas (SOAP) e cartas de encaminhamento. **Pesquisa e Revisão de Literatura:** Identificar artigos relevantes, sintetizar achados de ensaios clínicos e formular hipóteses de pesquisa. **Formulação de Hipóteses de Pesquisa:** Gerar perguntas de pesquisa PICO (População, Intervenção, Comparação, Desfecho) a partir de um cenário clínico. **Análise de Dados (Simulada):** Solicitar que o LLM interprete resultados de exames ou dados de ensaios clínicos (com dados fictícios ou anonimizados).

## Pitfalls
**Vagueza e Generalidade:** Prompts que não especificam o objetivo clínico, o perfil do paciente ou o contexto. **Ignorar o Contexto:** Falha em fornecer informações cruciais do paciente, levando a recomendações genéricas ou inadequadas. **Confiança Excessiva (Over-reliance):** Aceitar a saída do LLM sem verificação ou crítica clínica. **Violação de Privacidade:** Incluir dados de identificação pessoal (PII) nos prompts. **Falta de Iteração:** Não refinar o prompt após uma saída insatisfatória. **Ausência de Referência a Evidências:** Não solicitar que o LLM baseie sua resposta em diretrizes ou literatura específica.

## URL
[https://www.jmir.org/2025/1/e72644](https://www.jmir.org/2025/1/e72644)
