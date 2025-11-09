# Genomics & Precision Medicine Prompts

## Description
**Prompts de Genômica e Medicina de Precisão** referem-se a instruções especializadas e estruturadas projetadas para interagir com Grandes Modelos de Linguagem (LLMs) e modelos de IA multimodais no contexto da análise de dados genômicos, transcriptômicos, proteômicos e multi-ômicos, com o objetivo de avançar a medicina personalizada. Essa técnica de prompt engineering é crucial para tarefas que exigem um alto grau de precisão e conhecimento de domínio, como a interpretação de variantes genéticas, o design de experimentos de edição gênica (ex: CRISPR), a previsão de risco de doenças, a recomendação de terapias personalizadas e a comunicação de resultados genéticos complexos a pacientes. A eficácia desses prompts depende da capacidade de integrar dados biológicos complexos, literatura científica e diretrizes clínicas, muitas vezes utilizando técnicas avançadas como Geração Aumentada por Recuperação (RAG) e atribuição de papéis (role-playing) especializados ao modelo. O foco é transformar dados brutos de sequenciamento em insights clínicos acionáveis e protocolos de pesquisa automatizados.

## Examples
```
**1. Interpretação de Variantes Genéticas (ACMG/AMP):**
```
Você é um bioinformata especializado em classificação de variantes genéticas. Analise a seguinte variante no gene BRCA1: c.5266dupC (p.Gln1756Profs*10).
1. Classifique a variante de acordo com as diretrizes ACMG/AMP (ex: Patogênica, Provavelmente Patogênica, VUS, etc.).
2. Justifique a classificação citando as evidências PVS1, PS1-4, PM1-6, PP1-5, BA1, BS1-4, BP1-6 relevantes.
3. Descreva o fenótipo clínico associado e as implicações para o rastreamento de câncer.
```

**2. Design de Experimento CRISPR Automatizado:**
```
Assuma o papel de um engenheiro de genoma utilizando o sistema CRISPR-Cas12a.
Objetivo: Nocaute (knockout) do gene TGFβR1 em células de câncer de pulmão A549.
Tarefa: Gerar um protocolo completo.
1. Sugira 3 sequências de gRNA de alta eficácia e baixa atividade off-target, citando a ferramenta de predição utilizada.
2. Recomende o método de entrega (transfecção/transdução) mais adequado para a linhagem celular A549.
3. Descreva o protocolo de validação (ex: Western Blot, sequenciamento) e os primers de PCR necessários.
```

**3. Previsão de Resposta a Medicamentos (Farmacogenômica):**
```
Paciente: 65 anos, diagnóstico de câncer de cólon.
Genótipo: Variante *2/*3 no gene CYP2D6.
Medicamento: Irinotecano (metabolizado pela UGT1A1).
Prompt: Com base no genótipo CYP2D6 e no medicamento Irinotecano, avalie o risco de toxicidade e a eficácia.
1. Qual é o status de metabolizador do paciente para CYP2D6?
2. O Irinotecano é afetado por esta variante? Se não, qual gene é o principal determinante de toxicidade (UGT1A1)?
3. Forneça uma recomendação de ajuste de dose ou medicamento alternativo, citando as diretrizes CPIC.
```

**4. Geração de Relatório Clínico para Paciente:**
```
Você é um conselheiro genético.
Resultado: Mutação patogênica no gene MLH1, confirmando Síndrome de Lynch.
Público: Paciente leigo, nível de ansiedade alto.
Instrução: Crie um texto de 3 parágrafos que explique o resultado de forma clara, empática e não alarmante.
1. O que é a Síndrome de Lynch e o gene MLH1.
2. Quais são os riscos de câncer associados.
3. Quais são os próximos passos e opções de rastreamento (ex: colonoscopia anual).
```

**5. Análise de Dados Multi-Ômicos:**
```
Dados de entrada: Lista de 50 genes diferencialmente expressos (RNA-Seq) e 10 variantes somáticas (WES) em um tumor.
Prompt: Integre os dados de expressão e variantes para identificar as vias de sinalização mais impactadas.
1. Liste as 3 principais vias de sinalização (ex: KEGG, Reactome) que contêm tanto genes diferencialmente expressos quanto genes com variantes.
2. Sugira um alvo terapêutico (gene/proteína) que esteja no cruzamento dessas vias.
3. Explique a lógica biológica da sua sugestão.
```
```

## Best Practices
**1. Fornecer Contexto Genômico Detalhado:** Inclua o máximo de dados genômicos brutos (sequências, variantes VCF, dados de expressão) ou resumos estruturados (genes, mutações, fenótipos) diretamente no prompt. Quanto mais específico o contexto biológico, mais precisa será a resposta. **2. Definir o Papel (Role-Playing) e a Tarefa:** Comece o prompt atribuindo um papel especializado ao modelo (ex: "Você é um bioinformata especializado em oncogenômica" ou "Você é um conselheiro genético"). Em seguida, defina claramente a tarefa (ex: "Analisar a patogenicidade desta variante" ou "Gerar um protocolo de edição gênica"). **3. Utilizar RAG (Retrieval-Augmented Generation):** Para tarefas críticas, como aconselhamento genético ou design experimental, integre a técnica RAG. Isso envolve fornecer ao LLM documentos de referência (diretrizes clínicas, artigos científicos, protocolos de laboratório) para que ele baseie suas respostas em informações verificadas. **4. Especificar o Formato de Saída:** Peça a saída em um formato estruturado, como JSON, tabela Markdown ou um formato de relatório clínico específico, para facilitar a análise e a integração com outros sistemas. **5. Iteração e Refinamento:** A genômica é complexa. Use a saída do primeiro prompt como entrada para um segundo prompt, refinando a pergunta ou solicitando uma validação cruzada. Por exemplo, peça a análise de uma variante e, em seguida, peça a revisão dessa análise com base em um novo artigo.

## Use Cases
**1. Diagnóstico e Interpretação de Doenças Raras:** Análise rápida de painéis de sequenciamento para identificar variantes patogênicas e sugerir diagnósticos diferenciais. **2. Aconselhamento Genético Automatizado:** Geração de resumos de resultados e respostas a perguntas frequentes para pacientes, liberando o tempo dos conselheiros genéticos para casos mais complexos. **3. Otimização de Protocolos de Laboratório:** Design automatizado de primers, sondas e guias de CRISPR, reduzindo o tempo de planejamento experimental (como demonstrado pelo CRISPR-GPT [1]). **4. Farmacogenômica e Seleção de Dose:** Previsão da resposta individual a medicamentos com base no perfil genético do paciente, minimizando efeitos adversos e otimizando a eficácia. **5. Descoberta de Alvos Terapêuticos:** Integração de dados multi-ômicos (genômica, transcriptômica, proteômica) para identificar novos genes ou vias de sinalização para o desenvolvimento de medicamentos. **6. Análise de Risco Poligênico (PRS):** Interpretação de escores de risco poligênico para doenças comuns (ex: diabetes, doenças cardíacas) e tradução em recomendações de estilo de vida ou rastreamento.

## Pitfalls
**1. Alucinações e Imprecisão Clínica:** O maior risco é o LLM gerar informações clinicamente incorretas ou alucinar referências e dados genéticos. **Mitigação:** Sempre use RAG com fontes verificadas (ex: ClinVar, HGMD, diretrizes CPIC) e exija citações de artigos ou bases de dados. **2. Falta de Contexto Biológico:** Prompts muito curtos ou genéricos (ex: "O que este gene faz?") falham em fornecer o contexto necessário (ex: tipo de célula, tecido, condição da doença). **Mitigação:** Seja hiper-específico sobre o sistema biológico e o tipo de dado de entrada. **3. Viés e Desigualdade:** Os modelos de IA são treinados predominantemente em dados de populações de ascendência europeia. Isso pode levar a interpretações imprecisas ou enviesadas para variantes em populações sub-representadas. **Mitigação:** Inclua no prompt a etnia ou ancestralidade do paciente, se relevante, e peça ao modelo para considerar as limitações dos dados de referência. **4. Exposição de Dados Sensíveis (PHI):** A entrada de dados de pacientes (PHI - Protected Health Information) diretamente no prompt viola as regulamentações de privacidade (ex: HIPAA, LGPD). **Mitigação:** Sempre anonimize completamente os dados de entrada, usando IDs de variantes, sequências ou resumos de fenótipos, e nunca nomes, datas de nascimento ou números de prontuário. **5. Confusão entre Causa e Correlação:** O LLM pode confundir uma associação estatística (correlação) com uma relação causal biológica. **Mitigação:** Peça ao modelo para distinguir claramente entre evidências de associação e mecanismos biológicos comprovados.

## URL
[https://www.nature.com/articles/s41551-025-01463-z](https://www.nature.com/articles/s41551-025-01463-z)
