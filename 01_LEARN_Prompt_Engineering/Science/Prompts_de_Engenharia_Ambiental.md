# Prompts de Engenharia Ambiental

## Description
A Engenharia de Prompts para o setor Ambiental é a prática de criar instruções (prompts) altamente específicas e contextuais para modelos de linguagem (LLMs) com o objetivo de resolver problemas complexos de Engenharia Ambiental, como gestão de recursos hídricos, tratamento de resíduos, avaliação de impacto ambiental (AIA) e modelagem de poluição. Esta técnica é crucial para adaptar modelos de fundação genéricos (como GPT-4 ou Gemini) ao vocabulário técnico, regulamentações específicas e dados científicos do domínio ambiental, transformando-os em assistentes especializados, ou "WaterGPTs" [1]. O foco está em fornecer o máximo de contexto técnico, dados de entrada e restrições regulatórias para garantir que as saídas sejam precisas, confiáveis e aplicáveis no mundo real. É uma área em rápida evolução, com ênfase na redução do impacto ambiental da própria IA (Green Prompt Engineering) e na criação de modelos de linguagem de domínio adaptado [3].

## Examples
```
**1. Avaliação de Impacto Ambiental (AIA)**
`Aja como um consultor ambiental sênior. Analise o projeto de construção de uma rodovia de 50 km na Mata Atlântica (coordenadas: [inserir dados]). Com base na Resolução CONAMA 001/86, gere uma lista detalhada dos 10 principais impactos ambientais negativos esperados (diretos e indiretos) e sugira 3 medidas mitigadoras específicas para cada um. Formate a saída em uma tabela Markdown.`

**2. Otimização de Tratamento de Água/Efluentes**
`Você é um engenheiro de processos de uma Estação de Tratamento de Água (ETA). O efluente de entrada tem as seguintes características: DBO = 250 mg/L, DQO = 500 mg/L, pH = 6.5, Vazão = 1000 m³/dia. O padrão de descarte exige DBO < 30 mg/L. Descreva o processo de tratamento biológico mais eficiente (lodos ativados convencional) para atingir essa meta. Calcule o volume mínimo do reator biológico e a idade do lodo (SRT) necessária, explicando a lógica por trás dos cálculos.`

**3. Modelagem de Dispersão de Poluentes Atmosféricos**
`Aja como um modelador de qualidade do ar. Um novo emissor de SO2 será instalado em [Localização]. A taxa de emissão é de 50 g/s. Use o modelo Gaussiano de Pluma (ou cite um modelo mais adequado) para estimar a concentração máxima de SO2 a 1 km de distância, sob condições de estabilidade atmosférica D (moderadamente instável) e velocidade do vento de 5 m/s. Liste as suposições feitas e o passo a passo do cálculo.`

**4. Análise de Risco e Conformidade Regulatória**
`Você é um especialista em EHS (Meio Ambiente, Saúde e Segurança). Analise o anexo da NBR 10004 (Classificação de Resíduos Sólidos) e classifique o resíduo "borra oleosa de manutenção de máquinas" (código de origem [inserir código]). Justifique a classificação (Classe I ou II) e sugira o método de destinação final mais seguro e legalmente aceito no Brasil.`

**5. Síntese de Literatura Científica**
`Revise os 5 artigos mais recentes (2023-2025) sobre o uso de nanotecnologia para remoção de microplásticos em águas residuais. Sintetize os principais achados, as limitações da tecnologia e o custo-benefício em uma análise de 500 palavras. Inclua as referências no formato ABNT.`

**6. Design de Infraestrutura Verde**
`Aja como um engenheiro de drenagem urbana. Projete um jardim de chuva (rain garden) para uma área de estacionamento de 500 m² em uma cidade com precipitação média anual de 1500 mm. Descreva as camadas do solo, a seleção de espécies vegetais nativas (cite 3 exemplos) e o dimensionamento da área de infiltração para reter 80% do volume de escoamento de uma chuva de projeto de 25 mm.`
```

## Best Practices
**1. Especificidade e Contexto:** Sempre defina o **papel** do LLM (ex: "Aja como um engenheiro ambiental sênior") e forneça o **contexto** detalhado (ex: tipo de efluente, regulamentação local, dados de entrada).
**2. Grounding (Aterramento):** Use a técnica de Geração Aumentada por Recuperação (RAG) ou inclua dados e documentos de referência (normas, relatórios, dados de sensores) diretamente no prompt para evitar alucinações e garantir a aderência a fatos e regulamentos técnicos [1].
**3. Cadeia de Pensamento (CoT):** Para problemas complexos (ex: modelagem de dispersão de poluentes), instrua o LLM a detalhar o processo passo a passo antes de fornecer a resposta final. Ex: "Primeiro, liste as variáveis de entrada. Segundo, descreva o modelo matemático. Terceiro, aplique os dados e forneça o resultado."
**4. Formato de Saída Estruturado:** Peça a saída em formatos fáceis de processar, como tabelas Markdown, JSON ou código Python, para facilitar a integração com outras ferramentas de engenharia.
**5. Validação Humana:** Nunca confie cegamente nas saídas para decisões críticas. Use o LLM como um assistente para rascunhos, análises preliminares ou síntese de documentos, mas a validação final deve ser feita por um especialista humano [2].

## Use Cases
**1. Gestão de Recursos Hídricos:** Otimização de redes de distribuição de água, previsão de qualidade da água em rios e reservatórios, e design de sistemas de irrigação eficientes.
**2. Tratamento de Água e Efluentes:** Simulação de processos de tratamento (ex: aeração, sedimentação), cálculo de dosagem de produtos químicos e diagnóstico de falhas operacionais em Estações de Tratamento (ETAs e ETEs) [1].
**3. Avaliação de Impacto Ambiental (AIA) e Licenciamento:** Geração de rascunhos de relatórios de impacto, análise de conformidade regulatória e identificação de riscos ambientais em novos projetos.
**4. Modelagem e Previsão de Poluição:** Simulação da dispersão de poluentes atmosféricos ou aquáticos, previsão de eventos de poluição e análise de risco toxicológico.
**5. Engenharia de Resíduos Sólidos:** Otimização de rotas de coleta, classificação de resíduos perigosos e design conceitual de aterros sanitários ou usinas de reciclagem.
**6. Sustentabilidade e EHS (Meio Ambiente, Saúde e Segurança):** Elaboração de políticas de sustentabilidade corporativa, criação de procedimentos de segurança e treinamento de funcionários sobre normas ambientais [2].

## Pitfalls
**1. Alucinações Técnicas:** O LLM pode inventar normas, valores de referência ou procedimentos de cálculo que parecem corretos, mas são factualmente incorretos ou desatualizados. **Contramedida:** Sempre use a técnica de Grounding (RAG) e valide a saída com fontes oficiais.
**2. Falta de Contexto Geográfico/Regulatório:** O modelo pode sugerir soluções que não são aplicáveis devido a regulamentações locais (ex: CONAMA no Brasil, EPA nos EUA) ou condições climáticas/geológicas específicas. **Contramedida:** Inclua explicitamente no prompt o país, estado, município e a norma regulatória aplicável.
**3. Simplificação Excessiva:** Problemas de engenharia (ex: modelagem hidrológica, cinética de reatores) são complexos e o LLM pode simplificar demais, ignorando variáveis críticas. **Contramedida:** Use a Cadeia de Pensamento (CoT) e exija a listagem de todas as premissas e variáveis consideradas.
**4. Viés de Dados:** Se o modelo foi treinado predominantemente com dados de países desenvolvidos, ele pode sugerir tecnologias ou práticas inviáveis economicamente ou inadequadas para o contexto de países em desenvolvimento. **Contramedida:** Peça soluções adaptadas a restrições orçamentárias ou de infraestrutura específicas.
**5. Confusão de Unidades:** O modelo pode misturar unidades de medida (ex: mg/L com ppm, m³/s com L/s). **Contramedida:** Especifique o sistema de unidades desejado (ex: SI) e peça para o modelo confirmar as unidades na saída.

## URL
[https://www.nature.com/articles/s41545-025-00509-8](https://www.nature.com/articles/s41545-025-00509-8)
