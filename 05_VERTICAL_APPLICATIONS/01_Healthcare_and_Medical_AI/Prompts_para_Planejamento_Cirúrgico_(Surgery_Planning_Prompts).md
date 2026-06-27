# Prompts para Planejamento Cirúrgico (Surgery Planning Prompts)

## Description
Prompts para Planejamento Cirúrgico são instruções altamente estruturadas e orientadas por diretrizes, projetadas para serem usadas com Modelos de Linguagem Grande (LLMs) em ambientes clínicos. O objetivo principal é fornecer ao LLM dados clínicos detalhados de um paciente (demografia, comorbidades, achados de imagem, biologia tumoral, estadiamento) e solicitar uma saída estruturada e auditável que auxilie na tomada de decisão cirúrgica. Essa técnica visa padronizar o processo de planejamento, melhorar a segurança do paciente e otimizar os resultados cirúrgicos, muitas vezes incorporando a técnica RAG (Retrieval-Augmented Generation) para garantir que as sugestões do LLM sejam baseadas nas diretrizes clínicas mais recentes e específicas da área.

## Examples
```
1. **Prompt para Decisão de Neoadjuvância em Câncer de Mama**
```
**Papel:** Você é um assistente de IA para a Equipe Multidisciplinar de Câncer de Mama, especializado em oncologia cirúrgica.
**Contexto:** Paciente feminina, 55 anos. Carcinoma ductal invasivo, grau 3. Estadiamento clínico T2N1M0. Receptor de estrogênio (RE) positivo (90%), Receptor de progesterona (RP) positivo (80%), HER2 negativo (FISH não amplificado). Ki-67: 45%. Sem comorbidades significativas. A ultrassonografia mostra um tumor de 3,5 cm e um linfonodo axilar suspeito.
**Instrução:** Com base nas diretrizes NCCN/ESMO mais recentes, avalie a adequação da terapia neoadjuvante. Forneça a resposta no formato JSON com os campos: `Recomendação_Neoadjuvância` (Sim/Não), `Justificativa_Principal`, `Regime_Sugerido`.
```

2. **Prompt para Tipo de Reconstrução Pós-Mastectomia**
```
**Papel:** Você é um especialista em reconstrução mamária assistida por IA.
**Contexto:** Paciente de 48 anos, submetida a mastectomia total. Histórico de tabagismo (parou há 6 meses). IMC: 32. Irradiação prévia: Não. Preferência da paciente: Reconstrução imediata. Disponibilidade de tecido abdominal: Adequada. Condição vascular: Boa.
**Instrução:** Sugira o tipo de reconstrução mamária (Implante Direto, Expansor + Implante, TRAM, DIEP, Latissimus Dorsi) e avalie a viabilidade de Implante Direto. Forneça a saída em uma tabela Markdown com as colunas: `Opção_Recomendada`, `Viabilidade_Implante_Direto` (Alta/Média/Baixa), `Fatores_de_Risco`.
```

3. **Prompt para Plano de Ação Cirúrgica em Cirurgia Colorretal**
```
**Papel:** Você é um assistente de planejamento cirúrgico para cirurgia colorretal.
**Contexto:** Paciente masculino, 68 anos. Câncer retal médio (10 cm da margem anal). Estadiamento: cT3N1M0. Tratamento neoadjuvante: Quimiorradioterapia completa (CRT). Resposta: Resposta clínica completa (cCR) após CRT. Comorbidades: Hipertensão controlada.
**Instrução:** Desenvolva um plano cirúrgico inicial. Inclua: `Procedimento_Sugerido` (e.g., Ressecção Anterior Baixa, Abdominoperineal), `Abordagem` (Laparoscópica/Robótica/Aberta), `Necessidade_de_Estoma` (Temporário/Permanente/Não), e `Pontos_Críticos_Intraoperatórios`.
```

4. **Prompt para Avaliação de Risco Pré-Operatório**
```
**Papel:** Você é um especialista em avaliação de risco pré-operatório (ASA/POSSUM).
**Contexto:** Paciente de 75 anos. Cirurgia proposta: Reparo de aneurisma da aorta abdominal eletivo. Histórico: Infarto do miocárdio há 3 anos, Insuficiência Cardíaca Congestiva (Classe II NYHA), Diabetes Mellitus tipo 2 (HbA1c 8.5). Exames: Creatinina 1.8 mg/dL, Eletrocardiograma com Fibrilação Atrial.
**Instrução:** Calcule a pontuação de risco ASA e POSSUM (fisiológico e operatório). Liste os 3 principais riscos perioperatórios e sugira uma otimização pré-operatória específica para cada risco.
```

5. **Prompt para Documentação Cirúrgica Estruturada**
```
**Papel:** Você é um gerador de relatórios cirúrgicos estruturados.
**Contexto:** [Cole o texto integral do ditado cirúrgico ou notas intraoperatórias].
**Instrução:** Converta o texto bruto em um relatório cirúrgico estruturado em formato XML ou JSON, com os seguintes campos obrigatórios: `Data_Cirurgia`, `Cirurgião_Principal`, `Assistentes`, `Diagnóstico_Pré`, `Procedimento_Realizado`, `Achados_Intraoperatórios`, `Complicações` (Sim/Não), `Perda_Sanguínea` (mL), `Drenos` (Tipo e Localização).
```
```

## Best Practices
**1. Estrutura Rígida (Role, Context, Instruction, Format):** Sempre defina o **Papel** (e.g., cirurgião oncológico, radiologista intervencionista), forneça o **Contexto** clínico completo (dados do paciente), defina a **Instrução** clara (a pergunta cirúrgica) e exija um **Formato** de saída estruturado (JSON, Tabela Markdown).
**2. Incorporação de Diretrizes (RAG):** Para a precisão clínica, o LLM deve ser aumentado com as diretrizes mais recentes (e.g., NCCN, NICE, ESMO). O prompt deve referenciar implicitamente ou explicitamente a necessidade de aderência a essas fontes.
**3. Especificidade e Quantificação:** Use termos clínicos precisos e dados quantificáveis (e.g., tamanho do tumor em cm, Ki-67 em %, escore ASA). Evite ambiguidades.
**4. Validação e Auditoria:** Projete o prompt para que a saída seja facilmente auditável e comparável com o padrão-ouro (decisão da equipe multidisciplinar), facilitando a validação do modelo.
**5. Foco na Segurança:** Inclua sempre uma seção de 'Fatores de Risco' ou 'Pontos Críticos' na saída solicitada para garantir que o LLM considere a segurança do paciente.

## Use Cases
1. **Suporte à Decisão Pré-Operatória:** Auxiliar cirurgiões menos experientes ou em casos complexos a alinhar o plano cirúrgico com as diretrizes atuais.
2. **Padronização do Planejamento:** Garantir que todos os casos sejam avaliados com os mesmos critérios, reduzindo a variabilidade na prática clínica.
3. **Educação e Treinamento:** Usar o LLM para gerar planos de tratamento para casos simulados, servindo como ferramenta de aprendizado para residentes.
4. **Geração de Documentação:** Converter notas de voz ou texto livre em relatórios cirúrgicos estruturados e prontos para o prontuário eletrônico.
5. **Avaliação de Risco:** Calcular escores de risco (e.g., ASA, POSSUM) e identificar as principais comorbidades que precisam de otimização pré-operatória.

## Pitfalls
1. **Alucinações Clínicas:** O LLM pode gerar recomendações plausíveis, mas clinicamente incorretas ou não suportadas por evidências. **Mitigação:** Uso obrigatório de RAG e validação humana.
2. **Viés de Dados:** Se o modelo for treinado em dados de uma única instituição ou demografia, ele pode sugerir planos subótimos para populações diferentes. **Mitigação:** Auditoria de equidade e diversidade nos dados de treinamento.
3. **Falta de Contexto Visual:** O planejamento cirúrgico é inerentemente visual (imagens, anatomia). O LLM, por si só, não pode interpretar imagens. **Mitigação:** O prompt deve incluir dados estruturados extraídos de imagens por um radiologista ou outro modelo de IA.
4. **Dependência Excessiva:** Confiar cegamente na saída do LLM sem o julgamento clínico humano. **Mitigação:** O LLM deve ser posicionado como um 'assistente', não como um 'decisor'.
5. **Inconsistência de Formato:** Se o prompt não for rígido o suficiente, o LLM pode retornar o plano em um formato de texto livre, dificultando a integração com sistemas de prontuário eletrônico.

## URL
[https://pmc.ncbi.nlm.nih.gov/articles/PMC12588214/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12588214/)
