# Engenharia de Prompts para Aplicações Biomédicas

## Description
A Engenharia de Prompts para Aplicações Biomédicas refere-se à arte e ciência de criar instruções (prompts) precisas e contextuais para **Grandes Modelos de Linguagem (LLMs)**, como GPT-4 e Claude, a fim de automatizar ou semi-automatizar tarefas complexas no campo da Engenharia Biomédica e da pesquisa clínica. O foco principal está na **síntese de evidências** e na **triagem de literatura** para revisões sistemáticas e meta-análises, onde os LLMs são usados para classificar milhares de resumos de artigos científicos.

Essa técnica se baseia na conversão de critérios de elegibilidade clínica estruturados, como o framework **PICO (População, Intervenção, Comparação, Desfecho)**, em instruções acionáveis para o modelo. O objetivo é reduzir a carga de trabalho manual, acelerar a síntese de evidências e manter o rigor metodológico, permitindo que os pesquisadores se concentrem na avaliação crítica de textos completos. A eficácia depende da clareza, da especificidade e da estratégia de prompt adotada (e.g., *zero-shot*, *few-shot*, *soft* ou *strict*).

## Examples
```
## Exemplo 1: Triagem de Revisão Sistemática (Prompt Soft - Alta Sensibilidade)

**Cenário:** Triagem inicial de resumos para uma revisão sistemática sobre o uso de **Nanopartículas de Ouro (AuNPs)** para entrega de medicamentos em oncologia.

**Prompt:**
```
Você é um assistente de triagem de literatura. Sua tarefa é classificar o resumo fornecido como 'ACEITAR' ou 'REJEITAR' para uma revisão sistemática.

Critérios de Inclusão (PICO):
1. População (P): Estudos in vivo (animais ou humanos) ou in vitro que abordem células cancerosas.
2. Intervenção (I): Uso de Nanopartículas de Ouro (AuNPs) como sistema de entrega de medicamentos.
3. Comparação (C): Qualquer controle (e.g., medicamento livre, outro nanocarreador, placebo).
4. Desfecho (O): Avaliação da eficácia antitumoral ou toxicidade.
5. Desenho do Estudo: Artigo original (não revisão, editorial ou carta).

Regra de Decisão (Soft): Se o resumo não contradisser explicitamente qualquer um dos critérios acima, responda 'ACEITAR'. Responda 'REJEITAR' apenas se houver uma violação clara (e.g., estudo focado em doenças cardiovasculares, uso de nanopartículas de prata).

Resumo a ser avaliado: [INSERIR RESUMO AQUI]

Decisão:
```

## Exemplo 2: Triagem de Revisão Sistemática (Prompt Strict - Alta Especificidade)

**Cenário:** Triagem final de resumos para uma revisão sistemática focada em **Ensaios Clínicos Randomizados (ECR)** sobre o uso de **Stents Bioabsorvíveis (BVS)** em pacientes diabéticos.

**Prompt:**
```
Você é um revisor especialista. Classifique o resumo como 'ACEITAR' ou 'REJEITAR'.

Critérios de Inclusão (PICO/S):
1. População (P): Pacientes humanos, adultos (≥18 anos), com Diabetes Mellitus Tipo 2.
2. Intervenção (I): Implante de Stents Vasculares Bioabsorvíveis (BVS).
3. Comparação (C): Stents Metálicos Farmacológicos (DES).
4. Desfecho (O): Taxas de Reestenose ou Trombose de Stent em 12 meses.
5. Desenho do Estudo (S): **DEVE** ser um Ensaio Clínico Randomizado (ECR) explicitamente declarado.

Regra de Decisão (Strict): Responda 'ACEITAR' apenas se o resumo **mencionar explicitamente** todos os 5 critérios. Se qualquer critério estiver ausente, ambíguo ou não for explicitamente um ECR, responda 'REJEITAR'.

Resumo a ser avaliado: [INSERIR RESUMO AQUI]

Decisão:
```

## Exemplo 3: Otimização de Design de Dispositivo Médico

**Cenário:** Otimizar o design de um novo sensor vestível para monitoramento contínuo de glicose (CGM).

**Prompt:**
```
Atue como um Engenheiro Biomédico especializado em usabilidade e design de dispositivos vestíveis.

Tarefa: Analise os seguintes requisitos de design e sugira 3 melhorias de design para o sensor CGM, focando em conforto, durabilidade e precisão do sinal.

Requisitos Atuais:
- Material: Polímero de silicone rígido.
- Tamanho: 30mm x 15mm x 5mm.
- Local de Aplicação: Braço superior.
- Vida Útil: 7 dias.
- Conectividade: Bluetooth Low Energy (BLE).

Sugestões de Melhoria (Formato: 1. [Melhoria] - [Justificativa de Engenharia]):
```

## Exemplo 4: Análise de Dados e Bioinformática

**Cenário:** Interpretar um conjunto de dados de expressão gênica (RNA-seq) em um modelo de doença neurodegenerativa.

**Prompt:**
```
Você é um Bioinformata. Recebeu uma lista de 10 genes diferencialmente expressos (DEGs) em neurônios de pacientes com Alzheimer em comparação com controles.

Lista de DEGs (Up-regulated): APP, PSEN1, APOE, BACE1, MAPT, TREM2, CD33, PICALM, CLU, SORL1.

Tarefa:
1. Descreva a função biológica de 3 DEGs críticos (escolha os mais relevantes para a patogênese do Alzheimer).
2. Sugira uma via de sinalização (pathway) que conecte pelo menos 2 desses genes.
3. Proponha um alvo terapêutico (proteína ou gene) com base nesta análise.

Resposta estruturada:
```

## Exemplo 5: Engenharia de Tecidos e Biomateriais

**Cenário:** Desenvolver um novo arcabouço (scaffold) para regeneração de cartilagem.

**Prompt:**
```
Atue como um Engenheiro de Materiais Biomédicos.

Tarefa: Proponha um arcabouço de matriz extracelular (ECM) para regeneração de cartilagem articular, considerando as propriedades mecânicas e a biocompatibilidade.

Proposta de Arcabouço:
1. Material Polimérico (e.g., PCL, PLA, ou natural): [Escolha e Justificativa]
2. Estrutura (e.g., porosa, nanofibras): [Escolha e Justificativa]
3. Fator de Crescimento (e.g., TGF-β, BMP-2): [Escolha e Justificativa]
4. Método de Fabricação (e.g., Eletrofiação, Impressão 3D): [Escolha e Justificativa]
```

## Exemplo 6: Apoio à Decisão Clínica (Interpretação de Imagem)

**Cenário:** Auxiliar um radiologista a estruturar um relatório de ressonância magnética (RM) cardíaca.

**Prompt:**
```
Você é um sistema de apoio à decisão clínica especializado em Engenharia de Imagem Médica.

Tarefa: Com base nos achados de uma RM cardíaca, estruture um relatório conciso, focando na quantificação e na relevância clínica dos achados.

Achados da RM:
- Fração de Ejeção do Ventrículo Esquerdo (FEVE): 35% (gravemente reduzida).
- Realce Tardio com Gadolínio (LGE): Padrão subendocárdico em parede anterior e septo (sugerindo infarto prévio).
- Volume Diastólico Final do VE (VDV-VE): 220 ml (dilatação significativa).

Estrutura do Relatório:
1. Quantificação (Valores Chave):
2. Impressão Diagnóstica (Engenharia): [Análise da função e geometria ventricular]
3. Relevância Clínica: [Implicações para o prognóstico e tratamento]
```
```

## Best Practices
1. **Definir Critérios PICO Claros:** Traduzir os elementos PICO (População, Intervenção, Comparação, Desfecho) em linguagem inequívoca e específica para o LLM. Evitar termos vagos e usar palavras-chave exatas (e.g., "pacientes adultos (≥18 anos)" em vez de "adultos").
2. **Escolher a Abordagem de Prompt Adequada:**
    *   **Zero-Shot:** Para critérios simples e diretos, sem exemplos.
    *   **Few-Shot:** Incluir 2-3 exemplos rotulados (ACCEPT/REJECT) para orientar o modelo em casos ambíguos ou complexos, melhorando a precisão.
3. **Estratégia Soft vs. Strict:**
    *   **Prompt Soft (Alta Sensibilidade/Recall):** Instruir o modelo a **ACEITAR** a menos que um critério seja **explicitamente violado**. Ideal para a primeira triagem, minimizando falsos negativos.
    *   **Prompt Strict (Alta Especificidade/Precisão):** Instruir o modelo a **REJEITAR** a menos que **todos** os critérios sejam **explicitamente mencionados**. Reduz falsos positivos, mas pode aumentar falsos negativos.
4. **Formato de Saída Consistente:** Exigir uma resposta curta e padronizada (e.g., apenas "ACCEPT" ou "REJECT") para facilitar o processamento automatizado.
5. **Teste e Refinamento Iterativo:** Testar o prompt em um pequeno conjunto de validação e refinar as instruções com base nas métricas de desempenho (Precisão, Recall, F1-score).
6. **Supervisão Humana Indispensável:** Manter a supervisão humana para casos limítrofes, ambíguos ou para verificar a precisão das decisões do LLM, mitigando alucinações e vieses.

## Use Cases
*   **Revisões Sistemáticas e Meta-análises:** Triagem inicial de milhares de resumos para identificar artigos relevantes com base nos critérios PICO.
*   **Síntese Rápida de Evidências:** Geração de resumos e extração de dados-chave (e.g., desfechos, tamanho da amostra) de artigos biomédicos.
*   **Classificação de Literatura:** Categorização de artigos científicos por tipo de estudo (RCT, coorte, caso-controle) ou por tema específico (e.g., biomarcadores, dispositivos médicos).
*   **Geração de Documentação Regulatória:** Auxílio na redação de protocolos de estudo, relatórios de segurança e documentos de submissão regulatória, garantindo a conformidade com diretrizes específicas.
*   **Apoio à Decisão Clínica:** Criação de prompts para extrair informações de prontuários eletrônicos (EHRs) para auxiliar no diagnóstico ou na seleção de tratamento (com atenção à privacidade de dados).

## Pitfalls
*   **Prompts Excessivamente Longos ou Ambíguos:** Instruções complexas ou contraditórias que diluem o foco do modelo, levando a respostas inconsistentes.
*   **Alucinação:** O LLM infere ou "inventa" informações (e.g., rotular um estudo como "randomizado" quando o resumo não o menciona explicitamente).
*   **Violação de Privacidade de Dados:** O uso de LLMs baseados em nuvem (APIs) para triagem de dados sensíveis (e.g., prontuários) pode violar regulamentações de privacidade (e.g., HIPAA, LGPD).
*   **Viés de Treinamento:** Vieses nos dados de treinamento do LLM podem levar à sub-representação ou classificação incorreta de estudos de certas populações ou regiões.
*   **Falta de Transparência:** A natureza de "caixa preta" dos LLMs dificulta o rastreamento do processo de decisão, o que é problemático para o rigor científico.

## URL
[https://www.mdpi.com/2673-7426/5/1/15](https://www.mdpi.com/2673-7426/5/1/15)
