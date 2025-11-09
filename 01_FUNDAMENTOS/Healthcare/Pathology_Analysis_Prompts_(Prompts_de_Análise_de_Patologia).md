# Pathology Analysis Prompts (Prompts de Análise de Patologia)

## Description
**Prompts de Análise de Patologia** referem-se à aplicação de técnicas de **Prompt Engineering** para interagir com Large Language Models (LLMs) e modelos multimodais (VLM/LLM) com o objetivo de analisar, estruturar e interpretar dados complexos de patologia. O foco principal é a **extração de informações estruturadas** de relatórios de patologia em texto livre (histopatologia, citopatologia, relatórios cirúrgicos) e a **análise de imagens** de lâminas inteiras (Whole Slide Images - WSIs) em patologia digital [1] [2].

Essa técnica é crucial porque a maioria dos relatórios de patologia é gerada em formato de texto livre, o que dificulta a análise em larga escala, a pesquisa e a integração em sistemas de IA. Os prompts são projetados para guiar o LLM a identificar e extrair com precisão dados clínicos críticos, como tipo de tumor, grau histológico, status de receptores (ER, PR, HER2), margens cirúrgicas e estadiamento TNM, convertendo o texto não estruturado em um formato de dados consumível (geralmente JSON ou tabela) [3] [4].

A pesquisa recente (2023-2025) demonstra que LLMs de código aberto, quando bem promptados, podem igualar a precisão de modelos proprietários como o GPT-4 na extração de dados de patologia, o que tem implicações significativas para a privacidade e o custo em ambientes de saúde [1]. A engenharia de prompt adaptativa e a especificação de formato de saída são elementos-chave para o sucesso [3].

## Examples
```
**1. Extração Estruturada de Relatório de Câncer de Mama (Zero-Shot)**

```
Você é um analista de dados de patologia. Sua tarefa é extrair as seguintes informações do relatório de patologia fornecido e formatá-las estritamente como um objeto JSON.

Campos a extrair:
1.  `tumor_type`: (Ex: Carcinoma Ductal Invasivo, Carcinoma Lobular Invasivo)
2.  `histological_grade`: (Ex: Grau 1, Grau 2, Grau 3)
3.  `er_status`: (Ex: Positivo, Negativo, Indeterminado)
4.  `pr_status`: (Ex: Positivo, Negativo, Indeterminado)
5.  `her2_status`: (Ex: Positivo, Negativo, Equívoco)
6.  `surgical_margins`: (Ex: Livres, Comprometidas, Não avaliadas)

Relatório de Patologia:
[INSERIR TEXTO COMPLETO DO RELATÓRIO AQUI]

Saída JSON:
```

**2. Análise de Estadiamento TNM (Chain-of-Thought)**

```
Aja como um oncologista patologista. Analise o relatório de patologia a seguir e determine o estadiamento TNM (Tumor, Linfonodo, Metástase) com base nas diretrizes AJCC mais recentes.

Passos de Raciocínio (CoT):
1.  Identifique o tamanho do tumor primário (T).
2.  Identifique o número de linfonodos positivos e o status do linfonodo (N).
3.  Identifique a presença ou ausência de metástase à distância (M).
4.  Combine os achados para determinar o estadiamento TNM final.

Relatório: [INSERIR RELATÓRIO]

Estadiamento TNM: [RESPOSTA FINAL]
Raciocínio: [JUSTIFICATIVA PASSO A PASSO]
```

**3. Classificação de Achados de Patologia Digital (Multimodal)**

```
[PROMPT DE SISTEMA: Você está recebendo uma imagem WSI (Whole Slide Image) de uma biópsia renal e o texto do laudo.]

Instrução: Compare a imagem WSI com o texto do laudo. Identifique se há discrepância entre o achado morfológico na imagem (ex: glomeruloesclerose segmentar) e o diagnóstico textual.

Achado Primário na Imagem: [DESCREVA O ACHADO VISUALMENTE DOMINANTE]
Diagnóstico Textual: [INSERIR TEXTO DO LAUDO]

Pergunta: O diagnóstico textual está completo e consistente com o achado primário na imagem? Se não, qual é a discrepância?

Resposta:
```

**4. Extração de Dimensões e Focality (Prompt Específico)**

```
Extraia as dimensões do tumor e a focality do relatório. Responda estritamente com os campos 'max_dimension_mm' e 'focality'. Se a informação não estiver presente, use 'not specified'.

Relatório: "O espécime mede 5.0 x 3.0 x 2.0 cm. Há um foco invasivo medindo 12 mm no maior eixo. Não há outros focos identificados."

JSON de Saída:
{
  "max_dimension_mm": [VALOR],
  "focality": [Single focus/Multiple foci/not specified]
}
```

**5. Geração de Resumo Clínico para Prontuário Eletrônico**

```
Você é um assistente de documentação clínica. Crie um resumo conciso (máximo 3 frases) do relatório de patologia a seguir, destacando apenas o diagnóstico final, o grau e o status do receptor.

Relatório: [INSERIR RELATÓRIO]

Resumo Clínico:
```
```

## Best Practices
**1. Especificação de Formato de Saída (JSON/Tabela):** Sempre instrua o LLM a retornar a informação em um formato estruturado (JSON, tabela, YAML) para facilitar o processamento e a integração em sistemas de informação laboratorial (LIS) ou bancos de dados de pesquisa [1] [3].

**2. Prompting Zero-Shot e Few-Shot:** Para tarefas de extração de dados, comece com o **Zero-Shot Prompting** (apenas instruções) e, se a precisão for insuficiente, use o **Few-Shot Prompting** (instruções + 1 a 5 exemplos de relatórios e suas extrações corretas) para refinar o desempenho [2] [4].

**3. Cadeia de Pensamento (Chain-of-Thought - CoT):** Para diagnósticos complexos ou análise de achados múltiplos, peça ao LLM para "pensar em voz alta" ou justificar sua conclusão antes de fornecer a resposta final. Isso melhora a rastreabilidade e a precisão [5].

**4. Definição Clara de Campos de Extração:** Defina explicitamente os campos de dados a serem extraídos (ex: Tipo de Tumor, Grau Histológico, Status de Receptor, Margens Cirúrgicas) e os valores permitidos (ex: "Positivo/Negativo", "Grau 1/2/3") [3].

**5. Iteração e Refinamento Adaptativo:** O processo de engenharia de prompt deve ser iterativo. Ajuste o prompt dinamicamente com base no feedback de desempenho, especialmente para lidar com a terminologia idiossincrática de diferentes patologistas e instituições [3].

**6. Contextualização e Papel:** Comece o prompt definindo o papel do LLM (ex: "Você é um analista de patologia oncológica experiente") e o contexto da tarefa (ex: "Sua tarefa é extrair dados críticos de um relatório de patologia de câncer de mama") [3].

## Use Cases
**1. Extração Automatizada de Dados para Pesquisa:** Converter milhares de relatórios de patologia em texto livre em um banco de dados estruturado para estudos epidemiológicos, ensaios clínicos e desenvolvimento de modelos de IA [1].

**2. População de Registros Eletrônicos de Saúde (EHR):** Automatizar a inserção de dados críticos (ex: estadiamento, grau, status de biomarcadores) de relatórios de patologia diretamente no prontuário eletrônico do paciente, reduzindo o erro manual e acelerando o fluxo de trabalho clínico [3].

**3. Desenvolvimento de Modelos de IA em Patologia Digital:** Usar prompts para gerar rótulos (labels) de alta qualidade a partir de relatórios textuais, que são então usados como "verdade fundamental" (ground truth) para treinar modelos de Visão Computacional que analisam imagens de lâminas inteiras (WSIs) [2].

**4. Geração de Resumos Clínicos:** Criar resumos padronizados e concisos de relatórios complexos para facilitar a comunicação entre patologistas, oncologistas e outros especialistas clínicos [5].

**5. Controle de Qualidade e Auditoria:** Usar prompts para verificar a consistência e a completude dos relatórios, garantindo que todos os campos obrigatórios (ex: de acordo com os protocolos ICCR) tenham sido abordados [3].

## Pitfalls
**1. Alucinações e Imprecisão Clínica:** O LLM pode gerar informações clinicamente incorretas ou "alucinar" dados que não estão no relatório, especialmente em textos ambíguos ou com erros de digitação. A validação humana é indispensável [1].

**2. Falha na Extração de Campos Ausentes:** Se um campo de dados (ex: status de um receptor específico) não estiver explicitamente mencionado no relatório, o LLM pode falhar em retornar o valor "Não especificado" e, em vez disso, tentar inferir ou deixar o campo vazio, quebrando o formato estruturado [3].

**3. Sensibilidade à Variação de Linguagem:** A terminologia de patologia varia significativamente entre diferentes patologistas e instituições (ex: abreviações, jargões). Prompts não adaptativos podem ter desempenho ruim em relatórios com estilo de escrita idiossincrático [3].

**4. Quebra de Formato de Saída:** A instrução para retornar um formato estruturado (JSON, tabela) pode ser ignorada pelo LLM se o prompt for muito longo ou se a complexidade da extração for alta, resultando em texto livre que invalida a automação [4].

**5. Prejuízo de Dados (Data Bias):** Se o modelo base foi treinado predominantemente em relatórios de uma única etnia, tipo de câncer ou idioma (ex: inglês), ele pode ter um desempenho inferior ao analisar relatórios de populações ou idiomas diferentes (ex: português) [1].

## URL
[https://www.nature.com/articles/s43856-025-00808-8](https://www.nature.com/articles/s43856-025-00808-8)
