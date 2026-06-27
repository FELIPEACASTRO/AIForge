# Topic Modeling Prompts

## Description
**Prompts de Modelagem de Tópicos** referem-se à técnica de utilizar Large Language Models (LLMs) para realizar a tarefa de **Modelagem de Tópicos** (Topic Modeling), tradicionalmente feita por algoritmos estatísticos como LDA (Latent Dirichlet Allocation) ou modelos neurais como BERTopic. A abordagem baseada em prompts, popularizada por frameworks como o **TopicGPT** [1], reformula a modelagem de tópicos como uma série de tarefas de geração e classificação de linguagem natural.

Em vez de inferir distribuições de palavras e documentos, o LLM é instruído a:
1.  **Gerar Tópicos:** Analisar um *corpus* de documentos e criar rótulos de tópicos **generalizáveis** e descrições concisas.
2.  **Refinar Tópicos:** Mesclar tópicos duplicados ou irrelevantes e organizar a hierarquia.
3.  **Atribuir Tópicos:** Classificar novos documentos nos tópicos gerados, fornecendo uma **citação de suporte** (Grounding) para justificar a atribuição.

Essa técnica se destaca por gerar tópicos **mais coerentes e interpretáveis** por humanos (coerência semântica) do que os métodos tradicionais, além de permitir a modelagem **Zero-Shot** (sem a necessidade de treinamento em um *corpus* específico) [2]. A principal vantagem é a capacidade de aproveitar o vasto conhecimento de mundo do LLM para criar rótulos de tópicos de alta qualidade.

## Examples
```
**1. Geração de Tópicos de Alto Nível (Baseado em `generation_1.txt` [1])**

\`\`\`
Você receberá uma lista de documentos. Sua tarefa é identificar tópicos generalizáveis de alto nível que descrevam o conteúdo.

[Instruções]
1. Os rótulos dos tópicos devem ser o mais GENERALIZÁVEIS possível. Não devem ser específicos do documento.
2. Cada novo tópico deve ter um número de nível (e.g., [1]), um rótulo curto e uma descrição do tópico.
3. Os tópicos devem ser amplos o suficiente para acomodar futuros subtópicos.
4. Se um tópico já existir na lista [Tópicos Existentes], retorne o tópico existente.

[Tópicos Existentes]
{Tópicos}

[Documentos]
{Lista de documentos de entrada}

Sua resposta deve estar no formato:
[Nível] Rótulo do Tópico: Descrição do Tópico
\`\`\`

**2. Geração de Subtópicos (Nível 2)**

\`\`\`
O tópico de alto nível é: [1] Tecnologia: Discussões sobre inovações e dispositivos digitais.
Você receberá documentos que foram atribuídos a este tópico. Sua tarefa é gerar subtópicos mais específicos.

[Instruções]
1. Os novos subtópicos devem ser específicos, mas ainda generalizáveis dentro do escopo de [1] Tecnologia.
2. Cada subtópico deve ter um número de nível (e.g., [1.1]), um rótulo curto e uma descrição.

[Documentos]
{Lista de documentos de entrada atribuídos a [1] Tecnologia}

Sua resposta deve estar no formato:
[Nível] Rótulo do Subtópico: Descrição do Subtópico
\`\`\`

**3. Refinamento e Fusão de Tópicos (Baseado em `refinement.txt` [1])**

\`\`\`
Você receberá uma lista de tópicos que pertencem ao mesmo nível de uma hierarquia. Sua tarefa é fundir tópicos que são paráfrases ou duplicatas próximas.

[Regras]
1. Realize as seguintes operações quantas vezes forem necessárias:
   - Fundir tópicos relevantes em um único tópico.
   - Não faça nada e retorne "Nenhum" se nenhuma modificação for necessária.
2. Ao fundir, o formato de saída deve conter o indicador de nível, o rótulo e a descrição atualizados, seguidos pelos tópicos originais.

[Lista de Tópicos]
[1] IA Generativa: Modelos que criam conteúdo.
[2] Modelos de Criação de Conteúdo: Discussões sobre GPTs e DALL-E.

[Sua Resposta]
[1] Inteligência Artificial Generativa: Modelos e Aplicações de Criação de Conteúdo. Tópicos Originais: [1] IA Generativa, [2] Modelos de Criação de Conteúdo.
\`\`\`

**4. Atribuição de Tópicos com Grounding (Baseado em `assignment.txt` [1])**

\`\`\`
Você receberá um documento e uma hierarquia de tópicos. Atribua o documento ao tópico mais relevante na hierarquia.

[Instruções]
1. Os rótulos dos tópicos DEVEM estar presentes na hierarquia fornecida. Você NÃO DEVE criar novos tópicos.
2. A citação de suporte DEVE ser retirada do documento. Você NÃO DEVE inventar citações.

[Hierarquia de Tópicos]
[1] Finanças Pessoais: Orçamento, poupança e investimento.
[2] Saúde e Bem-Estar: Exercícios, dieta e saúde mental.

[Documento]
"A melhor maneira de começar a investir é com um ETF de baixo custo, garantindo a diversificação e minimizando as taxas."

Sua resposta deve estar no formato:
[Nível] Rótulo do Tópico: Raciocínio da Atribuição (Citação de Suporte)

[Sua Resposta]
[1] Finanças Pessoais: O documento discute estratégias de investimento ("A melhor maneira de começar a investir é com um ETF de baixo custo...").
\`\`\`

**5. Prompt de Análise de Tópicos (Sumarização)**

\`\`\`
Atue como um analista de dados. O tópico identificado para os documentos abaixo é: **[3] Feedback do Cliente sobre Usabilidade**.
Sua tarefa é resumir as 5 principais preocupações e as 3 principais sugestões de melhoria mencionadas nos documentos.

[Documentos]
{Lista de 100 feedbacks de clientes}

[Formato de Saída]
**Preocupações Principais:**
1. ...
2. ...
...
**Sugestões de Melhoria:**
1. ...
2. ...
\`\`\`
```

## Best Practices
**Clareza e Estrutura:** Use tags XML ou delimitadores claros (como `[Documento]`, `[Tópicos]`) para separar o texto de entrada das instruções. Isso ajuda o LLM a processar o contexto de forma mais eficiente [1].
**Iteração e Refinamento:** Não espere o resultado final em uma única etapa. Utilize prompts sequenciais (como Geração -> Refinamento -> Atribuição) para construir e validar a hierarquia de tópicos, como visto no framework TopicGPT [1].
**Generalização:** Ao gerar tópicos, instrua o modelo a criar rótulos **generalizáveis** e não específicos ao documento. Isso garante que os tópicos sejam úteis para classificar novos textos [1].
**Validação com Citações:** Exija que o LLM justifique a atribuição de um tópico a um documento com uma **citação direta** do texto. Isso aumenta a rastreabilidade e a confiança no resultado (Grounding) [1].
**Controle de Nível:** Defina explicitamente o nível de detalhe desejado (e.g., "apenas tópicos de alto nível" ou "subtópicos para [Tópico X]").
**Uso de Modelos Híbridos:** Para otimizar custos e desempenho, use modelos mais potentes (como GPT-4 ou Claude Opus) para as etapas de **Geração** e **Refinamento** (que são menos frequentes) e modelos mais leves (como GPT-3.5 ou Gemini Flash) para a etapa de **Atribuição** (que é mais massiva) [1].

## Use Cases
**Análise de Feedback do Cliente:** Identificar automaticamente os principais temas e problemas em avaliações de produtos, tickets de suporte ou comentários em mídias sociais.
**Classificação de Documentos Jurídicos/Regulatórios:** Categorizar grandes volumes de textos legais em tópicos como "Direito Contratual", "Propriedade Intelectual" ou "Regulamentação Ambiental" com alta precisão semântica.
**Pesquisa Acadêmica e Revisão de Literatura:** Analisar resumos de artigos científicos para identificar tendências emergentes, lacunas de pesquisa e a evolução de subcampos em uma área.
**Análise de Notícias e Mídia:** Monitorar a cobertura de eventos e identificar os ângulos e narrativas dominantes em diferentes fontes de notícias.
**Inteligência de Mercado:** Extrair tópicos de relatórios de concorrentes, patentes ou transcrições de chamadas de resultados para identificar estratégias e inovações de mercado.
**Organização de Conteúdo:** Criar automaticamente tags, categorias ou índices hierárquicos para websites, bibliotecas digitais ou sistemas de gerenciamento de conhecimento.

## Pitfalls
**Dependência da Qualidade do LLM:** A qualidade dos tópicos gerados é diretamente proporcional à capacidade de raciocínio e ao contexto do LLM. Modelos mais fracos podem gerar tópicos incoerentes ou redundantes.
**Custo e Latência:** A modelagem de tópicos com LLMs é significativamente mais cara e lenta do que os métodos tradicionais (LDA, BERTopic), especialmente para *corpora* muito grandes, pois cada documento ou lote requer uma chamada de API [1].
**Alucinações na Atribuição:** O LLM pode "alucinar" a citação de suporte ou atribuir um tópico com base em inferências que não estão explicitamente no texto, violando o princípio de *grounding*. A instrução de "NÃO inventar citações" deve ser rigorosa.
**Viés do Modelo:** Os tópicos gerados podem refletir os vieses presentes nos dados de treinamento do LLM, em vez de refletir apenas o conteúdo do *corpus* de entrada.
**Instruções Ambíguas:** Prompts mal formulados ou com regras conflitantes podem levar a resultados inconsistentes, como tópicos muito específicos (document-specific) ou muito amplos (semântica vaga).
**Limite de Contexto:** A modelagem de tópicos geralmente envolve a análise de um grande número de documentos. É necessário um mecanismo de agregação ou amostragem para lidar com o limite de contexto do LLM. O TopicGPT resolve isso em partes, mas é uma limitação inerente ao uso de LLMs [1].

## URL
[https://arxiv.org/abs/2311.01449](https://arxiv.org/abs/2311.01449)
