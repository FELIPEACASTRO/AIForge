# Automatic Prompt Engineering (APE)

## Description
A Engenharia de Prompt Automática (APE - Automatic Prompt Engineer) é uma estrutura proposta por Zhou et al. (2022) que automatiza o processo de geração e seleção de instruções (prompts) para Modelos de Linguagem Grande (LLMs). O APE enquadra o problema de geração de instruções como uma tarefa de síntese de linguagem natural, tratando-o como um problema de otimização de caixa preta. Em vez de um engenheiro humano criar prompts manualmente por tentativa e erro, o APE utiliza um LLM (o "Modelo de Inferência") para gerar um conjunto de prompts candidatos a partir de exemplos de entrada e saída (demonstrações) fornecidos. Em seguida, um segundo LLM (o "Modelo de Pontuação") avalia a probabilidade logarítmica de cada prompt candidato produzir as saídas desejadas para as entradas de demonstração. O prompt com a maior pontuação (ou seja, o que melhor explica as demonstrações) é selecionado como o prompt otimizado. O APE demonstrou ser capaz de descobrir prompts de raciocínio de Cadeia de Pensamento (CoT) que superam até mesmo os prompts criados por humanos, como o famoso "Let's think step by step" (Vamos pensar passo a passo).

## Examples
```
**1. Geração de Prompt Otimizado para Classificação de Sentimento**
**Demonstrações (Input/Output):**
- Input: "O filme foi espetacular, adorei cada minuto." Output: "Positivo"
- Input: "Atraso na entrega e produto danificado." Output: "Negativo"
- Input: "Não é bom nem ruim, apenas mediano." Output: "Neutro"
**Prompt de Inferência para o LLM:** "Gere 5 instruções de prompt diferentes que, quando dadas a um LLM, resultariam nas saídas fornecidas para as entradas correspondentes."
**Prompt Candidato Otimizado (Exemplo):** "Classifique o sentimento do texto fornecido em 'Positivo', 'Negativo' ou 'Neutro'. Pense cuidadosamente sobre o tom antes de responder."

**2. Geração de Prompt Otimizado para Resolução de Problemas Matemáticos (CoT)**
**Demonstrações (Input/Output):**
- Input: "Se um trem viaja a 60 km/h por 3 horas, qual a distância percorrida?" Output: "180 km"
- Input: "Qual é o resultado de 15 * 7?" Output: "105"
**Prompt de Inferência para o LLM:** "Gere 3 prompts que induzam um raciocínio passo a passo para resolver problemas de matemática."
**Prompt Candidato Otimizado (Exemplo):** "Resolva o seguinte problema de matemática. Para garantir a precisão, trabalhe a solução em uma abordagem passo a passo antes de fornecer a resposta final."

**3. Geração de Prompt Otimizado para Sumarização**
**Demonstrações (Input/Output):**
- Input: [Artigo longo sobre IA] Output: [Sumário conciso de 3 frases]
**Prompt de Inferência para o LLM:** "Gere 4 instruções de prompt para resumir textos longos, focando na concisão e retenção de informações-chave."
**Prompt Candidato Otimizado (Exemplo):** "Resuma o texto a seguir em no máximo 50 palavras, garantindo que os três pontos principais sejam preservados. Comece com 'Em resumo,'."

**4. Geração de Prompt Otimizado para Tradução com Contexto**
**Demonstrações (Input/Output):**
- Input: "The bank is on the river." Output: "O banco está na margem do rio."
- Input: "I need to go to the bank." Output: "Eu preciso ir ao banco (instituição financeira)."
**Prompt de Inferência para o LLM:** "Gere 2 prompts que instruam o LLM a considerar o contexto para desambiguação de palavras em traduções."
**Prompt Candidato Otimizado (Exemplo):** "Traduza o texto a seguir do inglês para o português. Analise o contexto da frase para escolher a tradução mais precisa para palavras com múltiplos significados."

**5. Geração de Prompt Otimizado para Extração de Entidades**
**Demonstrações (Input/Output):**
- Input: "A reunião com a Dra. Silva será em São Paulo, na sede da TechCorp." Output: "Pessoa: Dra. Silva; Local: São Paulo; Organização: TechCorp"
**Prompt de Inferência para o LLM:** "Gere 3 prompts para extrair entidades nomeadas (Pessoa, Local, Organização) de um texto."
**Prompt Candidato Otimizado (Exemplo):** "Extraia todas as entidades nomeadas do texto. Apresente o resultado em formato JSON com as chaves 'Pessoa', 'Local' e 'Organização'. Se uma entidade não estiver presente, use uma string vazia."
```

## Best Practices
**Foco na Qualidade dos Dados de Demonstração:** A qualidade e a diversidade dos exemplos de entrada/saída (demonstrações) fornecidos ao LLM de inferência são cruciais. Demonstrações ruins ou inconsistentes levarão a prompts candidatos ineficazes.
**Iteração e Refinamento:** O APE é um processo iterativo. Não se contente com o primeiro prompt otimizado. Use o prompt de melhor desempenho como um novo ponto de partida para gerar e testar mais variações.
**Modelo de Inferência vs. Modelo Alvo:** Idealmente, use um LLM mais potente (e.g., GPT-4, Claude 3 Opus) como o "Modelo de Inferência" para gerar prompts candidatos e um modelo mais leve ou o modelo alvo real para a "Seleção" e avaliação, economizando custos e tempo.
**Métricas de Avaliação Robustas:** Use métricas de avaliação que se alinhem diretamente com o objetivo da tarefa (e.g., acurácia para classificação, F1-score para extração de entidades, ROUGE para sumarização). Evite métricas genéricas que podem não capturar a qualidade real da saída.
**Aproveitar o CoT (Chain-of-Thought):** O APE demonstrou ser eficaz na descoberta de prompts de CoT aprimorados (como "Let's work this out in a step by step way..."). Sempre inclua a busca por prompts que induzam o raciocínio passo a passo.

## Use Cases
**Otimização de Prompts de Raciocínio (CoT):** O caso de uso mais notável é a descoberta de prompts de Cadeia de Pensamento (CoT) mais eficazes, como o aprimoramento do prompt "Let's think step by step" para "Let's work this out in a step by step way to be sure we have the right answer."
**Geração de Prompts para Tarefas Específicas:** Automatizar a criação de prompts para tarefas de Processamento de Linguagem Natural (PLN) onde a engenharia manual é tediosa, como classificação de texto, extração de entidades nomeadas (NER) e sumarização.
**Adaptação de Prompts a Diferentes LLMs:** Otimizar prompts para um modelo de linguagem específico (o "Modelo Alvo") sem a necessidade de acesso aos seus gradientes, tratando-o como uma caixa preta. Isso é útil para adaptar prompts entre diferentes APIs de LLMs.
**Melhoria de Desempenho em Benchmarks:** Usado em pesquisa para alcançar ou superar o desempenho de prompts criados por humanos em benchmarks como MultiArith e GSM8K.
**Otimização de Prompts em Ambientes de Produção:** Em ambientes de produção, o APE pode ser usado para refinar continuamente os prompts com base em dados de feedback do usuário, garantindo que o LLM mantenha o desempenho ideal à medida que os requisitos da tarefa evoluem.

## Pitfalls
**Dependência da Qualidade das Demonstrações:** O APE é altamente dependente da qualidade e da representatividade dos exemplos de entrada/saída fornecidos. Demonstrações insuficientes ou ruidosas podem levar à otimização para um prompt subótimo ou incorreto.
**Custo Computacional Elevado:** O processo de geração de múltiplos prompts candidatos e, em seguida, a pontuação de cada um deles usando um LLM (o Modelo de Pontuação) pode ser computacionalmente caro e demorado, especialmente para modelos grandes.
**Risco de Overfitting:** Existe o risco de o prompt otimizado se ajustar excessivamente aos dados de demonstração, resultando em um desempenho ruim em dados de teste não vistos (generalização fraca).
**Complexidade de Implementação:** A implementação do APE requer a orquestração de dois LLMs (Inferência e Pontuação) e a gestão de um processo de busca e avaliação, o que é mais complexo do que a engenharia de prompt manual.
**Limitação a Tarefas de Geração de Texto:** Embora eficaz para tarefas como classificação e CoT, o APE pode ser menos aplicável ou menos eficiente para tarefas que exigem interações complexas ou uso de ferramentas externas.

## URL
[https://arxiv.org/abs/2211.01910](https://arxiv.org/abs/2211.01910)
