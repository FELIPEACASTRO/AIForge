# Discovery Prompts

## Description
Discovery Prompts (Prompts de Descoberta) são uma técnica de Engenharia de Prompt focada em acelerar e aprofundar a fase de **Product Discovery** (Descoberta de Produto) e **User Research** (Pesquisa de Usuário). O cerne da técnica é instruir o Modelo de Linguagem Grande (LLM) a assumir o papel de um especialista em Lean Startup, Product Discovery ou UX Researcher. O objetivo principal é utilizar a IA para ajudar a **definir o problema, gerar hipóteses de solução, criar planos de pesquisa (entrevistas, surveys, testes) e sugerir métodos de validação de baixo custo e baixa tecnologia** ("processualizar antes de produtizar"). Em vez de pedir à IA para construir a solução final, o Discovery Prompt a utiliza para estruturar o processo de aprendizado e validação inicial, minimizando o desperdício de recursos em soluções não validadas. É uma abordagem que prioriza o aprendizado e a redução de risco antes do desenvolvimento completo de software.

## Examples
```
**Exemplo 1: Geração de Hipótese e Plano de Validação (Role-Playing)**

**Instrução:** Atue como um **Product Manager sênior** especializado em Lean Startup. Nosso problema é: "Usuários de pequenas empresas perdem muito tempo inserindo dados manualmente em nosso software de gestão, o que causa alta taxa de abandono após o primeiro mês."

**Prompt:** "Com base no problema, gere uma **Hipótese de Valor** clara (formato: 'Acreditamos que [ação] para [público] resultará em [resultado mensurável]'). Em seguida, proponha **três experimentos de baixo custo** (ex: Concierge MVP, Landing Page de Fumaça, Entrevistas) para validar esta hipótese antes de qualquer desenvolvimento de software. Para cada experimento, defina o **critério de sucesso**."

**Exemplo 2: Criação de Roteiro de Entrevista (User Research)**

**Instrução:** Atue como um **UX Researcher** com foco em entrevistas de profundidade. Nossa hipótese é: "A introdução de um recurso de importação automática de planilhas reduzirá o tempo de setup inicial em 50% para novos clientes."

**Prompt:** "Crie um **roteiro de entrevista semi-estruturado** com 8 perguntas abertas para validar esta hipótese. As perguntas devem focar na **dor atual** (como eles fazem hoje), na **necessidade** (o que eles esperam de uma solução) e na **disposição a pagar/usar** (o valor percebido). Inclua uma pergunta de 'teste de estresse' para refutar a hipótese."

**Exemplo 3: Análise de Concorrência e Lacunas (Benchmark)**

**Instrução:** Atue como um **Analista de Inteligência Competitiva**. Nosso produto é um aplicativo de meditação. Queremos descobrir o que falta em nossa oferta para usuários avançados.

**Prompt:** "Identifique **três concorrentes diretos e dois indiretos** (ex: jogos de quebra-cabeça) no mercado de bem-estar digital. Analise as **funcionalidades de 'descoberta'** (como o usuário encontra novos conteúdos) e as **estratégias de retenção** para usuários que usam o aplicativo há mais de 6 meses. Apresente os resultados em uma tabela, destacando as **lacunas** que podemos explorar em nosso Discovery."

**Exemplo 4: Definição de Métricas para MVP Manual**

**Instrução:** Atue como um **Especialista em Growth e Métrica**. Estamos testando manualmente um serviço de curadoria de notícias para executivos (o 'MVP Manual').

**Prompt:** "Quais são as **três métricas de sucesso** mais importantes para este MVP Manual, focadas em aprendizado e validação, e não em escala? Defina o **objetivo quantitativo** (ex: 'X% de Y') para cada métrica que indicaria que a hipótese de valor foi validada e que devemos 'produtizar' o serviço."

**Exemplo 5: Refutação de Hipótese (Pensamento Crítico)**

**Instrução:** Atue como um **Cético de Produto**. Nossa equipe está muito animada com a ideia de um 'bot de atendimento 24/7'.

**Prompt:** "Liste **cinco razões críticas** pelas quais um bot de atendimento 24/7 pode **falhar** em nosso contexto (Pequenas Empresas de Serviços). Para cada razão, sugira uma **pergunta de pesquisa** que devemos fazer aos usuários para tentar **refutar** a ideia do bot antes de construí-lo."

**Exemplo 6: Estruturação de Pesquisa Quantitativa (Survey)**

**Instrução:** Atue como um **Especialista em Pesquisa Quantitativa**. Queremos medir a frequência e a intensidade da dor de 'gerenciar faturas de fornecedores' em nossa base de 500 clientes.

**Prompt:** "Crie um **mini-survey** com 5 perguntas (incluindo perguntas de escala Likert e múltipla escolha) para medir a **frequência** e a **severidade** deste problema. Inclua uma pergunta demográfica chave para segmentação. O objetivo é obter dados para priorizar esta dor no roadmap."

**Exemplo 7: Análise de Dados Existentes (Feedback de Suporte)**

**Instrução:** Atue como um **Cientista de Dados de Produto**. Temos 500 tickets de suporte abertos no último mês.

**Prompt:** "Se eu te fornecer o texto desses 500 tickets, quais **cinco categorias de problemas** você sugeriria para agrupá-los? Quais **três palavras-chave** você usaria para identificar rapidamente se o problema está relacionado à 'usabilidade', 'valor' ou 'viabilidade técnica'? O objetivo é usar a IA para estruturar a análise de dados qualitativos."
```

## Best Practices
1. **Definição de Persona/Papel (Role-Playing):** Comece o prompt instruindo a IA a assumir o papel de um especialista (ex: "Atue como um Product Manager sênior...").
2. **Contexto Detalhado do Problema:** Forneça o máximo de contexto possível sobre o problema, o público-alvo e o objetivo de negócio.
3. **Foco na Validação:** Peça explicitamente à IA para sugerir **métodos de validação** (ex: "Quais são 3 experimentos de baixo custo para validar esta hipótese?").
4. **Perguntas Investigativas:** Use o prompt para gerar perguntas que aprofundem o entendimento do problema (ex: "Gere 10 perguntas de entrevista para usuários sobre este ponto de dor.").
5. **Iteração e Refinamento:** Use a saída da IA como ponto de partida para prompts de acompanhamento, refinando o problema ou a hipótese.

## Use Cases
1. **Geração de Hipóteses:** Criar hipóteses de valor, usabilidade e viabilidade para um novo recurso ou produto.
2. **Desenho de Pesquisa:** Elaborar roteiros de entrevistas, questionários de pesquisa (surveys) e planos de teste de usabilidade.
3. **Análise de Concorrência (Benchmark):** Solicitar à IA que identifique concorrentes indiretos e suas soluções para um problema específico.
4. **Definição de Métricas:** Gerar sugestões de métricas de sucesso (KPIs) para a fase de validação (ex: "Métricas de Sucesso para um MVP Manual").
5. **Priorização de Problemas:** Usar a IA para analisar dados de feedback e sugerir a priorização de dores do usuário.

## Pitfalls
1. **Confundir Descoberta com Solução:** Pedir à IA para "criar o produto" em vez de "estruturar a pesquisa sobre o problema".
2. **Falta de Contexto:** Fornecer um problema vago, resultando em sugestões de pesquisa genéricas e inúteis.
3. **Aceitar a Saída como Verdade Absoluta:** A IA pode sugerir métodos de pesquisa que não são adequados ao contexto real da empresa ou do produto.
4. **Ignorar a Execução Manual:** O valor do Discovery Prompt está em estruturar a validação manual. O erro é pular a validação e ir direto para o desenvolvimento.
5. **Viés de Confirmação:** Usar a IA apenas para confirmar uma ideia pré-concebida, em vez de buscar ativamente a refutação da hipótese.

## URL
[https://calirenato82.substack.com/p/prompt-ia-discovery-operacionalizar-produtizar](https://calirenato82.substack.com/p/prompt-ia-discovery-operacionalizar-produtizar)
