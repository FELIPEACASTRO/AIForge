# Prompts de Interpretação de Análise de Dados

## Description
Prompts de Interpretação de Análise de Dados são comandos estruturados fornecidos a Modelos de Linguagem Grande (LLMs) com o objetivo de transformar dados brutos ou resultados de análises estatísticas em **insights acionáveis**, **narrativas coesas** e **recomendações estratégicas**. A técnica vai além da simples geração de código ou resumo de dados, focando na **contextualização** e na **comunicação** dos achados. O prompt eficaz atua como uma ponte entre a complexidade técnica da análise de dados e a necessidade de comunicação clara para tomadores de decisão, especificando o papel do LLM, o público-alvo e o formato de saída desejado [1] [2].

## Examples
```
**1. Resumo Executivo Acionável**
```
Contexto: Analisei os dados de vendas do último trimestre (anexo: [tabela de resultados]). O principal achado é uma queda de 15% nas vendas da Região Sul, apesar de um aumento de 5% no orçamento de marketing.
Instrução: Gere um resumo executivo conciso (máximo 5 parágrafos) para o CEO. O resumo deve destacar a principal anomalia (Região Sul), propor 3 hipóteses para a queda e sugerir uma ação imediata para investigação.
```

**2. Interpretação Estatística e Storytelling**
```
Resultados: O teste A/B mostrou que a Variação B teve um aumento de 8% na taxa de conversão (p-valor = 0.01).
Instrução: Explique este resultado estatístico para a equipe de Produto, que não é técnica. Use a metáfora de uma "corrida" para ilustrar a significância estatística. O objetivo é convencê-los a implementar a Variação B imediatamente.
```

**3. Detecção de Anomalias e Causa Raiz**
```
Dados: O gráfico de série temporal (anexo) mostra um pico inesperado de suporte ao cliente na primeira semana de Outubro.
Instrução: Atue como um Analista de Operações. Liste 5 possíveis causas-raiz para este pico, classificadas por probabilidade. Para a causa mais provável, sugira 3 métricas de acompanhamento para monitorar a situação no futuro.
```

**4. Transformação de Dados em Narrativa para Cliente**
```
Achados: A análise de churn para o Cliente X (setor de varejo) indica que 70% dos cancelamentos ocorrem após o 6º mês, citando "complexidade da interface".
Instrução: Crie uma narrativa de 3 slides para uma apresentação ao cliente. O foco deve ser a "Oportunidade de Retenção". O texto deve ser positivo, focado em soluções e apresentar uma recomendação clara de "Redesenho da Jornada do Usuário" como próximo passo.
```

**5. Geração de Perguntas de Negócio a Partir de Insights**
```
Insight: A correlação entre o tempo gasto no aplicativo e a taxa de compra diminuiu 40% no último mês.
Instrução: Gere 5 perguntas de negócio críticas que um Gerente de Produto deve fazer para investigar este insight. As perguntas devem ser específicas e orientar a próxima fase da análise de dados.
```

**6. Interpretação de Modelo de Machine Learning**
```
Modelo: Um modelo de regressão logística para previsão de inadimplência tem as seguintes variáveis mais importantes (coeficientes): Renda (-0.45), Idade (+0.12), Histórico de Pagamento (-0.88).
Instrução: Explique o impacto de cada uma das 3 variáveis no risco de inadimplência. Use termos simples e forneça um exemplo prático para cada variável, como se estivesse treinando um novo analista de crédito.
```

## Best Practices
**1. Estrutura de Prompt Detalhada (Anatomia do Prompt):** Sempre inclua o **contexto** (o que foi analisado), o **objetivo** (o que se espera da interpretação), o **formato de saída** (ex: "resumo executivo", "lista de 5 insights", "narrativa para público leigo") e a **audiência-alvo** (ex: "CEO", "equipe técnica", "cliente").

**2. Forneça os Dados Brutos e os Resultados Chave:** Em vez de apenas descrever, cole os resultados da análise (tabelas, estatísticas, ou um resumo dos achados) diretamente no prompt. Isso reduz a chance de alucinação e garante que a interpretação seja baseada nos dados corretos.

**3. Itere e Refine:** Se a primeira interpretação for superficial, use prompts de acompanhamento para aprofundar. Exemplos: "Com base na sua resposta anterior, qual é a implicação financeira do Insight 3?" ou "Reescreva a interpretação para um público sem conhecimento técnico."

**4. Especifique a Perspectiva:** Peça ao LLM para assumir uma persona (ex: "Aja como um consultor de marketing sênior") para garantir que a interpretação seja relevante e acionável para o domínio específico.

**5. Validação e Ceticismo:** Sempre trate a saída do LLM como um rascunho. Verifique a validade estatística e a lógica dos insights antes de apresentá-los como fato. Use o LLM para identificar *possíveis* insights, mas a validação final é humana.

## Use Cases
**1. Geração de Resumos Executivos:** Transformar relatórios técnicos longos em resumos concisos e focados em decisões para a alta gerência.

**2. Storytelling de Dados:** Criar narrativas envolventes e acessíveis para comunicar achados complexos a públicos não técnicos (ex: marketing, vendas, clientes).

**3. Identificação de Causa Raiz:** Ajudar a formular hipóteses e investigar as razões por trás de anomalias, picos ou quedas nos dados.

**4. Formulação de Recomendações:** Converter insights estatísticos em recomendações de negócio claras e acionáveis, como mudanças em produtos, estratégias de marketing ou operações.

**5. Interpretação de Modelos de Machine Learning:** Explicar a importância das variáveis e o funcionamento interno de modelos complexos (ex: regressão, classificação) em linguagem simples (Explicabilidade de IA).

**6. Criação de Perguntas de Negócio:** Gerar perguntas de acompanhamento para orientar a próxima fase da análise, garantindo que o trabalho de dados esteja alinhado com os objetivos estratégicos da empresa [1] [2].

## Pitfalls
**1. Alucinação de Dados:** O LLM pode "inventar" estatísticas, tendências ou conclusões que não estão presentes nos dados fornecidos. **Mitigação:** Sempre cole os resultados da análise (tabelas, métricas) no prompt e peça ao LLM para citar a fonte dos números dentro do texto.

**2. Confirmação de Viés (Confirmation Bias):** O analista pode ser tentado a aceitar a interpretação do LLM sem crítica, especialmente se ela confirmar uma hipótese pré-existente. **Mitigação:** Peça ao LLM para atuar como um "Advogado do Diabo" e gerar uma interpretação alternativa que conteste os achados iniciais.

**3. Falta de Contexto:** Não fornecer o contexto de negócio (o que a empresa faz, qual é o objetivo da análise) leva a interpretações genéricas e não acionáveis. **Mitigação:** Inclua sempre uma seção de "Contexto" no prompt.

**4. Ignorar a Audiência:** Uma interpretação técnica para um público executivo ou vice-versa. **Mitigação:** Defina explicitamente a persona do LLM e o público-alvo da interpretação.

**5. Exposição de Dados Confidenciais:** Colar dados brutos ou sensíveis diretamente no prompt de um LLM público pode violar políticas de privacidade e segurança. **Mitigação:** Use apenas resumos, estatísticas agregadas ou dados anonimizados ao interagir com modelos de terceiros [3].

## URL
[https://www.codecademy.com/learn/prompt-engineering-for-analytics/modules/prompt-engineering-for-analytics/cheatsheet](https://www.codecademy.com/learn/prompt-engineering-for-analytics/modules/prompt-engineering-for-analytics/cheatsheet)
