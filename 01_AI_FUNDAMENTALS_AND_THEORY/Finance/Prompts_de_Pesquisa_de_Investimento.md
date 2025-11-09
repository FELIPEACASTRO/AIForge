# Prompts de Pesquisa de Investimento

## Description
A Engenharia de Prompt para Pesquisa de Investimento é a arte e a ciência de criar entradas de texto (prompts) que guiam Modelos de Linguagem Grande (LLMs) a realizar tarefas complexas de análise financeira, avaliação de ativos e planejamento de portfólio. Ela permite que profissionais de finanças e investidores individuais extraiam insights valiosos, realizem análises detalhadas e tomem decisões baseadas em dados de forma mais rápida e eficiente. É uma habilidade crucial para alavancar a IA na análise de mercado, gestão de riscos e identificação de oportunidades de crescimento.

## Examples
```
1. **Análise de Tendências de Mercado:**
   ```
   #CONTEXTO: Você é um analista financeiro experiente. Sua tarefa é identificar as tendências atuais no mercado de [Insira o tipo de mercado, ex: ações de tecnologia] e explicar como elas podem impactar as oportunidades de investimento.
   #META: Fornecer uma análise das últimas tendências de mercado e seus potenciais efeitos nos investimentos.
   #DIRETRIZES DE RESPOSTA: 1. Destaque 3-5 tendências-chave. 2. Explique como essas tendências podem afetar os investimentos positiva ou negativamente. 3. Sugira oportunidades ou riscos potenciais com base na análise.
   ```
2. **Avaliação de Oportunidades de Investimento:**
   ```
   #CONTEXTO: Você é um consultor financeiro. Sua tarefa é avaliar os riscos e as recompensas potenciais de uma oportunidade de investimento específica, como [Insira o tipo de investimento, ex: ações da Empresa X].
   #META: Fornecer uma avaliação clara do investimento, destacando seus prós, contras e retornos potenciais.
   #DIRETRIZES DE RESPOSTA: 1. Analise os benefícios potenciais. 2. Destaque os riscos ou desafios associados. 3. Forneça uma conclusão sobre se é uma oportunidade de alto, médio ou baixo risco.
   ```
3. **Criação de Portfólio Diversificado:**
   ```
   #CONTEXTO: Você é um estrategista de investimento. Sua tarefa é projetar um portfólio diversificado com base na tolerância a risco [Baixa/Média/Alta] e horizonte de investimento [Curto/Médio/Longo prazo] do usuário.
   #META: Fornecer um portfólio que equilibre risco e retornos potenciais em diferentes classes de ativos.
   #DIRETRIZES DE RESPOSTA: 1. Sugira uma mistura de investimentos (ex: ações, títulos, ETFs, imóveis). 2. Forneça uma alocação percentual para cada classe de ativo. 3. Explique o raciocínio por trás da alocação.
   ```
4. **Previsão de Crescimento Futuro:**
   ```
   #CONTEXTO: Você é um previsor financeiro. Sua tarefa é prever o potencial de crescimento de um investimento específico, como [Insira o tipo de investimento, ex: o mercado imobiliário em São Paulo], em um período de [Insira o prazo, ex: 5 anos].
   #META: Fornecer uma análise dos retornos esperados e do crescimento com base em dados e tendências de mercado.
   #DIRETRIZES DE RESPOSTA: 1. Analise dados históricos e tendências atuais. 2. Preveja o crescimento ou declínio potencial. 3. Sugira estratégias para maximizar retornos ou minimizar riscos.
   ```
5. **Análise de Demonstrações Financeiras:**
   ```
   #CONTEXTO: Você é um contador forense. Sua tarefa é analisar as demonstrações financeiras de [Insira o nome da empresa] e identificar quaisquer bandeiras vermelhas ou áreas de preocupação.
   #META: Fornecer um resumo das principais métricas financeiras (ex: P/L, Dívida/Patrimônio) e uma avaliação da saúde financeira da empresa.
   #DIRETRIZES DE RESPOSTA: 1. Calcule e interprete 5 métricas financeiras-chave. 2. Identifique tendências anormais ou riscos potenciais. 3. Forneça uma conclusão sobre a estabilidade financeira.
   ```
6. **Comparação de Classes de Ativos:**
   ```
   #CONTEXTO: Você é um consultor financeiro. Sua tarefa é comparar [Classe de Ativo A, ex: Ouro] vs. [Classe de Ativo B, ex: Bitcoin] e fornecer insights sobre seus benefícios e riscos.
   #META: Ajudar o usuário a entender qual classe de ativo pode ser a mais adequada para seus objetivos de [Insira o objetivo, ex: proteção contra inflação].
   #DIRETRIZES DE RESPOSTA: 1. Compare os prós e contras de cada classe. 2. Destaque riscos e retornos potenciais. 3. Recomende qual se alinha melhor com o objetivo.
   ```
```

## Best Practices
* **Seja Específico e Contextualize:** Inclua detalhes como o tipo de investimento, mercado, horizonte de tempo e tolerância a risco. Comece o prompt definindo o **papel** da IA (ex: "Aja como um analista de risco experiente").
* **Use Estrutura de Prompt:** Utilize cabeçalhos como `#CONTEXTO`, `#META`, `#INFORMAÇÃO` e `#DIRETRIZES DE RESPOSTA` para estruturar a solicitação e garantir uma saída organizada e relevante.
* **Peça por Raciocínio e Fontes:** Solicite que a IA explique o raciocínio por trás de suas conclusões e, se possível, cite as fontes de dados ou notícias utilizadas.
* **Iteração e Perguntas de Acompanhamento:** Se a resposta inicial não for perfeita, refine o prompt ou faça perguntas de acompanhamento para aprofundar a análise (ex: "Agora, analise o impacto da nova regulamentação X sobre esta ação").
* **Validação Humana:** Sempre verifique as informações fornecidas pela IA com dados de mercado atuais e fontes confiáveis antes de tomar qualquer decisão de investimento.

## Use Cases
* **Análise Fundamentalista:** Avaliação rápida de balanços, demonstrações de resultados e fluxo de caixa de empresas.
* **Análise de Sentimento de Mercado:** Resumo de notícias e mídias sociais para avaliar o sentimento em relação a um ativo ou setor.
* **Modelagem de Cenários:** Simulação de diferentes cenários econômicos (ex: alta inflação, recessão) e seu impacto em um portfólio.
* **Geração de Ideias de Investimento:** Identificação de tendências emergentes, setores subvalorizados ou ativos que se encaixam em critérios específicos (ex: alto dividendo, baixo P/L).
* **Due Diligence Simplificada:** Criação de memorandos de investimento ou resumos de pesquisa institucional em minutos.
* **Educação Financeira:** Explicação de conceitos complexos (ex: opções, futuros, derivativos) em linguagem simples.

## Pitfalls
* **Confiança Excessiva nos Dados da IA:** LLMs não têm acesso a dados de mercado em tempo real ou proprietários. A análise é baseada em dados de treinamento e informações públicas.
* **Alucinações Financeiras:** A IA pode "alucinar" fatos, números ou citações de fontes que não existem, levando a decisões de investimento incorretas.
* **Viés de Treinamento:** O modelo pode refletir vieses presentes em seus dados de treinamento, resultando em recomendações que favorecem certos ativos ou estratégias.
* **Falta de Contexto Pessoal:** A IA não conhece a situação financeira completa, obrigações fiscais ou aversão a risco do usuário, a menos que explicitamente fornecido.
* **Prompts Vagos:** Prompts genéricos (ex: "O que devo comprar?") resultam em respostas igualmente genéricas e inúteis. A especificidade é fundamental.

## URL
[https://www.godofprompt.ai/blog/chatgpt-prompts-for-investing?srsltid=AfmBOop7-8EQ9B1zU42uJF7XP5Cxwch1R9eNc6scYhvMwu9I6zSgDUoI](https://www.godofprompt.ai/blog/chatgpt-prompts-for-investing?srsltid=AfmBOop7-8EQ9B1zU42uJF7XP5Cxwch1R9eNc6scYhvMwu9I6zSgDUoI)
