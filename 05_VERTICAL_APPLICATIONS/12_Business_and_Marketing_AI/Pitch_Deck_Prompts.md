# Pitch Deck Prompts

## Description
**Pitch Deck Prompts** são uma forma especializada de engenharia de prompt projetada para alavancar Modelos de Linguagem Grande (LLMs) e IA generativa para auxiliar na criação, refinamento e estruturação de *pitch decks* prontos para investidores. Em vez de gerar a apresentação inteira, esses prompts se concentram em gerar conteúdo de alta qualidade para slides específicos (como Problema, Solução, Tamanho de Mercado, Equipe, Pedido de Investimento) ou refinar o conteúdo existente para clareza, concisão e apelo ao investidor. A técnica depende de fornecer à IA um contexto detalhado sobre a *startup*, seu mercado e sua situação financeira, frequentemente usando uma abordagem estruturada e multi-etapas, onde a IA atua como um copiloto ou consultor estratégico [1]. O objetivo é traduzir a visão técnica e de produto do fundador em uma narrativa financeira e de mercado convincente.

## Examples
```
1. **Configuração de Papel e Contexto (Slide Problema):**
```
Aja como um Analista de Capital de Risco experiente, especializado em B2B SaaS. Minha startup, [Nome da Startup], está levantando uma rodada Seed de $1.5M. Nosso produto é [breve descrição do produto]. Gere um Slide de Problema conciso e convincente para o nosso pitch deck, focando nos pontos de dor de [Cliente Alvo] e na falha do mercado atual em resolvê-los.
```
2. **Solução e Fosso Competitivo (Slide Solução):**
```
Com base no Slide de Problema que você acabou de gerar, crie agora o Slide de Solução. A solução deve ser apresentada como a resposta inevitável ao problema. Inclua explicitamente nosso fosso competitivo: [Mencione tecnologia proprietária, efeito de rede ou dados exclusivos].
```
3. **Dimensionamento de Mercado (TAM/SAM/SOM):**
```
Gere o conteúdo do slide de Análise de Mercado. Nosso TAM é de [X] em [Região]. Segmentamos [Segmento Específico]. Forneça um cálculo claro, de cima para baixo e de baixo para cima, para nosso SAM e SOM, e sugira uma estratégia convincente de 'cunha de mercado'.
```
4. **Métricas de Tração e Unidade Econômica (Slide Tração):**
```
Elabore o conteúdo do Slide de Tração. Concentre-se apenas em métricas de usuários pagos: [Número de Usuários Pagos], [Receita Recorrente Mensal - MRR], [Taxa de Churn de Clientes] e [Razão LTV/CAC]. Apresente esses dados de forma a demonstrar crescimento exponencial e forte economia unitária.
```
5. **O Pedido e Uso dos Fundos (Slide Ask):**
```
Estamos pedindo $1.5M. Detalhe o slide 'Uso dos Fundos', alocando o capital em três categorias principais (ex: Desenvolvimento de Produto, Vendas e Marketing, Operações) com porcentagens específicas. A narrativa deve justificar como este financiamento leva ao próximo marco principal (Série A).
```
6. **Refinamento de Linguagem:**
```
Revise os slides de Problema e Solução. Torne a linguagem mais concisa e impactante, reduzindo a contagem de palavras em 30%. Garanta que o tom seja confiante e se dirija diretamente à perspectiva do investidor.
```
7. **Slide de Equipe Focado em Expertise:**
```
Gere o conteúdo do Slide de Equipe. Destaque a expertise única dos nossos três co-fundadores: [Fundador 1 - Expertise], [Fundador 2 - Expertise], [Fundador 3 - Expertise]. O foco deve ser em por que esta equipe específica está unicamente qualificada para executar esta visão.
```
```

## Best Practices
1. **Prompting Estruturado e Multi-Etapas:** Utilize uma sequência de prompts, dedicando um ou mais prompts a cada slide essencial (Problema, Solução, Mercado, Finanças, etc.) para garantir profundidade e foco.
2. **Contexto Detalhado e Persona:** O prompt deve incluir dados específicos, o público-alvo (ex: VCs de Seed-stage), a fase de investimento e, crucialmente, instruir a IA a agir como um "Analista de Capital de Risco" ou "Fundador Experiente" para refinar o tom.
3. **Foco em Métricas de Investidor:** Peça à IA para focar em métricas que importam para investidores (ex: LTV/CAC, retenção, crescimento de receita) e evitar métricas de vaidade.
4. **Refinamento Iterativo:** Use prompts de acompanhamento para refinar a saída da IA, solicitando clareza, concisão ou uma abordagem diferente (ex: "Torne este slide mais conciso e destaque o fosso competitivo").
5. **Mantenha o Foco no Negócio:** Instrua a IA a manter o conteúdo dos slides principais focado no negócio e na estratégia, movendo detalhes técnicos excessivos para um apêndice.

## Use Cases
1. **Geração de Conteúdo:** Criação de texto persuasivo para slides individuais (ex: declaração de problema, descrição da solução, biografias da equipe).
2. **Estrutura e Fluxo:** Definição da sequência e conteúdo ideais para um *deck* de 10-12 slides com base na fase e setor da empresa.
3. **Refinamento de Linguagem:** Tradução de jargão técnico para uma linguagem clara e amigável ao investidor.
4. **Análise Competitiva:** Geração de pontos-chave de discussão para o slide "Concorrência", destacando a vantagem competitiva única.
5. **Narrativa Financeira:** Elaboração da narrativa em torno das projeções financeiras e do slide "Pedido de Investimento" (*Ask*).

## Pitfalls
1. **Detalhe Técnico Excessivo:** O conteúdo gerado pela IA pode ser muito técnico. O prompt deve pedir explicitamente para manter os slides principais focados no negócio e mover as especificações para o apêndice [2].
2. **Conteúdo Genérico ("AI Slop"):** Confiar apenas na IA sem fornecer dados proprietários e específicos resulta em conteúdo genérico e pouco convincente, que não se destaca para investidores.
3. **Métricas de Vaidade:** A IA pode focar em "tração de vaidade" (ex: usuários gratuitos) se não for instruída a priorizar métricas de usuários pagos, retenção e receita [2].
4. **Modelo de Negócio Obscuro:** A IA pode falhar em articular claramente o modelo de receita, preços e premissas de margem se o prompt for vago ou não incluir dados financeiros específicos.
5. **Ignorar IA Responsável/Conformidade:** Para *startups* de IA, a falha em abordar a IA responsável e a conformidade no *pitch* é vista como um *red flag* por muitos VCs [2].

## URL
[https://medium.com/@stunspot/one-sane-prompt-for-pitch-deck-creation-7f97905d69e3](https://medium.com/@stunspot/one-sane-prompt-for-pitch-deck-creation-7f97905d69e3)
