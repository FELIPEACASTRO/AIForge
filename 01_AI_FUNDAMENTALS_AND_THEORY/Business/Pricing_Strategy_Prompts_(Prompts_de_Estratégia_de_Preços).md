# Pricing Strategy Prompts (Prompts de Estratégia de Preços)

## Description
**Prompts de Estratégia de Preços** são um conjunto de instruções estruturadas e detalhadas fornecidas a modelos de linguagem (LLMs) para auxiliar na análise, criação e otimização de estratégias de precificação de produtos ou serviços. Eles transformam a IA em um consultor de precificação, capaz de processar dados complexos (como análise competitiva, elasticidade de preço, custos e valor percebido) e gerar recomendações acionáveis. Essa técnica é amplamente utilizada no contexto de negócios, especialmente em SaaS (Software as a Service) e e-commerce, onde a precificação dinâmica e baseada em valor é crucial para a maximização da receita e da margem de lucro [1] [2]. A eficácia desses prompts reside na sua capacidade de integrar princípios de economia, psicologia comportamental e análise de dados em uma única consulta, exigindo que a IA atue em um papel específico (persona) e utilize frameworks de negócio reconhecidos (como Jobs-to-Be-Done ou ReAct Prompting) [1] [2].

## Examples
```
**1. Análise Competitiva e Posicionamento de Preço:**
"Atue como um analista de preços sênior. Compile uma tabela comparativa de preços para os 5 principais concorrentes no nicho de 'Software de Gestão de Projetos para PMEs'. Inclua: nome do plano, preço mensal/anual, principais recursos de valor e táticas de desconto (se houver). Com base nesta análise, sugira um ponto de preço inicial para o nosso novo plano 'Pro' que maximize a aquisição, mantendo uma margem de 40%."

**2. Cálculo de Preço Baseado em Valor (Value-Based Pricing):**
"Nosso software de automação de marketing economiza, em média, 10 horas de trabalho por mês para um cliente. Se o custo médio de uma hora de trabalho do funcionário-alvo é de R$ 50,00, calcule um preço de assinatura mensal ideal usando o modelo de precificação baseado em valor. Assuma que capturamos 20% do valor economizado. Apresente o cálculo passo a passo e justifique a taxa de captura de valor."

**3. Simulação de Elasticidade de Preço:**
"Crie um cenário de simulação de elasticidade de preço para o nosso produto 'Ebook de Finanças Pessoais'. O preço atual é R$ 99,00 e vendemos 500 unidades por mês. Simule o impacto na receita e no lucro se o preço for ajustado em ±10% e ±20%. Qual é o preço que maximiza a receita total, assumindo uma elasticidade de demanda de -1.5?"

**4. Desenho de Arquitetura de Níveis (Tiered Pricing):**
"Desenhe uma estrutura de preços de três níveis (Básico, Premium, Empresarial) para um serviço de streaming de vídeo B2B. Defina o principal diferenciador de cada nível (por exemplo, número de usuários, qualidade de vídeo, suporte), o preço psicológico recomendado para cada um e um 'upgrade nudge' (incentivo de upgrade) para mover clientes do Básico para o Premium."

**5. Otimização de Preço Psicológico:**
"Audite a página de preços atual do nosso produto SaaS (URL: [URL da página]). Aplique a técnica de 'Charm Pricing' (preços terminados em 9) e 'Price Anchoring' (ancoragem de preço) para o plano 'Premium' de R$ 199,00. Sugira o novo preço e o texto de ancoragem que deve ser usado para aumentar a percepção de valor e a taxa de conversão."

**6. Estratégia de Desconto e Promoção:**
"Elabore uma política de descontos para o nosso serviço de consultoria. Especifique os tipos de desconto permitidos (volume, fidelidade, sazonal), o teto máximo de desconto para novos clientes (em %) e as diretrizes para evitar a desvalorização da marca. Apresente o resultado como um guia de governança de descontos."

**7. Mapeamento de Disposição a Pagar (Willingness-to-Pay):**
"Usando o framework Jobs-to-Be-Done, defina duas personas distintas para o nosso aplicativo de meditação. Para cada persona, estime uma faixa de 'disposição a pagar' (WTP) justificada por seus 'jobs' funcionais e emocionais. Identifique também os gatilhos de 'justiça de preço percebida' para cada uma."
```

## Best Practices
**1. Contextualização Detalhada:** Sempre forneça à IA o máximo de contexto possível, incluindo o problema de negócio, o nicho de mercado, o público-alvo (ICP) e o contexto geográfico. Prompts como "Atue como um analista de preços sênior" definem o papel da IA e melhoram a qualidade da resposta [2].
**2. Fornecimento de Dados:** Integre dados reais ou hipotéticos, mas realistas, como relatórios de margem, dados de churn, custo por hora de trabalho ou elasticidade de demanda. A IA funciona melhor quando tem "dados de entrada" para analisar e processar [1].
**3. Especificação do Formato de Saída:** Peça o resultado em um formato estruturado (tabela comparativa, cálculo passo a passo, lista de verificação, diagrama de árvore de decisão) para garantir que a saída seja diretamente acionável e fácil de aplicar [2].
**4. Foco em Valor e Psicologia:** Direcione a IA para frameworks de precificação avançados, como **Value-Based Pricing** (Preço Baseado em Valor) e **Behavioral Pricing Theory** (Teoria da Precificação Comportamental), para ir além da precificação baseada em custos [2].
**5. Iteração e Refinamento:** Use os prompts como ponto de partida. O resultado da IA deve ser um rascunho ou uma recomendação que precisa ser validada e refinada por um especialista humano em precificação [1].

## Use Cases
**1. Otimização de Receita em SaaS:** Auxiliar empresas de Software as a Service a definir modelos de precificação por níveis (tiered pricing), métricas de uso e pontos de preço que maximizem o MRR (Receita Recorrente Mensal) e minimizem o churn [2].
**2. Análise de Competitividade em E-commerce:** Usar dados de vendas e concorrência (Amazon, Etsy, etc.) para identificar produtos com preço incorreto e recomendar ajustes para aumentar a competitividade e a margem de lucro [1].
**3. Lançamento de Novos Produtos:** Determinar o preço de entrada ideal para um novo produto ou serviço, utilizando o cálculo de preço baseado em valor (VBP) e mapeamento da disposição a pagar (WTP) do cliente [2].
**4. Estruturação de Descontos e Promoções:** Criar políticas de governança de descontos para evitar a desvalorização da marca, definindo tetos de desconto e tipos de promoção permitidos (volume, fidelidade, sazonal) [2].
**5. Expansão Internacional:** Desenvolver matrizes de localização de preços para diferentes mercados globais, ajustando os valores pela Paridade de Poder de Compra (PPP) e considerando impostos e percepções culturais [2].

## Pitfalls
**1. Confiar Apenas na IA:** O maior erro é aceitar a recomendação de preço da IA sem validação humana e testes A/B. A IA pode não capturar nuances culturais, regulatórias ou a dinâmica de mercado em tempo real [1].
**2. Falta de Contexto e Dados:** Usar prompts genéricos sem fornecer dados específicos (custos, margens, dados de clientes, concorrência) leva a respostas superficiais e ineficazes. A qualidade da saída é diretamente proporcional à qualidade do *input* [2].
**3. Ignorar a Psicologia do Preço:** Focar apenas em números e custos, ignorando o impacto de técnicas de precificação psicológica (como ancoragem, efeito isca ou preços terminados em 9), que são cruciais para a percepção de valor [2].
**4. Não Especificar o Formato:** Não pedir um formato de saída estruturado (tabela, lista, cálculo) resulta em longos parágrafos de texto que são difíceis de extrair e aplicar diretamente no negócio [2].
**5. Desconsiderar a Localização:** Para produtos globais, não incluir a localização de preços (ajustes por Paridade de Poder de Compra - PPP, impostos locais) pode levar a preços não competitivos ou a arbitragem de mercado [2].

## URL
[https://medium.com/@slakhyani20/10-chatgpt-prompts-for-pricing-strategy-creation-8ecb05e47d68](https://medium.com/@slakhyani20/10-chatgpt-prompts-for-pricing-strategy-creation-8ecb05e47d68)
