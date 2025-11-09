# Investor Relations Prompts

## Description
**Prompts de Relações com Investidores (IR)** são instruções estruturadas e detalhadas fornecidas a Modelos de Linguagem Grande (LLMs) para auxiliar profissionais de Relações com Investidores na criação, análise e gestão de comunicações com a comunidade financeira. Esses prompts são projetados para automatizar tarefas repetitivas, como resumir relatórios financeiros, gerar rascunhos de comunicados de imprensa, analisar o sentimento do mercado e preparar materiais para *earnings calls* e reuniões com investidores. O objetivo principal é aumentar a eficiência, garantir a consistência da mensagem e liberar a equipe de RI para se concentrar em interações estratégicas e de alto valor. A eficácia desses prompts reside na inclusão de dados específicos da empresa, requisitos de conformidade regulatória e o público-alvo da comunicação.

## Examples
```
**1. Resumo de Resultados Trimestrais para Imprensa:**
`"Aja como um especialista em comunicação de Relações com Investidores. Com base nos seguintes dados financeiros [INSERIR DADOS], gere um rascunho de comunicado de imprensa de 500 palavras para o nosso relatório de resultados do Q3. O comunicado deve focar em [MÉTRICAS CHAVE, e.g., crescimento de receita de 15% e margem EBITDA de 22%], manter um tom otimista, mas realista, e incluir uma citação do CEO sobre a estratégia para o próximo ano. Garanta que a linguagem seja compatível com a SEC (Securities and Exchange Commission)."`

**2. Análise de Sentimento do Mercado:**
`"Analise os seguintes relatórios de analistas [INSERIR TEXTOS/LINKS] e as menções no Twitter/X com a hashtag #NossaEmpresa nos últimos 7 dias. Identifique os três principais temas de preocupação dos investidores e os três principais pontos positivos. Apresente os resultados em uma tabela Markdown com uma pontuação de sentimento (de -5 a +5) para cada tema."`

**3. Rascunho de Q&A para Earnings Call:**
`"Com base no nosso último relatório 10-Q e nas perguntas mais frequentes dos investidores no trimestre passado, gere 5 perguntas desafiadoras que provavelmente serão feitas durante a próxima *earnings call*. Para cada pergunta, forneça uma resposta concisa e aprovada pelo jurídico, focando em [TÓPICO SENSÍVEL, e.g., a desaceleração na China e o impacto da nova regulamentação]."`

**4. E-mail de Atualização para Investidores de Varejo:**
`"Escreva um e-mail de atualização trimestral para investidores de varejo. O e-mail deve ser acessível, evitar jargões excessivos e resumir o desempenho da empresa no Q4. Destaque a importância do nosso novo produto [NOME DO PRODUTO] e reitere nosso compromisso com a sustentabilidade (ESG). O tom deve ser caloroso e de agradecimento."`

**5. Comparação com Concorrentes:**
`"Compare a nossa empresa, [NOME DA EMPRESA], com os concorrentes [CONCORRENTE A] e [CONCORRENTE B] em termos de [MÉTRICAS, e.g., P/E Ratio, crescimento de receita YOY e *free cash flow*]. Use os dados mais recentes disponíveis publicamente. Crie um *slide deck* de 3 slides com gráficos de barras para visualização, incluindo uma breve análise das nossas vantagens competitivas."`
```

## Best Practices
**1. Contextualização e Especificidade:** Sempre forneça o máximo de contexto possível. Inclua dados financeiros específicos, o público-alvo (analistas, investidores de varejo, mídia) e o tom desejado (otimista, cauteloso, informativo).
**2. Revisão Humana Obrigatória:** Nunca use a saída da IA para comunicações regulatórias ou públicas sem uma revisão e validação rigorosas por um profissional de Relações com Investidores (RI) e jurídico. A precisão e a conformidade são primordiais.
**3. Proteção de Dados Sensíveis:** Evite inserir informações financeiras não públicas, estratégias confidenciais ou dados pessoais de investidores em modelos de IA de uso geral. Utilize soluções de IA *on-premise* ou com garantias de privacidade de dados.
**4. Defina o Formato de Saída:** Peça explicitamente o formato desejado (e.g., "gere uma tabela em Markdown", "escreva um e-mail formal", "crie um resumo de 5 parágrafos").
**5. Mantenha a Consistência da Voz:** Inclua instruções sobre a voz e o estilo da empresa (e.g., "Mantenha o tom formal e alinhado com a nossa política de comunicação corporativa").

## Use Cases
**1. Automação de Resumos e Análises:** Geração rápida de resumos de relatórios anuais, *earnings calls* e *press releases* para consumo interno ou externo.
**2. Preparação para Reuniões com Investidores:** Criação de *scripts* de perguntas e respostas (Q&A), *talking points* e *pitch decks* personalizados para diferentes tipos de investidores (e.g., *hedge funds*, fundos de pensão, investidores de varejo).
**3. Monitoramento e Análise de Sentimento:** Rastreamento e análise do sentimento do mercado, cobertura da mídia e relatórios de analistas para identificar preocupações emergentes e tendências de investimento.
**4. Comunicação de Crise:** Rascunho de planos de comunicação de crise e mensagens-chave para gerenciar a percepção do investidor durante eventos negativos (e.g., recall de produtos, litígios, mudanças na liderança).
**5. Conformidade e Governança:** Auxílio na redação de seções de relatórios regulatórios (com revisão humana), e na preparação de materiais sobre ESG (Ambiental, Social e Governança) para atender às crescentes demandas dos investidores.

## Pitfalls
**1. Alucinações e Imprecisão de Dados:** A IA pode gerar informações factualmente incorretas ou inventar dados financeiros. Isso é catastrófico em RI, onde a precisão é legalmente exigida.
**2. Violação de Confidencialidade (Data Leakage):** Usar modelos de IA públicos para processar dados não públicos (como resultados financeiros preliminares ou estratégias de fusão e aquisição) pode resultar em vazamento de informações sensíveis.
**3. Falta de Nuance e Tom:** A IA pode não capturar a nuance e o tom exigidos em comunicações de RI, especialmente em situações de crise ou ao abordar tópicos sensíveis como governança corporativa.
**4. Conformidade Regulatória (Compliance):** A IA não é um advogado. Confiar cegamente em seu conteúdo para relatórios regulatórios (como 8-K, 10-Q, 10-K) pode levar a erros de conformidade e penalidades severas da SEC ou de outros órgãos reguladores.
**5. Viés e Repetição:** Se o modelo for treinado em dados enviesados, ele pode perpetuar uma visão excessivamente otimista ou pessimista, ou simplesmente repetir a linguagem de relatórios anteriores, perdendo a oportunidade de comunicação estratégica.

## URL
[https://promptdrive.ai/ai-prompts-investor-relations/](https://promptdrive.ai/ai-prompts-investor-relations/)
