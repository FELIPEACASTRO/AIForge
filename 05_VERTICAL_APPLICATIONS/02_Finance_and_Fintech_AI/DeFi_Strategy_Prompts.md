# DeFi Strategy Prompts

## Description
**Prompts de Estratégia DeFi** são instruções altamente estruturadas e contextuais fornecidas a Modelos de Linguagem Grande (LLMs) para simular análises financeiras complexas, avaliar riscos e gerar estratégias de investimento no ecossistema de Finanças Descentralizadas (DeFi). Eles transformam a IA em um assistente de pesquisa e análise, capaz de processar informações sobre protocolos, *tokenomics*, *yield farming*, perdas impermanentes (*impermanent loss* - IL) e segurança de contratos inteligentes. A eficácia desses prompts reside na capacidade de atribuir um papel especializado à IA e solicitar uma análise multifacetada e objetiva, auxiliando o usuário a tomar decisões de investimento mais informadas e seguras no volátil mercado DeFi [1].

## Examples
```
**1. Análise de Protocolo e Risco**
`Atue como um analista de risco sênior. Estou considerando investir no protocolo [Nome do Protocolo] na rede [Nome da Rede]. Forneça uma análise objetiva que inclua: 1) Visão geral do mecanismo de funcionamento, 2) Análise de Tokenomics (distribuição, inflação/deflação), 3) Principais riscos técnicos (auditorias, chaves de administração) e econômicos (governança, *rug pull*), e 4) Comparação de APY com [Protocolo Concorrente]. Conclua com um resumo de Prós e Contras para uma decisão de investimento.`

**2. Estratégia de *Yield Farming* e Perda Impermanente**
`Sou um *yield farmer* experiente. Explique o conceito de Perda Impermanente (IL) de forma didática. Em seguida, analise o par de liquidez [Token A]/[Token B] na DEX [Nome da DEX]. Quais são os riscos de IL esperados para uma variação de preço de 25% em [Token A]? Sugira 3 estratégias para mitigar a IL neste par, focando em pools de baixa volatilidade ou soluções de liquidez concentrada.`

**3. Avaliação de Segurança de Contrato Inteligente**
`Atue como um auditor de segurança de blockchain. Explique os 5 principais vetores de ataque a contratos inteligentes em DeFi (ex: *reentrancy*, *flash loans*). Em seguida, para o protocolo [Nome do Protocolo], pesquise e resuma o status de suas auditorias de segurança (quem auditou, data, *findings* críticos resolvidos). Forneça dicas práticas para um investidor leigo verificar a segurança de um contrato antes de interagir.`

**4. Otimização de Empréstimo e Empréstimo (Lending/Borrowing)**
`Quero otimizar meu capital em [Token A] através de empréstimo e empréstimo (*lending/borrowing*). Compare as plataformas [Plataforma 1] e [Plataforma 2] com base em: 1) Taxas de APY para *lending* de [Token A], 2) Taxas de juros para *borrowing* de [Token B], 3) Fator de colateralização e risco de liquidação. Qual plataforma oferece a melhor relação risco-recompensa para uma estratégia de *looping* conservadora?`

**5. Análise de Tendências e Narrativas de Mercado**
`Atue como um analista de tendências de mercado DeFi. Quais são as 3 principais narrativas de investimento para o próximo trimestre (ex: *Restaking*, *Real World Assets - RWA*, *Layer 2* específicas)? Para a narrativa [Narrativa Escolhida], identifique 2 projetos promissores e justifique a escolha com base em inovação tecnológica, apoio da comunidade e potencial de crescimento de TVL (Total Value Locked).`
```

## Best Practices
**1. Seja Específico e Contextualizado:** Defina claramente o protocolo, o token, a estratégia (ex: *yield farming*, *staking*, *lending*) e o objetivo (ex: *maximizar APY*, *minimizar IL*, *análise de risco*). **2. Defina o Papel da IA:** Comece o prompt atribuindo um papel especializado à IA, como "Atue como um analista de risco DeFi sênior" ou "Você é um estrategista de *yield farming* experiente". **3. Exija Análise de Risco:** Sempre inclua uma seção que solicite a análise de riscos técnicos (contrato inteligente), econômicos (tokenomics) e de mercado (liquidez, volatilidade). **4. Solicite Formato Estruturado:** Peça a saída em um formato fácil de consumir, como uma tabela comparativa, uma lista de prós e contras, ou um resumo executivo. **5. Use Dados Atuais (Se Possível):** Se a plataforma de IA permitir, forneça dados em tempo real ou peça à IA para buscar informações atualizadas sobre APY, liquidez e auditorias.

## Use Cases
**1. Análise de Risco de Protocolo:** Avaliar a segurança e a sustentabilidade econômica de novos protocolos DeFi antes de investir. **2. Otimização de Estratégias de Rendimento (*Yield Farming*):** Determinar os melhores pares de liquidez, plataformas de *lending* ou *staking* para maximizar o APY e gerenciar a Perda Impermanente (IL). **3. Educação e Simulação:** Explicar conceitos complexos de DeFi (ex: *tokenomics*, *liquidation*, *rebase*) e simular cenários de mercado para fins educacionais. **4. Due Diligence de Contratos Inteligentes:** Obter um resumo das auditorias de segurança e dos riscos técnicos associados a um contrato inteligente específico. **5. Comparação de Oportunidades:** Comparar objetivamente diferentes plataformas (DEXs, *lending protocols*, *perpetuals*) com base em métricas específicas (taxas, liquidez, segurança).

## Pitfalls
**1. Confiar em Dados Desatualizados:** LLMs não têm acesso nativo a dados em tempo real de APY, TVL ou preços de tokens. O prompt deve ser construído para solicitar análise de *estrutura* e *risco*, não previsões de preço ou dados voláteis. **2. Prompts Genéricos:** Perguntas como "Qual o melhor investimento DeFi?" resultam em respostas vagas e inúteis. A falta de contexto (rede, token, objetivo) é o erro mais comum. **3. Ignorar a Fonte:** A IA pode alucinar ou fornecer informações incorretas sobre auditorias ou *tokenomics*. O usuário deve sempre usar a saída do prompt como ponto de partida para **verificação manual** no site oficial do protocolo ou em ferramentas de análise on-chain. **4. Excesso de Complexidade:** Tentar incluir muitas variáveis e restrições em um único prompt pode confundir a IA e levar a uma resposta incompleta ou incoerente. É melhor dividir a análise em prompts sequenciais.

## URL
[https://medium.com/@limingchao333/5-prompts-for-defi-investment-you-should-know-af4f1cca5770](https://medium.com/@limingchao333/5-prompts-for-defi-investment-you-should-know-af4f1cca5770)
