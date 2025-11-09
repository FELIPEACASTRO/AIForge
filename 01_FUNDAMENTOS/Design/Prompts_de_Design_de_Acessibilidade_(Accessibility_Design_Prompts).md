# Prompts de Design de Acessibilidade (Accessibility Design Prompts)

## Description
Prompts de Design de Acessibilidade são instruções específicas e detalhadas fornecidas a Modelos de Linguagem Grande (LLMs) ou ferramentas de IA generativa para auxiliar no processo de design e desenvolvimento de produtos digitais (websites, aplicativos, sistemas) que sejam utilizáveis por pessoas com a mais ampla gama de habilidades e deficiências. [1] [2]

Esta técnica de *Prompt Engineering* se concentra em incorporar explicitamente padrões de acessibilidade (como WCAG - Web Content Accessibility Guidelines, ADA, ATAG) e considerações de design inclusivo nas solicitações de IA. O objetivo é alavancar a capacidade da IA para:
*   **Análise de Conformidade:** Avaliar o design existente ou proposto em relação a critérios técnicos de acessibilidade.
*   **Geração de Soluções Inclusivas:** Sugerir alternativas de design, microcopy, ou fluxos de usuário que atendam a necessidades específicas (ex: baixa visão, deficiência motora, deficiência cognitiva).
*   **Documentação e Treinamento:** Criar listas de verificação, protocolos de teste e resumos de requisitos de acessibilidade. [1]

Ao fornecer contexto, padrões e o público-alvo, o designer ou desenvolvedor transforma a IA em um "co-piloto de acessibilidade", acelerando a integração de práticas inclusivas desde as fases iniciais do projeto. [3]

## Examples
```
**1. Análise de Contraste e Cor:**
`"Aja como um especialista em WCAG 2.1 AA. Analise a seguinte paleta de cores (Primária: #007BFF, Secundária: #6C757D, Fundo: #FFFFFF). Para cada par de cores de texto/fundo, calcule a taxa de contraste e indique se atende ao requisito de texto normal e texto grande. Se não atender, sugira a cor de texto mais próxima que atenda."`

**2. Geração de Texto Alternativo (Alt Text) em Lote:**
`"Eu tenho uma pasta de 50 imagens de produtos de e-commerce. Para cada imagem, gere um texto alternativo conciso e descritivo, focando na função e no conteúdo visual. O público-alvo são usuários de leitores de tela. Exemplo de entrada: 'imagem_produto_123.jpg (tênis de corrida azul com detalhes em amarelo)'. Formate a saída como uma tabela CSV com 'Nome do Arquivo' e 'Texto Alternativo'."`

**3. Otimização de Navegação por Teclado:**
`"Aja como um testador de acessibilidade. Descreva o fluxo de navegação por teclado para um formulário de checkout de 5 etapas. Identifique possíveis armadilhas de foco (focus traps) ou elementos não focáveis. Sugira melhorias no atributo 'tabindex' e na ordem de foco para garantir uma experiência fluida para usuários que não usam mouse."`

**4. Criação de Microcopy para Acessibilidade Cognitiva:**
`"Escreva microcopy claro e empático para as seguintes mensagens de erro em um aplicativo bancário, focando em usuários com deficiência cognitiva. Use linguagem simples (nível de leitura de 5ª série) e evite jargões: 1. Senha incorreta. 2. Saldo insuficiente. 3. Sessão expirada. Para cada erro, forneça uma solução clara e imediata."`

**5. Geração de Diretrizes de Design Inclusivo:**
`"Com base nas diretrizes WCAG 2.2, gere um conjunto de 5 regras de design para garantir que os elementos interativos (botões e links) em um aplicativo móvel sejam acessíveis a usuários com deficiência motora. Inclua requisitos para tamanho mínimo de área de toque (target size) e espaçamento entre elementos."`

**6. Avaliação de Estrutura de Conteúdo:**
`"Analise o seguinte esboço de artigo de blog. Avalie se a estrutura de títulos (H1, H2, H3) é lógica e se o uso de listas e parágrafos curtos maximiza a legibilidade para usuários com dislexia ou deficiência cognitiva. Sugira uma reestruturação se necessário."`
```

## Best Practices
**1. Seja Específico e Contextualizado:** Sempre inclua o padrão de acessibilidade (ex: WCAG 2.1 AA, ADA 2025) e o público-alvo (ex: usuários com deficiência visual, deficiência motora) no prompt. [1] [2]
**2. Referencie Documentos:** Anexe ou peça à IA para referenciar documentos internos (guias de estilo, resultados de testes de usabilidade) ou externos (WCAG, manuais de AT) para respostas mais precisas. [1]
**3. Defina o Papel da IA:** Comece o prompt com "Aja como um especialista em design inclusivo e acessibilidade" para definir o tom e o foco da resposta. [3]
**4. Foco na Ação:** Peça resultados acionáveis, como "Gerar uma lista de verificação", "Comparar conformidade", ou "Escrever microcopy".
**5. Itere e Refine:** Use o resultado inicial da IA como base e refine o prompt para aprofundar a análise ou mudar o foco (ex: de contraste para navegação por teclado).

## Use Cases
**1. Otimização de UI/UX:** Geração de temas de interface, sugestão de paletas de cores com contraste aprovado e otimização de layouts para diferentes necessidades (ex: alto contraste, texto ampliado). [1]
**2. Teste de Conformidade Rápido:** Criação de listas de verificação e protocolos de teste automatizados para verificar a aderência a padrões como WCAG 2.1 ou 2.2 (níveis A ou AA) em estágios iniciais de desenvolvimento. [3]
**3. Geração de Conteúdo Inclusivo:** Redação de microcopy, mensagens de erro e rótulos de formulário que sejam claros, concisos e fáceis de entender para usuários com deficiência cognitiva. [1]
**4. Documentação e Treinamento:** Criação de resumos de requisitos legais (ex: ADA, Seção 508) e tradução de diretrizes técnicas complexas em linguagem simples para equipes de design e desenvolvimento.
**5. Design de Interação Específico:** Sugestão de fluxos de usuário para tecnologias assistivas (ex: navegação por teclado, comandos de voz) e definição de tamanhos de área de toque para usuários com deficiência motora. [1]

## Pitfalls
**1. Confiança Excessiva na IA:** A IA é uma ferramenta, não um substituto para testes manuais e com usuários reais. A conformidade gerada pela IA deve ser sempre verificada. [2]
**2. Prompts Vagos:** Solicitações como "Torne este design acessível" são muito amplas e resultam em respostas genéricas e inúteis. A falta de padrões e contextos específicos é a principal armadilha. [4]
**3. Ignorar o Contexto Humano:** A IA pode falhar em capturar nuances do design ou do contexto de uso que afetam a acessibilidade real (ex: a relevância de um texto alternativo para o contexto da página).
**4. Viés nos Dados de Treinamento:** Se os dados de treinamento da IA não incluírem exemplos de design verdadeiramente inclusivo, as sugestões podem perpetuar práticas de design não acessíveis.
**5. Falha em Referenciar Padrões:** Não especificar o padrão (WCAG 2.1, 2.2, 3.0) ou o nível de conformidade (A, AA, AAA) pode levar a resultados que não atendem aos requisitos legais ou de projeto.

## URL
[https://clickup.com/p/ai-prompts/ui-ux-design-and-accessibility](https://clickup.com/p/ai-prompts/ui-ux-design-and-accessibility)
