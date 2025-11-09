# Litigation Strategy Prompts

## Description
A Engenharia de Prompt para Estratégia de Litígio (Litigation Strategy Prompts) é a aplicação da engenharia de prompt no domínio jurídico, focada em otimizar a interação com Modelos de Linguagem Grande (LLMs) para auxiliar em tarefas de litígio. O objetivo principal é transformar a IA em uma ferramenta de suporte estratégico, capaz de realizar análises de casos, pesquisa jurídica, revisão de documentos (eDiscovery) e elaboração de rascunhos de documentos processuais. A técnica enfatiza a **atribuição de um papel** específico à IA (ex: advogado sênior em PI), a exigência de **raciocínio estruturado** (como o formato IRAC: Issue, Rule, Application, Conclusion) para mitigar alucinações e a necessidade de fornecer **contexto amplo** sem induzir a resposta. O foco está na obtenção de resultados verificáveis e na manutenção do privilégio advogado-cliente e da confidencialidade dos dados.

## Examples
```
**Exemplo 1: Análise de Risco e Estratégia (IRAC)**
```
**Papel:** Você é um advogado de litígio sênior especializado em direito de propriedade intelectual.
**Contexto:** [INSERIR FATOS DO CASO, DOCUMENTOS, JURISPRUDÊNCIA RELEVANTE].
**Tarefa:** Analise o caso de [NOME DO CASO] e determine a probabilidade de sucesso em uma moção de liminar.
**Instruções:**
1. Apresente sua análise no formato IRAC (Issue, Rule, Application, Conclusion).
2. Identifique os três argumentos mais fortes para a defesa e os três mais fortes para o autor.
3. Conclua com uma recomendação estratégica (ex: acordo, litígio, moção específica).
4. Cite as seções relevantes da lei de PI e precedentes.
```

**Exemplo 2: Resumo de Regulamento Complexo**
```
**Papel:** Você é o consultor geral de uma empresa de tecnologia de alto crescimento.
**Tarefa:** Forneça uma análise abrangente e precisa sobre a Regulamentação de Cibersegurança [NOME DA REGULAMENTAÇÃO, ex: LGPD, GDPR, NY DFS 500].
**Instruções:**
1. Resuma as principais disposições e obrigações.
2. Identifique os prazos de conformidade.
3. Explique o escopo e a aplicabilidade.
4. Discuta as implicações do não cumprimento.
5. Mantenha um tom profissional e objetivo.
6. Cite a seção específica da regulamentação para cada ponto.
```

**Exemplo 3: Revisão de Contrato para Treinamento de IA (Mitigação de Risco)**
```
**Documento:** [ANEXAR DPA/CONTRATO DE FORNECEDOR]
**Tarefa:** Revise o DPA anexo e extraia todas as disposições relacionadas ao **treinamento de modelo de IA**.
**Instruções de Saída:**
1. Responda claramente: O fornecedor reserva o direito de usar dados do cliente para treinamento de IA?
2. Existem limites (ex: anonimização, exclusão, restrições)?
3. Cite precisamente o número da seção e o título (ex: Seção 4.2 – Uso de Dados).
4. Escreva em linguagem simples, não jurídica, em um único parágrafo conciso.
```

**Exemplo 4: Refinamento de Prompt (Metaprompting)**
```
**Contexto:** [INSERIR SEU PROMPT ANTERIOR]
**Tarefa:** Com base no meu prompt anterior, identifique as áreas-chave ou conceitos que posso ajustar ou refinar para melhorar a qualidade e a precisão da sua próxima resposta.
**Instruções:**
1. Destaque palavras, frases ou ideias que, se clarificadas ou alteradas, impactariam significativamente a direção ou profundidade da sua análise.
2. Sugira "botões" ou "alavancas" que posso usar, como: aumentar o nível de detalhe, mudar o tom (mais técnico/formal), ou focar em um aspecto diferente do tópico.
```

**Exemplo 5: Elaboração de Contra-Argumento**
```
**Papel:** Você é um advogado de litígio cético.
**Contexto:** [INSERIR O ARGUMENTO PRINCIPAL DA PARTE CONTRÁRIA].
**Tarefa:** Desenvolva uma lista de 5 a 7 contra-argumentos jurídicos sólidos e bem fundamentados contra a tese apresentada.
**Instruções:**
1. Para cada contra-argumento, forneça uma breve justificativa e cite um precedente ou princípio legal que o suporte.
2. Mantenha um tom persuasivo e agressivo.
```

**Exemplo 6: Estrutura de Super-Prompt (Template)**
```
**[TÓPICO/DOCUMENTO]:** [Descrição do documento ou tópico e o tipo de solicitação (ex: análise legal, revisão de contrato)].
**[REQUISITOS DE PESQUISA E RACIOCÍNIO]:**
1. Inclua leis, regulamentos e precedentes relevantes, com citações de parágrafos/seções.
2. Analise cada seção separadamente e sinalize linguagem problemática com recomendações específicas.
3. Forneça uma análise crítica de pontos fortes, fracos, riscos e oportunidades.
4. Compartilhe uma lista de suposições e limitações na sua análise, bem como contra-argumentos e implicações de negócios.
**[INSTRUÇÕES DE FORMATO DE RESPOSTA]:**
1. Comece com um resumo executivo de 3 a 5 tópicos.
2. Use cabeçalhos e sub-cabeçalhos claros.
3. Coloque em negrito as principais descobertas e recomendações.
4. Conclua com uma seção de "próximos passos" com recomendações acionáveis.
```
```

## Best Practices
1. **Atribua um Papel à IA (Role-Playing):** Peça à IA para atuar como um especialista na área de prática relevante (ex: "Você é um advogado de defesa criminal com 20 anos de experiência").
2. **Exija Raciocínio Estruturado (IRAC):** Solicite que a IA estruture sua análise no formato IRAC (Issue, Rule, Application, Conclusion) para facilitar a verificação do raciocínio.
3. **Forneça Contexto Amplo, mas Não Induza a Resposta:** Dê o máximo de contexto e documentos de fundo, mas formule perguntas abertas para evitar vieses (não "lidere a testemunha").
4. **Use a IA para Refinar o Próprio Prompt (Metaprompting):** Peça à IA para identificar áreas-chave ou conceitos no seu prompt que, se ajustados, melhorariam a qualidade da resposta.
5. **Peça Perguntas Verificáveis:** Solicite saídas que possam ser facilmente verificadas, como citações precisas de seções de leis ou documentos, mesmo que as citações da IA precisem ser checadas.
6. **Confie, mas Verifique (Trust but Verify):** Trate a saída da IA como o trabalho de um advogado júnior e sempre realize a revisão humana final.

## Use Cases
1. **Análise de Risco Processual:** Avaliar os pontos fortes e fracos de um caso, identificando riscos e oportunidades.
2. **Revisão de Documentos (eDiscovery):** Extrair cláusulas específicas, resumir DPAs (Data Processing Agreements) ou identificar informações sensíveis em grandes volumes de texto.
3. **Pesquisa Jurídica:** Resumir regulamentos complexos (ex: NY DFS 500), encontrar precedentes relevantes e citar seções específicas.
4. **Elaboração de Documentos:** Gerar rascunhos de memorandos, petições, ou seções de briefs, mantendo um tom profissional e objetivo.
5. **Desenvolvimento de Estratégia:** Criar planos de projeto detalhados para litígios e desenvolver contra-argumentos.

## Pitfalls
1. **Confidencialidade e Privacidade:** Usar ferramentas públicas (como ChatGPT ou Claude) com dados confidenciais. **Mitigação:** Usar contas empresariais, ferramentas com criptografia e redação de informações sensíveis (ex: renomear empresas para "Empresa 1").
2. **Alucinações (Inacurácia):** A IA pode fornecer informações incorretas ou citações falsas. **Mitigação:** Sempre exigir revisão humana ("Trust but verify") e solicitar citações para verificação.
3. **Vieses e Indução:** Fazer perguntas fechadas ou tendenciosas que podem enviesar a análise da IA. **Mitigação:** Usar perguntas abertas e fornecer contexto neutro.
4. **Perda de Privilégio Advogado-Cliente:** O uso inadequado de ferramentas de IA pode comprometer o privilégio. **Mitigação:** Priorizar ferramentas que garantam a manutenção do privilégio.
5. **Dependência Excessiva:** Confiar cegamente na saída da IA sem o devido julgamento profissional. **Mitigação:** Tratar a saída como um rascunho ou sugestão, não como um produto final.

## URL
[https://www.lsuite.co/blog/mastering-ai-legal-prompts](https://www.lsuite.co/blog/mastering-ai-legal-prompts)
