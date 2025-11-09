# Grant Application Prompts

## Description
**Prompts para Aplicação de Financiamento (Grant Application Prompts)** são técnicas de engenharia de prompt focadas em utilizar Modelos de Linguagem Grande (LLMs) para auxiliar e otimizar o processo de redação de propostas de financiamento, bolsas de estudo ou subsídios (grants). Essa categoria de prompts é essencialmente uma aplicação de nicho da engenharia de prompt, onde a IA é instruída a assumir o papel de um redator, editor, analista de dados ou consultor de captação de recursos, com o objetivo de gerar conteúdo persuasivo, preciso e alinhado com as diretrizes específicas de um edital [1]. O uso eficaz desses prompts permite que organizações sem fins lucrativos, pesquisadores e empresas acelerem a fase de rascunho, realizem análises comparativas de editais, criem resumos executivos concisos e garantam a conformidade com os requisitos do financiador [3]. A chave para o sucesso reside em fornecer à IA um contexto rico, dados de referência específicos e instruções passo a passo, transformando-a em uma poderosa ferramenta de suporte, e não em um substituto para a expertise humana [2].

## Examples
```
**Exemplo 1: Geração de Esboço (Outline)**
```
**Função:** Você é um consultor de captação de recursos experiente.
**Tarefa:** Crie um esboço detalhado para uma proposta de financiamento de 10 páginas para o edital [Nome do Edital/Fundo].
**Diretrizes:** O esboço deve incluir as seguintes seções obrigatórias: Resumo Executivo (máx. 1 página), Declaração de Necessidade (máx. 2 páginas), Metodologia e Atividades (máx. 3 páginas), Orçamento e Justificativa (máx. 2 páginas), e Avaliação e Sustentabilidade (máx. 2 páginas).
**Contexto:** Nosso projeto é [Breve descrição do projeto].
```

**Exemplo 2: Resumo Executivo Persuasivo**
```
**Função:** Você é um redator de propostas persuasivo.
**Tarefa:** Usando o texto da seção de Metodologia e a Declaração de Necessidade (anexados), escreva um Resumo Executivo conciso e impactante de 250 palavras.
**Foco:** O resumo deve destacar o problema urgente, a solução inovadora do nosso projeto e o impacto mensurável esperado na comunidade [Público-alvo].
**Tom:** Profissional e inspirador.
```

**Exemplo 3: Análise de Lacunas e Alinhamento**
```
**Função:** Você é um analista de conformidade.
**Tarefa:** Compare as diretrizes de elegibilidade e os critérios de avaliação do edital [Nome do Edital] com a nossa proposta de projeto (anexada).
**Saída:** Gere uma lista de 5 pontos de não-conformidade ou lacunas de informação na nossa proposta que precisam ser abordadas para maximizar a pontuação.
```

**Exemplo 4: Refinando a Declaração de Necessidade**
```
**Função:** Você é um editor técnico.
**Tarefa:** Revise a Declaração de Necessidade (anexada) para melhorar a clareza, a fluidez e a força dos dados estatísticos.
**Instrução Específica:** Substitua todas as frases passivas por ativas e garanta que cada estatística citada esteja diretamente ligada a uma meta do projeto.
```

**Exemplo 5: Geração de Indicadores de Avaliação**
```
**Função:** Você é um especialista em monitoramento e avaliação (M&A).
**Tarefa:** Com base na seção de Metodologia (anexada), crie 5 Indicadores de Desempenho Chave (KPIs) SMART (Específicos, Mensuráveis, Alcançáveis, Relevantes, Temporais) para o projeto.
**Formato:** Tabela com colunas: KPI, Meta, Fonte de Dados, Frequência de Coleta.
```

**Exemplo 6: Brainstorming de Ângulos de Financiamento**
```
**Função:** Você é um estrategista de captação de recursos.
**Tarefa:** Gere 5 ângulos de financiamento inovadores para o nosso projeto de [Tema do Projeto] que se alinhem com as prioridades atuais de financiadores corporativos e fundações privadas.
**Restrição:** Os ângulos devem focar em [Ex: Tecnologia, Sustentabilidade, Equidade Social].
```
```

## Best Practices
**1. Forneça Contexto e Função (Role-Playing):** Defina claramente o papel da IA (ex: "Você é um consultor de captação de recursos com 10 anos de experiência") e o contexto do projeto, incluindo a missão da organização, o público-alvo e os resultados desejados [1] [2]. **2. Alimente a IA com Dados de Referência:** Anexe ou insira o máximo de documentos de referência possível, como relatórios anuais, propostas de sucesso anteriores, diretrizes do edital e o orçamento detalhado. A qualidade da saída da IA depende diretamente da qualidade e especificidade da entrada [1] [2]. **3. Quebre Tarefas Complexas:** Em vez de pedir uma proposta completa, divida o processo em etapas menores e focadas: esboço, rascunho da seção de necessidade, rascunho da seção de metodologia, etc. [2]. **4. Mantenha a Voz da Organização:** Use a IA para refinar, resumir ou gerar rascunhos, mas sempre revise e edite para garantir que o tom e a voz autêntica da sua organização sejam mantidos [3]. **5. Especifique o Formato e o Limite:** Inclua requisitos de formatação (ex: "Use linguagem científica contemporânea," "Formato MLA") e limites de palavras ou parágrafos (ex: "Resuma em 500 palavras") [2].

## Use Cases
**1. Rascunho Rápido de Seções:** Acelerar a criação de rascunhos iniciais para seções como a Declaração de Necessidade, Metodologia ou Resumo Executivo, reduzindo o tempo de bloqueio criativo [3]. **2. Análise de Conformidade e Alinhamento:** Comparar rapidamente os requisitos de um edital com o escopo do projeto para identificar lacunas e garantir que a proposta esteja perfeitamente alinhada com os critérios de avaliação do financiador [1]. **3. Geração de Conteúdo Específico:** Criar descrições de projetos concisas, justificativas orçamentárias detalhadas ou indicadores de desempenho (KPIs) SMART, que exigem precisão e aderência a formatos técnicos [2]. **4. Refinamento e Edição:** Usar a IA como um editor para melhorar a clareza, o tom e a gramática do texto, além de adaptar a linguagem para diferentes públicos (ex: de técnico para leigo) [2]. **5. Brainstorming e Estratégia:** Gerar ideias inovadoras para ângulos de financiamento, títulos de projetos ou abordagens de sustentabilidade que podem não ter sido consideradas pela equipe [1].

## Pitfalls
**1. Confiança Cega e "Alucinações":** O erro mais comum é confiar cegamente na IA para gerar fatos, estatísticas ou citações. A IA pode "alucinar" dados ou referências. **Sempre verifique** todas as informações críticas, especialmente dados financeiros e fontes citadas [2]. **2. Perda da Voz Autêntica:** Permitir que a IA escreva grandes seções sem revisão pode resultar em uma linguagem genérica, despersonalizada ou que não reflete a paixão e a missão da organização, o que é crucial para convencer o revisor [3]. **3. Prompts Únicos e Complexos:** Tentar resolver toda a proposta com um único prompt longo e complexo. Isso sobrecarrega a IA e leva a resultados inconsistentes. A iteração e a divisão de tarefas são essenciais [2]. **4. Ignorar as Diretrizes do Edital:** Não fornecer à IA as diretrizes específicas do edital. A IA não pode garantir a conformidade se não souber as regras do jogo (ex: limite de páginas, fontes permitidas, estrutura obrigatória) [1]. **5. Vazamento de Informações Confidenciais:** Incluir informações sensíveis ou proprietárias nos prompts. Lembre-se de que os dados inseridos podem ser usados para treinamento do modelo ou armazenados, portanto, sanitize o conteúdo antes de usá-lo [2].

## URL
[https://bouviergrant.com/prompt-engineering-using-ai-and-large-language-models-llms-for-grant-writing/](https://bouviergrant.com/prompt-engineering-using-ai-and-large-language-models-llms-for-grant-writing/)
