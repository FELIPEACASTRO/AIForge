# Deposition Preparation Prompts

## Description
Prompts de Engenharia de IA focados em auxiliar advogados e paralegais na preparação de depoimentos e interrogatórios. Envolvem a análise de transcrições, documentos de descoberta (discovery), e fatos do caso para gerar perguntas estratégicas, identificar inconsistências, resumir informações e criar cronologias. O objetivo é otimizar o tempo de preparação e aprimorar a estratégia legal, reduzindo o tempo gasto em tarefas repetitivas e de organização de documentos. Esta técnica é crucial no campo jurídico, onde a eficiência e a precisão na preparação de litígios são de extrema importância [1].

## Examples
```
1. **Análise de Transcrição:** "Atue como um paralegal experiente. Analise a transcrição do depoimento anexa e extraia todas as declarações do depoente relacionadas à 'negligência grave' e 'dano financeiro'. Formate a saída como uma tabela com as colunas: 'Número da Linha', 'Declaração Exata' e 'Implicação Legal'."

2. **Geração de Perguntas de Interrogatório:** "Com base nos fatos do caso (anexados) e no resumo da testemunha [Nome da Testemunha], gere 10 perguntas de contra-interrogatório destinadas a testar a credibilidade da testemunha, focando em inconsistências de tempo e lacunas de memória. O tom deve ser neutro e investigativo."

3. **Criação de Cronologia:** "Crie uma cronologia detalhada dos eventos relevantes para o caso [Nome do Caso] a partir dos documentos de descoberta fornecidos. Use o formato de lista numerada, incluindo a data, o evento e a fonte do documento (se disponível)."

4. **Identificação de Pontos Fracos:** "Analise os fatos do caso e os argumentos da parte adversa (anexados). Liste 5 potenciais pontos fracos na nossa teoria do caso e sugira uma linha de questionamento para o depoimento que possa mitigar esses riscos."

5. **Simulação de Depoimento (Role-Playing):** "Você é o depoente [Nome do Depoente]. Eu sou o advogado. Responda às minhas perguntas de forma evasiva e defensiva, como faria um depoente hostil. Meu primeiro prompt será: 'Onde você estava na noite de 15 de janeiro?'"

6. **Resumo de Documentos:** "Resuma o documento de 50 páginas anexo (Relatório de Perícia Técnica) em um parágrafo de 150 palavras, destacando apenas as conclusões que apoiam a nossa moção de julgamento sumário. O resumo deve ser claro e objetivo."

7. **Preparação de Testemunha:** "Crie um roteiro de 5 pontos-chave que devo cobrir ao preparar a testemunha [Nome da Testemunha] para o depoimento, garantindo que ela entenda a importância de responder apenas à pergunta feita e manter a calma sob pressão."
```

## Best Practices
**1. Forneça Contexto e Função (Role-Playing):** Comece o prompt definindo a função da IA (ex: "Você é um advogado de defesa experiente", "Você é um paralegal focado em litígios") e o contexto do caso. **2. Alimente a IA com Dados Relevantes:** Anexe ou insira o máximo de dados possível (transcrições, fatos do caso, documentos de descoberta) para que a IA tenha material para análise. **3. Seja Específico no Formato de Saída:** Peça o formato exato que você precisa (ex: "Liste 10 perguntas de acompanhamento em formato de tabela", "Crie um resumo de 500 palavras com marcadores"). **4. Use a Técnica de Prompt Iterativo:** Refine as saídas da IA com prompts de acompanhamento (ex: "Ajuste o tom para ser mais agressivo", "Reescreva as perguntas para focar apenas na linha do tempo"). **5. Mantenha a Confidencialidade:** **NUNCA** insira informações confidenciais ou privilegiadas em modelos de IA de uso geral. Use apenas ferramentas de IA legal específicas e seguras que garantam a privacidade dos dados.

## Use Cases
**1. Análise de Transcrições:** Destilar transcrições longas de depoimentos para identificar rapidamente declarações-chave, inconsistências ou admissões de responsabilidade. **2. Geração de Roteiros de Perguntas:** Criar rascunhos de perguntas para interrogatório direto ou cruzado, focando em áreas específicas como credibilidade da testemunha, linha do tempo ou danos. **3. Organização de Fatos do Caso:** Gerar cronologias de eventos ou listas de fatos materiais a partir de grandes volumes de documentos de descoberta (discovery). **4. Brainstorming Estratégico:** Identificar pontos fracos na própria teoria do caso ou prever possíveis linhas de ataque da parte adversa. **5. Preparação de Testemunhas:** Criar listas de tópicos e orientações para preparar clientes e testemunhas para o ambiente de depoimento, garantindo que estejam prontos para responder a perguntas difíceis.

## Pitfalls
**1. Violação de Confidencialidade:** Inserir informações confidenciais, privilegiadas ou protegidas por sigilo profissional em modelos de IA que não são seguros ou específicos para o setor jurídico. **2. Alucinações e Imprecisão:** Confiar cegamente nas informações ou citações geradas pela IA. A IA pode "alucinar" fatos ou citar jurisprudência inexistente, exigindo sempre a verificação humana. **3. Falta de Contexto:** Usar prompts muito genéricos sem fornecer os documentos de origem (transcrições, petições) ou o contexto legal específico, resultando em saídas inúteis ou irrelevantes. **4. Dependência Excessiva:** Permitir que a IA substitua o raciocínio estratégico do advogado. A IA é uma ferramenta de produtividade, não um substituto para a análise legal profunda e a experiência em litígios. **5. Não Iterar:** Aceitar a primeira saída da IA. Prompts de preparação de depoimento geralmente exigem refinamento e ajustes de tom ou foco através de prompts de acompanhamento.

## URL
[https://www.mycase.com/blog/ai/chatgpt-for-lawyers/](https://www.mycase.com/blog/ai/chatgpt-for-lawyers/)
