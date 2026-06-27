# Risk Assessment Prompts

## Description
Prompts de Avaliação de Risco são uma categoria de engenharia de prompt focada em instruir Modelos de Linguagem Grande (LLMs) a realizar tarefas sistemáticas de identificação, análise, priorização e mitigação de riscos em diversos domínios. Essa técnica é amplamente utilizada em contextos de negócios, finanças, segurança cibernética e compliance para automatizar e aprimorar o processo tradicional de avaliação de risco. O objetivo é alavancar a capacidade do LLM de processar grandes volumes de dados e cenários complexos para gerar relatórios de risco estruturados, listas de verificação de compliance e estratégias de mitigação proativas.

## Examples
```
1. **Risco Cibernético:** 'Atue como um analista de segurança cibernética. Dada a nossa infraestrutura de TI (lista de sistemas e tecnologias), identifique e priorize as 5 principais vulnerabilidades de segurança. Para cada uma, sugira uma estratégia de mitigação imediata e de longo prazo.'
2. **Risco Financeiro:** 'Analise o impacto potencial de um aumento de 3% na taxa de juros sobre o nosso fluxo de caixa e a rentabilidade do projeto X. Forneça uma análise de sensibilidade e três cenários de hedge.'
3. **Risco de Compliance:** 'Crie uma lista de verificação de compliance para a nossa empresa de SaaS que opera na UE, focando especificamente no GDPR e na Lei de Mercados Digitais (DMA). Destaque as áreas de maior risco de não conformidade.'
4. **Risco Operacional:** 'Para a nossa nova linha de produção automatizada, liste os 7 principais riscos operacionais, incluindo falhas técnicas e erros humanos. Desenvolva um plano de contingência de três etapas para o risco de maior impacto.'
5. **Risco de Projeto:** 'Avalie o risco de atraso no projeto de lançamento do produto Y, considerando as dependências atuais (lista de dependências) e a alocação de recursos. Calcule a probabilidade de atraso e o impacto financeiro estimado.'
6. **Risco de Terceiros:** 'Desenvolva uma matriz de avaliação de risco para a seleção de um novo fornecedor de logística. Os critérios devem incluir estabilidade financeira, histórico de interrupções na cadeia de suprimentos e práticas de segurança de dados.'
7. **Risco de Reputação:** 'Simule a reação do público a uma falha de serviço em nossa plataforma de mídia social. Identifique os principais riscos de reputação e elabore um rascunho de comunicado de imprensa para gerenciamento de crise.'
```

## Best Practices
1. **Contextualização Detalhada:** Sempre forneça ao LLM o máximo de contexto possível, incluindo o domínio de risco (financeiro, TI, etc.), o escopo da avaliação e quaisquer dados relevantes (ex: lista de ativos, regulamentos aplicáveis).
2. **Definir a Persona:** Atribua uma persona especializada ao LLM (ex: 'Atue como um Gerente de Risco Certificado', 'Seja um Auditor de Compliance') para garantir que a resposta seja estruturada e utilize a terminologia apropriada.
3. **Estrutura de Saída Clara:** Solicite a saída em um formato estruturado (ex: tabela, lista de verificação, matriz de risco) para facilitar a análise e a integração em fluxos de trabalho existentes.
4. **Validação Humana:** Use o LLM como um assistente para gerar rascunhos e identificar riscos, mas a decisão final e a validação crítica devem ser sempre realizadas por um especialista humano.
5. **Iteração e Refinamento:** Use prompts de acompanhamento para aprofundar a análise (ex: 'Agora, detalhe as métricas de monitoramento para o Risco X') ou para refinar a mitigação proposta.

## Use Cases
1. **Modelagem de Risco:** Geração de cenários de estresse e análise de sensibilidade para riscos financeiros e de mercado.
2. **Compliance e Auditoria:** Criação de listas de verificação de conformidade regulatória e identificação de lacunas em políticas internas.
3. **Segurança de Aplicações:** Análise de código e arquitetura para identificar vulnerabilidades de segurança (ex: injeção de prompt, vazamento de dados).
4. **Gestão de Crises:** Simulação de crises (ex: desastres naturais, ataques cibernéticos) e desenvolvimento de planos de resposta e comunicação.
5. **Due Diligence:** Avaliação rápida de riscos operacionais e estratégicos em fusões e aquisições.

## Pitfalls
1. **Vazamento de Dados (Data Leakage):** Inserir informações confidenciais ou proprietárias no prompt, expondo-as ao LLM e, potencialmente, a terceiros.
2. **Alucinação e Imprecisão:** O LLM pode gerar riscos ou estratégias de mitigação plausíveis, mas factualmente incorretas ou inadequadas para o contexto específico, exigindo verificação.
3. **Viés (Bias):** O modelo pode perpetuar vieses presentes nos dados de treinamento, resultando em avaliações de risco que discriminam ou ignoram certos grupos ou cenários.
4. **Injeção de Prompt:** O risco de um usuário mal-intencionado manipular o prompt para fazer o LLM ignorar as instruções de segurança e compliance e gerar uma resposta perigosa.
5. **Superestimação da Capacidade:** Confiar cegamente na saída do LLM sem validação, especialmente em áreas regulamentadas ou de alto impacto.

## URL
[https://www.bizway.io/chatgpt-prompts/risk-assessment](https://www.bizway.io/chatgpt-prompts/risk-assessment)
