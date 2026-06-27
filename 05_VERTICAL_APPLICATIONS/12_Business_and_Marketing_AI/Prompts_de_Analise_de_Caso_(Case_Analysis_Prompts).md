# Prompts de Análise de Caso (Case Analysis Prompts)

## Description
Prompts de Análise de Caso são uma técnica de Engenharia de Prompt que visa instruir um Modelo de Linguagem Grande (LLM) a atuar como um analista, consultor ou especialista para examinar um cenário, problema ou situação complexa (o "caso"). O objetivo é obter uma análise estruturada, diagnósticos, recomendações e planos de ação detalhados, em vez de apenas respostas factuais.

Essa técnica é fundamentalmente estrutural, exigindo que o prompt defina claramente:
1.  **O Papel (Persona):** O LLM deve assumir um papel específico (ex: "Consultor de Estratégia", "Advogado Sênior", "Analista de Dados").
2.  **O Contexto do Caso:** Todos os dados, fatos, restrições e objetivos relevantes para a análise.
3.  **A Estrutura de Análise:** O formato de saída desejado (ex: "Análise SWOT", "5 Forças de Porter", "Parecer Jurídico").
4.  **O Resultado Esperado:** A pergunta específica a ser respondida ou a decisão a ser tomada.

Ao fornecer uma estrutura clara e um contexto rico, os Prompts de Análise de Caso transformam o LLM de um mero gerador de texto em uma ferramenta poderosa de raciocínio e solução de problemas complexos, aplicável em diversos domínios como negócios, direito, tecnologia e saúde.

## Examples
```
1.  **Análise de Negócios (SWOT):**
    ```
    Aja como um Consultor de Estratégia Sênior. Analise o caso da empresa 'TechNova', uma startup de SaaS que viu seu crescimento estagnar em 5% no último trimestre, apesar de um aumento de 20% nos gastos com marketing. O produto é um software de gestão de projetos.
    Realize uma Análise SWOT completa, focando em: Forças (características únicas do produto), Fraquezas (estrutura de preços e suporte ao cliente), Oportunidades (expansão para o mercado europeu) e Ameaças (novo concorrente com preço 50% menor).
    Conclua com três recomendações de ação prioritárias para retomar o crescimento para 15% no próximo trimestre.
    ```

2.  **Análise Jurídica (Parecer):**
    ```
    Aja como um Advogado Sênior especializado em Direito do Consumidor no Brasil.
    Caso: Um cliente comprou um produto online que chegou danificado. A loja se recusa a aceitar a devolução, alegando que o dano ocorreu durante o transporte, responsabilidade da transportadora. O cliente tem 7 dias de prazo de arrependimento.
    Elabore um Parecer Jurídico conciso, citando os artigos relevantes do Código de Defesa do Consumidor (CDC) e jurisprudência aplicável.
    Qual a probabilidade de sucesso em uma ação judicial e qual a melhor estratégia para o cliente?
    ```

3.  **Análise Técnica (Root Cause Analysis - RCA):**
    ```
    Aja como um Engenheiro de DevOps.
    Caso: Ocorreu uma falha crítica no sistema de e-commerce durante a Black Friday, resultando em 4 horas de inatividade. O log indica um pico de requisições no banco de dados (DB) seguido por um deadlock. O DB estava rodando em uma instância de médio porte.
    Realize uma Análise de Causa Raiz (RCA) usando o método dos 5 Porquês.
    Identifique a causa raiz e proponha um plano de mitigação de três etapas (curto, médio e longo prazo) para evitar a recorrência.
    ```

4.  **Análise de Marketing (Segmentação):**
    ```
    Aja como um Analista de Marketing Digital.
    Caso: Uma empresa de fitness lançou um novo aplicativo de treino em casa, mas a taxa de retenção após o primeiro mês é de apenas 15%. O público-alvo inicial era "jovens adultos (18-30 anos)".
    Analise o caso e sugira uma nova segmentação de público-alvo mais promissora.
    Crie um perfil de persona detalhado para o novo segmento e sugira uma nova proposta de valor (Value Proposition) focada nas necessidades dessa persona.
    ```

5.  **Análise de Cenário (Decisão Estratégica):**
    ```
    Aja como um CEO.
    Caso: Sua empresa precisa decidir entre duas opções de investimento: Opção A (Investir R$ 10 milhões em P&D para um produto de alto risco/alta recompensa) ou Opção B (Investir R$ 5 milhões em otimização de processos para um ganho de eficiência de 15%).
    Analise os prós e contras de cada opção, considerando o cenário econômico atual (inflação alta, taxa de juros incerta).
    Recomende a melhor opção e justifique sua decisão com base em uma matriz de risco/recompensa.
    ```

6.  **Análise de Dados (Interpretação de Relatório):**
    ```
    Aja como um Cientista de Dados.
    Interprete o seguinte conjunto de dados de vendas de um produto nos últimos 6 meses: [Janeiro: 100k, Fevereiro: 120k, Março: 90k, Abril: 150k, Maio: 110k, Junho: 180k].
    Identifique tendências, anomalias e possíveis correlações com eventos externos (ex: Março teve um feriado prolongado, Junho teve uma campanha de descontos).
    Projete a venda para o próximo trimestre (Julho, Agosto, Setembro) e justifique a projeção.
    ```

7.  **Análise de Produto (Feature Prioritization):**
    ```
    Aja como um Product Manager.
    Caso: Você tem três novas funcionalidades para priorizar: A (correção de bugs críticos), B (nova funcionalidade solicitada pelo cliente mais importante), C (melhoria de usabilidade que afeta 80% dos usuários).
    Use a estrutura RICE (Reach, Impact, Confidence, Effort) para analisar e priorizar as funcionalidades.
    Apresente a pontuação RICE para cada uma e a ordem de prioridade recomendada.
    ```

8.  **Análise de Saúde (Diagnóstico Diferencial):**
    ```
    Aja como um Médico Clínico Geral.
    Caso: Paciente de 45 anos apresenta fadiga crônica, ganho de peso inexplicável e sensibilidade ao frio. Exames de sangue mostram TSH elevado e T4 livre baixo.
    Realize um Diagnóstico Diferencial, listando as possíveis condições e a mais provável.
    Sugira os próximos passos de investigação e o tratamento inicial recomendado.
    ```

9.  **Análise de Sustentabilidade (ESG):**
    ```
    Aja como um Consultor de ESG (Ambiental, Social e Governança).
    Caso: Uma mineradora está sob pressão pública devido a um pequeno vazamento de rejeitos. A empresa tem um histórico de boa governança, mas o pilar ambiental está em risco.
    Analise o impacto do vazamento na reputação e nas métricas ESG da empresa.
    Proponha uma estratégia de comunicação de crise e três ações concretas para fortalecer o pilar ambiental nos próximos 12 meses.
    ```

10. **Análise de Carreira (Plano de Desenvolvimento):**
    ```
    Aja como um Coach de Carreira.
    Caso: Um profissional de TI com 5 anos de experiência em desenvolvimento Front-end deseja migrar para a área de Machine Learning. Ele tem um conhecimento básico de Python e estatística.
    Analise o gap de habilidades e crie um Plano de Desenvolvimento Individual (PDI) de 6 meses.
    O PDI deve incluir cursos, projetos práticos e métricas de sucesso para a transição de carreira.
    ```
```

## Best Practices
*   **Defina o Papel (Persona) com Clareza:** Comece sempre com "Aja como um [Especialista/Profissional]" para direcionar o tom, o conhecimento e a perspectiva da resposta.
*   **Forneça Contexto Rico:** Inclua todos os dados, restrições, histórico e objetivos do caso. A qualidade da análise depende diretamente da riqueza do contexto fornecido.
*   **Estruture a Saída:** Use frases como "Use a estrutura [X]", "Elabore um [Y]", ou "Responda em formato de [Z]" para garantir que o LLM entregue uma análise organizada e utilizável (ex: SWOT, 5 Porquês, PDI).
*   **Seja Específico no Objetivo:** O prompt deve terminar com uma pergunta clara ou uma solicitação de decisão (ex: "Qual a melhor estratégia?", "Recomende a opção A ou B e justifique").
*   **Iteração e Refinamento:** Se a primeira análise for superficial, use prompts de acompanhamento para aprofundar pontos específicos (ex: "Agora, aprofunde a análise da Ameaça X e proponha contramedidas detalhadas").

## Use Cases
*   **Consultoria Empresarial:** Simulação de cenários de mercado, análise de viabilidade de novos produtos, planejamento estratégico (Análise SWOT, PESTEL).
*   **Área Jurídica:** Elaboração de pareceres preliminares, análise de risco em litígios, interpretação de cláusulas contratuais complexas e identificação de precedentes.
*   **Desenvolvimento de Produto:** Priorização de funcionalidades (RICE, MoSCoW), análise de feedback de usuários e diagnóstico de problemas de usabilidade.
*   **TI e Engenharia:** Análise de Causa Raiz (RCA) para falhas de sistema, planejamento de arquitetura e avaliação de riscos de segurança.
*   **Academia e Educação:** Criação de estudos de caso para ensino, resolução de problemas complexos em disciplinas como finanças e administração.

## Pitfalls
*   **Contexto Insuficiente:** Fornecer apenas um resumo vago do caso. O LLM não pode analisar o que não sabe.
*   **Objetivo Vago:** Pedir apenas "analise este caso". O LLM pode entregar uma análise genérica sem foco na decisão ou resultado necessário.
*   **Confiança Excessiva em Dados Fictícios:** O LLM pode "alucinar" dados, estatísticas ou precedentes legais. A análise deve ser sempre validada por um especialista humano, especialmente em áreas críticas como direito e saúde.
*   **Ignorar a Estrutura:** Não definir o formato de saída. Isso resulta em um texto corrido e difícil de extrair informações acionáveis.
*   **Viés de Confirmação:** Estruturar o caso de forma a induzir o LLM a confirmar uma hipótese pré-existente, em vez de realizar uma análise objetiva.

## URL
[https://www.sybill.ai/blogs/chatgpt-create-case-studies](https://www.sybill.ai/blogs/chatgpt-create-case-studies)
