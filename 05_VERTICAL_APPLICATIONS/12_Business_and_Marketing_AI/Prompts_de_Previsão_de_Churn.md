# Prompts de Previsão de Churn

## Description
Prompts de Previsão de Churn são instruções de engenharia de prompt projetadas para utilizar Grandes Modelos de Linguagem (LLMs) na análise de dados de clientes e na identificação de indivíduos com alto risco de cancelamento (churn). Eles atuam como uma camada de interpretação e análise sobre dados estruturados ou não estruturados (como logs de uso, histórico de suporte, feedback e sentimentos), permitindo que o LLM classifique clientes, sugira ações de retenção personalizadas e justifique suas previsões. Essa técnica é crucial para equipes de Customer Success, Marketing e RevOps, transformando a previsão de churn de um exercício puramente estatístico em uma ferramenta de inteligência acionável e contextualizada [1] [2]. A abordagem mais avançada envolve a criação de *embeddings* de dados de clientes (como a concatenação de atributos categóricos) e o uso desses vetores como entrada para modelos de classificação, com LLMs sendo usados para gerar os *embeddings* ou para interpretar os resultados e gerar estratégias de retenção [3].

## Examples
```
1. **Classificação de Risco e Ação de Retenção:**
   `"Analise os seguintes dados do cliente {ID_CLIENTE}: Uso do Produto: {USO_MES_PASSADO}, Pontuação de Feedback (NPS): {NPS}, Histórico de Tickets de Suporte: {TICKETS_ULTIMOS_6_MESES}. Se o uso caiu mais de 50% OU o NPS for negativo, classifique o cliente como 'Alto Risco' e sugira a 'Próxima Melhor Ação' (ex: 'Oferta de Desconto', 'Chamada Proativa do CS', 'Envio de Conteúdo Educacional'). Retorne o resultado em formato JSON com os campos: 'Risco', 'Justificativa', 'Ação_Sugerida'."`

2. **Análise de Sentimento para Churn:**
   `"Com base nas últimas 10 interações de suporte e comentários em redes sociais do cliente {ID_CLIENTE}, realize uma análise de sentimento. Se o sentimento médio for 'Negativo' e o cliente não tiver usado o recurso principal {NOME_RECURSO} na última semana, preveja a probabilidade de churn (0-100%) e o principal motivo percebido."`

3. **Segmentação de Clientes em Risco:**
   `"Utilize os dados de uso e demográficos para identificar 5 grupos distintos de clientes que demonstraram sinais de churn nos últimos 90 dias. Para cada grupo, descreva o 'Padrão de Comportamento de Risco' e sugira uma 'Campanha de Retenção' específica."`

4. **Interpretação de Dados Categóricos (Para LLM como Embedder/Intérprete):**
   `"O cliente possui os seguintes atributos categóricos: {TIPO_PLANO}, {FREQUENCIA_LOGIN}, {STATUS_PAGAMENTO}, {MOTIVO_CANCELAMENTO_ANTERIOR}. Concatene esses atributos em uma frase coerente e, em seguida, gere um vetor de embedding de 1536 dimensões para esta frase. (Instrução para uso em arquitetura de ML, como em [3])."`

5. **Criação de Alerta Proativo:**
   `"Atue como um especialista em Customer Success. Revise o perfil do cliente {ID_CLIENTE} e os seguintes dados: {DADOS_COMPLETOS}. Crie um alerta interno para a equipe de CS, incluindo: 'Status de Risco', 'Sinais de Alerta (Bullet Points)' e 'Recomendação Imediata'."`

6. **Análise de Causa Raiz (Post-Churn):**
   `"O cliente {ID_CLIENTE} cancelou o serviço. Analise o histórico completo de interações, uso e feedback. Determine a 'Causa Raiz Principal' do churn e forneça 3 'Lições Aprendidas' para evitar casos semelhantes no futuro."`

7. **Simulação de Cenário:**
   `"Se o cliente {ID_CLIENTE} receber uma oferta de 20% de desconto e um mês de consultoria gratuita, qual é a probabilidade estimada de que ele permaneça? Justifique sua resposta com base no histórico de sucesso de ofertas semelhantes para clientes com o perfil {PERFIL_CLIENTE}."`
```

## Best Practices
- **Contextualização Detalhada:** Forneça ao LLM o máximo de contexto possível sobre o cliente, incluindo dados de uso, histórico de suporte e feedback.
- **Definição Clara de Churn:** Defina explicitamente o que constitui "churn" para o seu negócio (ex: inatividade por 30 dias, cancelamento de assinatura).
- **Estrutura de Saída:** Peça ao LLM para formatar a saída de forma estruturada (JSON, tabela) para facilitar a integração com sistemas de CRM ou automação.
- **Métricas de Risco:** Utilize métricas quantificáveis (ex: queda de 50% no uso, pontuação de sentimento abaixo de 3/5) para classificar o risco de forma objetiva.
- **Sugestões Acionáveis:** Exija que o LLM não apenas preveja o churn, mas também sugira a próxima melhor ação de retenção (Next Best Action).

## Use Cases
- **Customer Success:** Identificação proativa de clientes em risco e sugestão de intervenções personalizadas (e-mail, chamada, oferta).
- **Marketing e Vendas:** Segmentação de clientes para campanhas de retenção direcionadas e personalização de mensagens de valor.
- **Análise de Dados:** Interpretação de dados não estruturados (texto de feedback, transcrições de chamadas) para extrair sinais de alerta de churn.
- **Desenvolvimento de Produto:** Análise de causa raiz do churn para informar o *roadmap* de produto e priorizar correções de falhas ou melhorias de recursos.
- **Modelagem Preditiva Híbrida:** Geração de *embeddings* de alta qualidade a partir de dados de clientes (estruturados e não estruturados) para alimentar modelos de Machine Learning tradicionais, melhorando a precisão da previsão [3].

## Pitfalls
- **Dependência Excessiva de Dados Brutos:** Alimentar o LLM com grandes volumes de dados brutos sem pré-processamento ou sumarização pode levar a resultados imprecisos ou a um alto custo computacional.
- **Viés de Dados:** Se os dados de treinamento ou de entrada refletirem um viés histórico (ex: apenas clientes de baixo valor recebiam ofertas de retenção), o LLM pode perpetuar esse viés nas suas sugestões.
- **Falta de Contexto:** Prompts genéricos que não definem claramente as variáveis de entrada ou o objetivo da previsão (ex: "Preveja o churn") resultam em saídas vagas e não acionáveis.
- **Alucinações:** O LLM pode "alucinar" justificativas ou sugerir ações de retenção que não são viáveis ou não se baseiam nos dados fornecidos.
- **Ignorar a Arquitetura Híbrida:** Achar que o LLM pode substituir modelos estatísticos tradicionais. A pesquisa sugere que a combinação de *embeddings* de LLM com classificadores tradicionais (como Regressão Logística) pode ser a abordagem mais robusta [3].

## URL
[https://www.getcensus.com/blog/top-10-llm-prompts-for-revops-and-marketing-teams](https://www.getcensus.com/blog/top-10-llm-prompts-for-revops-and-marketing-teams)
