# Customer Segmentation Prompts

## Description
**Prompts de Segmentação de Clientes** são instruções especializadas fornecidas a Modelos de Linguagem Grande (LLMs) para auxiliar na identificação, análise, refinamento e aplicação de segmentos de clientes. Esta técnica de *Prompt Engineering* aproveita a capacidade do LLM de processar grandes volumes de dados textuais (como feedback de clientes, transcrições de chat, descrições de produtos comprados, ou dados demográficos em formato de texto) para realizar tarefas complexas de segmentação. Em vez de apenas gerar conteúdo, o prompt atua como um motor analítico, orientando o LLM a analisar dados, definir critérios, criar personas e desenvolver estratégias de segmento. A eficácia reside na capacidade do LLM de sintetizar informações complexas e traduzir insights de dados em linguagem natural acionável, tornando a segmentação mais rápida e acessível.

## Examples
```
1. **Segmentação Comportamental (RFM):**
   "Atue como um analista de dados. Analise a lista de clientes fornecida (ID, Data da Última Compra, Frequência de Compra, Valor Total Gasto) e crie 4 segmentos distintos (Campeões, Clientes em Risco, Novos Clientes, Clientes Adormecidos). Para cada segmento, forneça uma descrição e sugira a melhor ação de marketing."

2. **Criação de Persona Sintética:**
   "Com base nos seguintes dados de um segmento de clientes (Idade Média: 35, Localização: Capitais do Sudeste, Produtos Comprados: SaaS de produtividade, Frequência: Mensal, Principal Dor: Falta de tempo), crie uma persona detalhada, incluindo nome, cargo, objetivos, desafios e uma citação representativa."

3. **Refinamento de Segmento por Feedback:**
   "Analise os 50 comentários de clientes abaixo sobre o Produto X. Identifique os temas recorrentes e sugira um novo sub-segmento para clientes que expressam 'dificuldade na integração' e 'alto valor percebido'. Nomeie o sub-segmento e justifique a separação."

4. **Segmentação de Conteúdo (Intenção):**
   "Classifique os 20 títulos de artigos de blog abaixo em três categorias de intenção de compra (Conscientização, Consideração, Decisão). Para cada categoria, sugira um 'call-to-action' (CTA) ideal para o segmento de 'Pequenas Empresas'."

5. **Análise de Segmento de Alto Valor:**
   "Qual é o perfil psicográfico e demográfico do nosso segmento de clientes que gasta mais de R$ 5.000 por ano? Use os dados de transação e localização fornecidos para traçar um perfil e identificar 3 canais de comunicação mais eficazes para alcançá-los."

6. **Prompt de Ação Pós-Segmentação:**
   "Crie uma sequência de 3 e-mails de reengajamento para o segmento 'Clientes em Risco' (última compra há 6 meses). O tom deve ser empático e o objetivo é oferecer um incentivo personalizado de 15% de desconto. Inclua o assunto e o corpo de cada e-mail."

7. **Identificação de Variáveis de Segmentação:**
   "Atue como um estrategista de marketing. Quais são as 5 variáveis de segmentação mais cruciais que devemos considerar para um novo produto de 'Alimentos Orgânicos Congelados' no mercado brasileiro? Justifique cada variável (ex: Demográfica, Comportamental, Psicográfica)."
```

## Best Practices
*   **Definição Clara de Papel e Tarefa:** Comece o prompt definindo o papel do LLM (ex: "Atue como um analista de dados", "Você é um estrategista de marketing") e especifique a tarefa de segmentação.
*   **Fornecimento de Contexto/Dados:** Inclua os dados de entrada (ou a estrutura dos dados) e os critérios de segmentação desejados (ex: "Segmentar por RFM", "Com base em dados demográficos e de compra").
*   **Restrição de Saída:** Peça um formato de saída estruturado (ex: "Retorne uma tabela com 4 colunas", "Gere um JSON com os perfis"). Isso facilita a ingestão dos resultados em sistemas de marketing.
*   **Iteração e Refinamento:** Use o LLM para refinar segmentos existentes. Em vez de começar do zero, peça: "Refine o Segmento A, focando em clientes que usam o recurso X."
*   **Foco na Ação:** O objetivo final da segmentação é a ação. Inclua no prompt a solicitação de sugestões de marketing ou comunicação para cada segmento identificado.

## Use Cases
*   **Marketing Personalizado:** Criação de campanhas de e-mail, anúncios e conteúdo de blog direcionados a dores e interesses específicos de cada segmento.
*   **Desenvolvimento de Produto:** Identificação de lacunas no mercado ou necessidades não atendidas ao analisar o feedback de segmentos específicos.
*   **Otimização de Preços:** Determinação de estratégias de preços e ofertas ideais para segmentos de alto valor ou sensíveis a preço.
*   **Previsão de Churn:** Uso de LLMs para analisar dados comportamentais e textuais de clientes em risco, permitindo intervenções proativas e personalizadas.
*   **Criação de Personas:** Geração rápida de personas detalhadas e realistas para guiar equipes de design, marketing e vendas.

## Pitfalls
*   **Prompts Vagos:** Pedir apenas "Segmentar meus clientes" sem fornecer dados, critérios ou o objetivo da segmentação. Isso leva a resultados genéricos e inúteis.
*   **Excesso de Dados Brutos:** Tentar alimentar o LLM com um arquivo CSV gigante. LLMs são melhores em processar dados sumarizados ou textuais. Para grandes volumes, use ferramentas de análise de dados e peça ao LLM para interpretar os *insights*.
*   **Confiança Excessiva:** Tratar a segmentação do LLM como verdade absoluta. A segmentação gerada por IA deve ser validada por analistas humanos e testada em campanhas reais.
*   **Ignorar o Contexto:** Não fornecer ao LLM o contexto do negócio (ex: tipo de produto, mercado-alvo, objetivos de receita), resultando em segmentos academicamente corretos, mas comercialmente irrelevantes.
*   **Viés nos Dados de Entrada:** Se os dados de entrada (ex: feedback de clientes) contiverem viés, o LLM irá amplificá-lo na segmentação, levando a estratégias de marketing injustas ou ineficazes.

## URL
[https://www.airops.com/prompts/data-analysis-marketing-prompts-ai](https://www.airops.com/prompts/data-analysis-marketing-prompts-ai)
