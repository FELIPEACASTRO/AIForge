# RFM Analysis Prompts

## Description
**RFM Analysis Prompts** são técnicas de Engenharia de Prompt focadas em alavancar o poder dos Grandes Modelos de Linguagem (LLMs) para automatizar e aprofundar a Análise RFM (Recência, Frequência, Valor Monetário). A Análise RFM é uma metodologia de segmentação de clientes que classifica os clientes com base em seu comportamento de compra. Em vez de apenas calcular as pontuações RFM e os segmentos, os prompts são usados para: 1. **Gerar Insights:** Criar descrições ricas e narrativas para cada segmento de clientes (ex: "Campeões", "Clientes em Risco"). 2. **Desenvolver Estratégias:** Sugerir ações de marketing hiperpersonalizadas, ofertas e campanhas de retenção ou reengajamento específicas para cada segmento. 3. **Estruturar a Saída:** Receber os resultados em formatos estruturados (como JSON) para integração direta em sistemas de Automação de Marketing ou CRM. Essa abordagem transforma dados brutos de segmentação em estratégias de negócios acionáveis e personalizadas.

## Examples
```
**Exemplo 1: Geração de Ofertas Personalizadas (Estruturado)**

```
**Papel:** Você é um especialista em análise de dados e marketing de varejo.
**Contexto:** Recebeu dados de análise de clustering RFM para 8 segmentos de clientes.
**Tarefa:** Para cada segmento, gere uma oferta de marketing e um insight sobre o grupo.
**Restrições:** O desconto máximo permitido é de 20%. As ofertas devem ser focadas em aumentar a Frequência ou o Valor Monetário.
**Dados de Análise (JSON/Markdown):**
{
  "segmento_1": {"nome": "Campeões", "R_avg": 5, "F_avg": 15, "M_avg": 500, "categoria_favorita": "Eletrônicos"},
  "segmento_2": {"nome": "Clientes em Risco", "R_avg": 120, "F_avg": 3, "M_avg": 150, "categoria_favorita": "Vestuário"}
}
**Formato de Saída (JSON):**
{
  "segmento_1": {"oferta": "Desconto de 10% em todos os novos lançamentos de Eletrônicos.", "insight": "Clientes de alto valor e lealdade, focados em novidades."},
  "segmento_2": {"oferta": "Frete grátis e 15% de desconto em qualquer item de Vestuário.", "insight": "Clientes que estão se tornando inativos; a reativação requer um incentivo forte."}
}
```

**Exemplo 2: Descrição Narrativa do Segmento**

```
**Prompt:** Descreva o segmento de clientes "Clientes em Risco" com uma narrativa de marketing de 100 palavras. O objetivo é que a equipe de marketing entenda rapidamente quem são e qual é a urgência. Use um tom profissional e direto.
**Dados:** Recência Média: 95 dias. Frequência Média: 2 compras. Valor Monetário Médio: R$ 180.
```

**Exemplo 3: Sugestão de Canais de Comunicação**

```
**Prompt:** Com base nas características do segmento "Hibernando" (R_avg: 200 dias, F_avg: 1, M_avg: R$ 50), sugira os 3 canais de comunicação mais eficazes para reengajamento e justifique brevemente cada escolha.
```

**Exemplo 4: Identificação de Padrões de Compra**

```
**Prompt:** Analise os dados de transação do segmento "Quase Lá" e identifique o padrão de compra mais comum (ex: compra sempre no final do mês, compra itens complementares).
**Dados:** [Lista de IDs de transação e datas/itens para o segmento]
```

**Exemplo 5: Prompt de Classificação de Segmento (Zero-Shot)**

```
**Prompt:** Classifique o seguinte cliente em um dos 4 segmentos RFM (Campeão, Leal, Em Risco, Perdido) e justifique a classificação: Recência: 15 dias, Frequência: 12 compras, Valor Monetário: R$ 850.
```
```

## Best Practices
**Estruturação de Dados:** Sempre forneça ao LLM dados de análise RFM pré-processados e estruturados (ex: resultados de clustering, médias de R, F, M por segmento). **Contextualização do Modelo:** Defina o papel do LLM (ex: "Você é um especialista em análise de dados e marketing") e o objetivo da tarefa. **Restrições Claras:** Inclua restrições específicas (ex: limite máximo de desconto, tom de voz da oferta) para garantir a relevância e a segurança da saída. **Saída Estruturada:** Utilize a capacidade de saída estruturada (JSON, Pydantic) do LLM para receber os insights e ofertas em um formato facilmente integrável a sistemas de CRM ou automação de marketing. **Iteração e Refinamento:** Use os insights gerados pelo LLM para refinar a segmentação RFM e as estratégias de marketing em um ciclo contínuo de melhoria.

## Use Cases
**Marketing Hiperpersonalizado:** Geração automática de textos de e-mail, notificações push ou mensagens de SMS com ofertas específicas para o comportamento de compra de cada segmento RFM. **Geração de Conteúdo para CRM:** Criação de descrições detalhadas e acionáveis para cada segmento de clientes dentro de um sistema de CRM, facilitando a tomada de decisão da equipe de vendas. **Otimização de Campanhas:** Uso dos insights do LLM para refinar os parâmetros de campanhas de anúncios (ex: qual segmento deve receber anúncios de reengajamento vs. qual deve receber anúncios de *upsell*). **Análise Preditiva:** Embora o RFM seja descritivo, os prompts podem ser usados para prever o próximo passo lógico do cliente (ex: "Qual a probabilidade de o segmento 'Em Risco' fazer uma nova compra nos próximos 30 dias, e qual oferta maximizaria essa chance?"). **Relatórios Automatizados:** Criação de relatórios de desempenho de segmentação RFM em linguagem natural, transformando tabelas e gráficos em narrativas de fácil compreensão para executivos.

## Pitfalls
**Alucinação de Estratégia:** O LLM pode gerar estratégias de marketing que parecem plausíveis, mas não são viáveis ou não se alinham com as políticas da empresa. **Solução:** Fornecer restrições claras e validar as sugestões com especialistas humanos. **Dados Brutos Não Estruturados:** Tentar alimentar o LLM com dados de transação brutos e extensos. **Solução:** O LLM é melhor para *interpretar* dados de análise (como médias de cluster) do que para *processar* grandes volumes de dados numéricos. **Viés de Segmentação:** Se os dados de entrada do RFM ou do clustering estiverem enviesados, o LLM amplificará esse viés nas descrições e ofertas. **Solução:** Garantir a qualidade e a representatividade dos dados de segmentação antes de usar o prompt. **Falta de Contexto de Negócio:** O LLM não conhece a margem de lucro ou o custo de aquisição do cliente. **Solução:** Incluir informações críticas de negócio no prompt (ex: "O custo de aquisição para este segmento é alto, priorize a retenção").

## URL
[https://www.ionio.ai/blog/how-to-leverage-llms-for-customer-segmentation-and-offer-generation](https://www.ionio.ai/blog/how-to-leverage-llms-for-customer-segmentation-and-offer-generation)
