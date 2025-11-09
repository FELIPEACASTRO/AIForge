# Budgeting Prompts

## Description
**Budgeting Prompts** (Prompts de Orçamentação) são comandos de engenharia de prompt estruturados e detalhados, fornecidos a modelos de Linguagem Grande (LLMs) ou ferramentas de Inteligência Artificial (IA) especializadas, com o objetivo de auxiliar no planejamento, análise, otimização e gestão de orçamentos, tanto no âmbito pessoal quanto no corporativo. Essa técnica transforma a IA em um assistente financeiro virtual, capaz de processar dados brutos (como listas de despesas, rendas e objetivos), aplicar regras financeiras estabelecidas (como a regra 50/30/20), identificar padrões de gastos, sugerir cortes de custos e criar planos financeiros detalhados e personalizados. A eficácia do *Budgeting Prompt* reside na clareza e na riqueza de contexto fornecido, permitindo que a IA gere saídas acionáveis, como planilhas, checklists e relatórios comparativos.

## Examples
```
1.  **Orçamento Pessoal com Regra 50/30/20:**
    "Minha renda líquida mensal é de R$ 7.500. Desejo seguir a regra 50/30/20 (50% Essenciais, 30% Desejos, 20% Investimentos). Crie um modelo de orçamento mensal em formato de tabela, mostrando os valores ideais para cada categoria e fornecendo três exemplos de despesas típicas para cada uma delas."

2.  **Otimização de Gastos:**
    "Atue como um consultor financeiro. Analise a seguinte lista de despesas mensais: [Insira lista de despesas com valores]. Identifique as três áreas com maior potencial de corte de gastos sem comprometer meu bem-estar e sugira um plano de ação para economizar R$ 500 por mês, apresentando o resultado em um formato de checklist."

3.  **Planejamento de Despesas Sazonais:**
    "Todo início de ano tenho despesas sazonais: IPVA (R$ 2.500), Matrícula Escolar (R$ 3.200) e Seguro do Carro (R$ 1.800). Crie um planejamento financeiro mensal para diluir o custo total dessas despesas ao longo dos 12 meses, indicando o valor exato que devo reservar mensalmente para cada item."

4.  **Criação de Rastreador de Orçamento (Tracker):**
    "Crie um template de rastreador de orçamento simples para [Excel/Google Sheets]. O template deve incluir colunas para Data, Categoria (Renda, Moradia, Alimentação, Transporte, Lazer, Investimentos), Descrição, Valor e Saldo. Inclua também a fórmula para calcular o saldo acumulado."

5.  **Proposta de Orçamento de Projeto (Empresarial):**
    "Com base no 'Plano de Projeto Alpha' e nos 'Relatórios de Despesas do Q3', elabore uma proposta de orçamento detalhada para a equipe de desenvolvimento enxuta (lean team). A proposta deve incluir custos de software, salários (3 desenvolvedores juniores, 1 sênior), e uma reserva de contingência de 15%."

6.  **Comparação de Estratégias de Orçamento:**
    "Compare as abordagens de orçamentação *Zero-Based Budgeting* (Orçamento Base Zero) e *Incremental Budgeting* (Orçamento Incremental) para uma startup de SaaS. Descreva os prós e contras de cada método e recomende qual seria mais adequado para uma empresa em fase de alto crescimento, justificando a escolha."

7.  **Análise de Viabilidade de Meta Financeira:**
    "Meu objetivo é economizar R$ 50.000 em 3 anos para dar entrada em um imóvel. Minha renda mensal é de R$ 6.000 e minhas despesas fixas somam R$ 3.500. Calcule o valor mensal necessário para atingir essa meta e sugira ajustes no meu orçamento atual para viabilizar a economia, assumindo um retorno de investimento conservador de 0,5% ao mês."
```

## Best Practices
As melhores práticas para a criação de *Budgeting Prompts* maximizam a precisão e a utilidade da resposta da IA:

*   **Fornecimento de Contexto Detalhado:** Sempre inclua dados específicos como renda líquida, despesas fixas, objetivos de economia e o período de tempo (mensal, anual). Quanto mais dados e contexto (ex: "sou estudante", "sou expatriado", "tenho dívidas de cartão"), mais personalizada será a saída.
*   **Definição de Regras e Métodos:** Especifique o método de orçamentação desejado (ex: 50/30/20, Orçamento Base Zero) ou as restrições que a IA deve seguir.
*   **Role-Playing (Definição de Papel):** Inicie o prompt definindo o papel da IA (ex: "Atue como meu consultor financeiro", "Você é um analista de custos sênior"), o que tende a refinar o tom e a profundidade da análise.
*   **Especificação do Formato de Saída:** Peça explicitamente o formato desejado (ex: "em formato de tabela", "como um checklist", "com fórmulas para Google Sheets") para garantir uma saída estruturada e fácil de usar.
*   **Utilização de Dados Externos (em ferramentas avançadas):** Em plataformas que permitem a integração de documentos, referencie arquivos específicos (ex: "analise o 'Relatório Financeiro Q1'") para que a IA possa realizar análises baseadas em dados reais e internos.

## Use Cases
A técnica de *Budgeting Prompts* é aplicável em diversas situações financeiras:

| Categoria | Caso de Uso | Descrição |
| :--- | :--- | :--- |
| **Finanças Pessoais** | Criação de Orçamento Mensal | Elaborar planos de gastos baseados em renda e metas (ex: regra 50/30/20). |
| **Planejamento de Metas** | Viabilidade de Objetivos | Calcular quanto é preciso economizar mensalmente para atingir metas de longo prazo (ex: entrada de imóvel, viagem). |
| **Otimização de Despesas** | Análise de Padrões de Gastos | Identificar e sugerir cortes em áreas de consumo excessivo. |
| **Finanças Empresariais** | Propostas de Orçamento | Gerar orçamentos para projetos, departamentos ou equipes específicas. |
| **Conformidade e Fiscal** | Resumo de Regulamentações | Obter resumos de atualizações fiscais e criar checklists de conformidade orçamentária. |
| **Criação de Ferramentas** | Geração de Templates | Criar templates de planilhas financeiras (trackers) com fórmulas prontas para uso. |

## Pitfalls
Existem armadilhas comuns que podem comprometer a eficácia dos *Budgeting Prompts*:

*   **Falta de Especificidade:** Prompts vagos ou genéricos (ex: "Crie um orçamento") resultarão em respostas igualmente genéricas e pouco úteis. A IA precisa de números e objetivos claros.
*   **Viés e Imprecisão dos Dados de Entrada:** Se os dados de renda ou despesas fornecidos estiverem incorretos ou incompletos, a análise da IA será falha (*Garbage In, Garbage Out*).
*   **Confiança Excessiva (Over-reliance):** A IA é uma ferramenta de apoio e não substitui a expertise de um planejador financeiro certificado. Decisões financeiras complexas e de alto risco exigem validação humana.
*   **Risco de Segurança de Dados:** **Nunca** se deve incluir informações pessoais sensíveis, como CPF, senhas, números de cartão de crédito ou dados bancários completos, nos prompts. A maioria dos LLMs não é projetada para lidar com dados PII (Informações de Identificação Pessoal) de forma segura.
*   **Ignorar a Revisão:** É crucial revisar e ajustar o orçamento gerado pela IA para garantir que ele reflita a realidade e as prioridades pessoais ou empresariais.

## URL
[https://www.infomoney.com.br/minhas-financas/prompts-para-planejamento-orcamento-ia/](https://www.infomoney.com.br/minhas-financas/prompts-para-planejamento-orcamento-ia/)
