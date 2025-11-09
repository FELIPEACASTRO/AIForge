# Funnel Analysis Prompts

## Description
**Prompts de Análise de Funil** (*Funnel Analysis Prompts*) são uma técnica avançada de *Prompt Engineering* focada em instruir Modelos de Linguagem Grande (LLMs) a processar dados de funis de marketing, vendas, produto (AARRR) ou experiência do usuário (UX), a fim de identificar gargalos (*leakages*), calcular taxas de conversão e sugerir otimizações acionáveis. A técnica exige que o usuário forneça à IA o **contexto** (o que está sendo analisado), os **dados brutos** (métricas por estágio) e a **instrução** (o que a IA deve fazer com esses dados, geralmente atuando como um analista ou consultor). O objetivo é transformar dados quantitativos em *insights* qualitativos e estratégicos, permitindo que a IA simule um processo de consultoria analítica. Esta técnica é fundamental para a otimização de *growth* e *performance* em ambientes digitais.

## Examples
```
**Exemplo 1: E-commerce (Funil de Compra)**
**Ato:** Atue como um Analista de Otimização de Conversão (CRO) para um e-commerce de moda.
**Contexto:** O funil de compra é: **Visita à Página Inicial -> Visualização de Produto -> Adição ao Carrinho -> Checkout -> Compra Concluída**.
**Dados:** Nas últimas 4 semanas, tivemos 100.000 visitas, 40.000 visualizações de produto, 8.000 adições ao carrinho, 2.000 inícios de checkout e 500 compras concluídas. O AOV (Valor Médio do Pedido) é R$ 300.
**Instrução:** Identifique o estágio com a maior taxa de abandono, calcule o potencial de receita incremental se a taxa de conversão desse estágio fosse aumentada em 10%, e sugira 3 táticas de CRO específicas para esse ponto de vazamento.

**Exemplo 2: SaaS (Funil de Ativação)**
**Ato:** Atue como um Gerente de Produto focado em Ativação (Activation Manager) para um software SaaS B2B de gestão de projetos.
**Contexto:** O funil de ativação é: **Cadastro -> Instalação do App -> Criação do Primeiro Projeto -> Convidar Membro da Equipe -> Uso Semanal**.
**Dados:** Dos 5.000 novos cadastros no último mês, 3.500 instalaram o app, 1.500 criaram o primeiro projeto, 500 convidaram um membro e apenas 100 atingiram o uso semanal.
**Instrução:** Analise a transição de "Instalação do App" para "Criação do Primeiro Projeto". Qual é a principal barreira percebida? Crie um *prompt* de e-mail de *onboarding* de 3 etapas para reduzir esse *drop-off*, focando no valor imediato (*Aha! Moment*).

**Exemplo 3: Geração de Leads (Funil de Marketing)**
**Ato:** Atue como um Especialista em Automação de Marketing para uma empresa de consultoria B2B.
**Contexto:** O funil de leads é: **Visita ao Blog -> Download de Ebook (MQL) -> Solicitação de Demonstração (SQL) -> Reunião Agendada**.
**Dados:** No último trimestre, 50.000 visitas ao blog geraram 5.000 downloads, 200 solicitações de demonstração e 50 reuniões agendadas.
**Instrução:** Concentre-se na conversão de MQL para SQL. Analise a taxa de conversão e sugira 5 critérios de *lead scoring* (pontuação de leads) que, se implementados, poderiam melhorar a qualidade dos leads que chegam ao time de vendas.

**Exemplo 4: Conteúdo e Engajamento (Funil de Mídia)**
**Ato:** Atue como um Estrategista de Conteúdo para um canal de notícias online.
**Contexto:** O funil de engajamento é: **Visualização de Artigo -> Rolagem de 50% -> Clique em Artigo Relacionado -> Inscrição na Newsletter**.
**Dados:** Onde está o maior *drop-off*? Proponha uma alteração no *call-to-action* (CTA) da newsletter e um novo formato de conteúdo (ex: quiz, infográfico) para o estágio de "Clique em Artigo Relacionado" para aumentar a conversão para a newsletter.

**Exemplo 5: Análise de Retenção (Funil de Churn)**
**Ato:** Atue como um Cientista de Dados de Cliente (Customer Data Scientist) para um serviço de streaming por assinatura.
**Contexto:** O funil de *churn* (abandono) é: **Assinatura Ativa -> Uso Semanal -> Redução de Uso -> Cancelamento -> Reativação**.
**Dados:** 10.000 usuários ativos. 500 reduziram o uso no último mês. Desses, 100 cancelaram. 10 reativaram.
**Instrução:** Descreva o perfil dos 100 usuários que cancelaram (com base em dados fictícios de engajamento: assistiram menos de 2 horas/semana, não usaram a função de lista de favoritos). Com base nesse perfil, crie um *prompt* para um modelo de IA gerar 3 ofertas de retenção personalizadas e o momento ideal para enviá-las.

**Exemplo 6: Funil de Produto (UX/UI)**
**Ato:** Atue como um Designer de UX/UI.
**Contexto:** O funil de uso de um recurso é: **Abertura do Recurso -> Interação com o Filtro -> Aplicação do Filtro -> Salvar Configuração**.
**Dados:** 5.000 aberturas do recurso, 4.000 interações com o filtro, 1.500 aplicações do filtro, 500 salvamentos de configuração.
**Instrução:** O *drop-off* entre "Interação com o Filtro" e "Aplicação do Filtro" é alto. Liste 3 hipóteses de usabilidade (UX) para esse vazamento e sugira um teste A/B de interface (UI) para validar a hipótese mais provável.

**Exemplo 7: Funil de Vendas Complexas (B2B)**
**Ato:** Atue como um Consultor de Estratégia de Vendas.
**Contexto:** O funil de vendas é: **Prospecção -> Qualificação -> Proposta -> Negociação -> Fechamento**.
**Dados:** 100 prospecções, 50 qualificações, 20 propostas enviadas, 10 negociações, 5 fechamentos.
**Instrução:** Analise a conversão de "Proposta" para "Negociação". Qual é a taxa de conversão? Crie um *template* de *prompt* para o time de vendas usar no CRM, solicitando à IA uma análise preditiva do risco de perda para cada proposta, com base em 3 variáveis de entrada (ex: tempo de resposta do cliente, número de *stakeholders* envolvidos, valor da proposta).
```

## Best Practices
**1. Defina o Funil com Clareza:** Antes de tudo, mapeie os estágios do funil de forma lógica e sequencial (ex: AARRR - Aquisição, Ativação, Retenção, Receita, Referência). A clareza do funil é a base para a análise da IA. **2. Forneça Dados Estruturados e Contextualizados:** Apresente os dados de forma organizada (tabela, lista) e inclua o contexto de negócio (indústria, público-alvo, modelo de receita). **3. Atribua um "Ato" (Persona):** Peça à IA para atuar como um especialista específico (ex: "Atue como um Analista de CRO", "Atue como um Cientista de Dados"). Isso melhora a qualidade e o foco das respostas. **4. Solicite Ações e Hipóteses:** Não peça apenas a identificação do problema, mas também a sugestão de ações, testes A/B ou hipóteses de causa raiz. **5. Use a Análise para Cenários:** Peça à IA para calcular o impacto potencial de melhorias (ex: "Calcule o aumento de receita se a conversão do estágio X for de 5% para 7%").

## Use Cases
**1. Otimização de Conversão (CRO):** Identificar o ponto exato de maior abandono em um funil de e-commerce ou SaaS para focar esforços de otimização. **2. Análise de Ativação de Produto:** Entender por que novos usuários não estão completando o *onboarding* ou atingindo o *Aha! Moment* em aplicativos. **3. Estratégia de Retenção e *Churn*:** Analisar o funil de abandono para prever e prevenir o *churn* (cancelamento) de clientes, gerando ofertas de retenção personalizadas. **4. *Lead Scoring* e Qualificação:** Ajudar equipes de marketing e vendas a refinar critérios de pontuação de leads (MQL para SQL) com base em dados de conversão histórica. **5. Simulação de Cenários (*What-If*):** Calcular o impacto financeiro potencial de melhorias hipotéticas na taxa de conversão de um estágio específico. **6. Diagnóstico de UX/UI:** Aplicar a lógica de funil a fluxos de uso de recursos de software para identificar falhas de usabilidade.

## Pitfalls
**1. Dados Incompletos ou Viesados:** Fornecer à IA dados parciais, desatualizados ou com viés de atribuição (ex: atribuir todas as vendas ao último clique). A IA só pode analisar o que é fornecido. **2. Funil Mal Definido:** Não mapear claramente os estágios do funil de forma lógica e sequencial, resultando em análises confusas ou irrelevantes. **3. Falta de Contexto de Negócio:** Não informar à IA o modelo de negócio (SaaS, e-commerce, B2B), o público-alvo ou os objetivos específicos, levando a recomendações genéricas. **4. Ignorar a Qualidade dos Dados:** Não incluir na análise a validação da integridade dos dados (ex: duplicação de eventos, *bots*), o que pode distorcer as taxas de conversão. **5. Foco Excessivo em Métricas de Vaidade:** Pedir à IA para otimizar métricas de topo de funil (ex: visualizações) sem conectar o impacto nas métricas de fundo (ex: receita, LTV). **6. Recomendações Não Acionáveis:** Solicitar análises sem pedir explicitamente por *táticas* ou *hipóteses de teste* concretas, resultando em *insights* teóricos. **7. Não Segmentar a Análise:** Analisar o funil como um todo, sem segmentar por canal (orgânico vs. pago), dispositivo (mobile vs. desktop) ou coorte de usuários, perdendo a oportunidade de identificar vazamentos específicos. **8. Confundir Causa com Correlação:** Aceitar as conclusões da IA sem aplicar o próprio julgamento crítico, especialmente em análises de causa raiz, onde a correlação pode ser confundida com causalidade.

## URL
[https://founderpal.ai/prompts-examples/funnel-analysis](https://founderpal.ai/prompts-examples/funnel-analysis)
