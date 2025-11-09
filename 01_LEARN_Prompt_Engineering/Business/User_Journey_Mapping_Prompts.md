# User Journey Mapping Prompts

## Description
**Prompts de Mapeamento da Jornada do Usuário (User Journey Mapping Prompts)** são uma categoria de técnicas de *Prompt Engineering* focadas em alavancar Modelos de Linguagem Grande (LLMs) para simular, analisar e otimizar a experiência de um cliente ou usuário com um produto ou serviço.

Em vez de apenas gerar conteúdo, esses prompts instruem a IA a atuar como um especialista (ex: Consultor Omnichannel, Designer de Serviço, Cientista de Dados) para desconstruir a jornada em seus componentes essenciais: **estágios**, **pontos de contato (touchpoints)**, **ações**, **pensamentos**, **emoções** e **dores (pain points)**.

O objetivo principal é transformar dados brutos ou descrições de cenários em um mapa estruturado e acionável, permitindo que equipes de UX, Produto e Marketing identifiquem lacunas, momentos de verdade e oportunidades de inovação. A técnica é particularmente poderosa para simular cenários complexos (ex: acessibilidade, comparação com concorrentes) e para gerar rapidamente a primeira versão de um mapa, que pode ser refinada por humanos.

## Examples
```
**1. Blueprint da Jornada Completa (End-to-End):**
`"Construa um mapa detalhado da jornada do cliente para [PRODUTO/SERVIÇO] focado na persona [DESCRIÇÃO DA PERSONA]. Divida a jornada em: Conscientização, Consideração, Compra, Onboarding, Uso e Retenção. Para cada estágio, liste: (a) objetivos do cliente, (b) pontos de contato, (c) emoções, (d) dores e (e) KPIs mensuráveis. Conclua com os três maiores 'momentos da verdade' e ações sugeridas para otimizá-los."`

**2. Cenário Específico da Persona (Deep-Dive):**
`"Crie um cenário granular para a persona 'João, 35 anos, Gerente de Marketing' comprando [PRODUTO]. Descreva passo a passo o que ele pensa, sente e faz em cada ponto de contato, desde a primeira conscientização até um mês após a compra. Destaque os altos e baixos emocionais. Finalize com uma tabela resumindo os pontos de fricção e correções rápidas (quick-win fixes)." `

**3. Auditoria Omnichannel de Pontos de Contato:**
`"Assuma o papel de um consultor omnichannel. Liste todos os pontos de contato online e offline que um cliente encontra ao interagir com [MARCA] (site, app, redes sociais, e-mail, suporte por telefone). Para cada ponto de contato, especifique seu propósito primário, métrica de sucesso, problemas típicos e uma melhoria de Experiência do Cliente (CX) recomendada, ranqueada por impacto vs. esforço."`

**4. Visualização da Curva de Emoção:**
`"Imagine que você é um designer de serviço traçando uma curva de emoção. Descreva, em sequência, a intensidade emocional do cliente (-5 a +5) em cada etapa ao assinar e usar [SERVIÇO DE ASSINATURA]. Forneça uma explicação narrativa para cada ponto de dados e recomende intervenções de design para achatar vales negativos e amplificar picos positivos."`

**5. Ideação de Jornada Futura (Future-State):**
`"Você está facilitando um sprint de design. Imagine uma jornada de 'estado futuro' de 12 meses para [MARCA] que elimina os três principais pontos de dor de hoje: [DOR 1], [DOR 2] e [DOR 3]. Descreva pontos de contato e tecnologias inovadoras (ex: chat de IA, suporte preditivo) introduzidas em cada estágio e explique como elas transformam a experiência do cliente. Forneça um roadmap de implementação priorizado por ROI."`

**6. Otimização Orientada por Dados (Funil SaaS):**
`"Atue como um cientista de dados otimizando nossa jornada de onboarding SaaS. Dados do funil: Inscrição [70%], Primeiro Momento de Valor [40%], Ativação [25%]. Identifique os dois estágios com maior queda. Hipotetize as causas raiz, projete três ideias de teste A/B para resolvê-las e defina as métricas de sucesso para cada teste."`

**7. Revisão de Acessibilidade e Inclusão:**
`"Avalie a jornada de um cliente com [NECESSIDADE DE ACESSIBILIDADE, ex: deficiência visual] usando nosso site de e-commerce. Detalhe as barreiras encontradas durante a descoberta do produto e o checkout. Recomende correções compatíveis com WCAG e melhorias de design inclusivo, indicando ganhos rápidos versus melhorias de longo prazo."`
```

## Best Practices
**1. Fornecer Contexto Detalhado:** Sempre comece o prompt definindo o problema, o produto/serviço e a persona alvo. Quanto mais contexto (como dados de funil, estágios atuais da jornada, ou dor de negócio), mais rica será a saída da IA.

**2. Especificar o Formato de Saída:** Peça explicitamente o formato desejado (ex: "Entregue o resultado em uma tabela Markdown com 5 colunas", "Use uma curva de emoção de -5 a +5", "Conclua com um playbook de ações").

**3. Usar Frameworks Conhecidos:** Integre frameworks de design e negócios (como **Jobs-to-Be-Done**, **Omnichannel**, ou **WCAG** para acessibilidade) para guiar a IA a um resultado mais estruturado e profissional.

**4. Focar em Pontos de Dor Específicos:** Em vez de mapear a jornada inteira, use prompts para focar em estágios críticos (ex: "Pós-Compra", "Onboarding SaaS") ou problemas específicos (ex: "Taxa de abandono no checkout").

**5. Solicitar Ações Otimizadas:** Não peça apenas o mapa; peça sugestões de otimização, como "três maiores 'momentos da verdade'", "melhorias ranqueadas por impacto vs. esforço" ou "ideias de teste A/B".

## Use Cases
**1. Design de Produto e UX:**
*   **Identificação de Lacunas:** Revelar pontos de fricção e momentos de frustração que levam ao abandono ou *churn*.
*   **Priorização de Recursos:** Usar a análise de "impacto vs. esforço" da IA para decidir quais melhorias de UX desenvolver primeiro.

**2. Marketing e Vendas:**
*   **Criação de Conteúdo:** Mapear os pensamentos e emoções do cliente em cada estágio para criar mensagens de marketing mais ressonantes e direcionadas.
*   **Otimização de Funil:** Usar prompts orientados por dados para identificar os estágios de maior queda no funil de vendas e sugerir testes A/B.

**3. Estratégia de Negócios e Inovação:**
*   **Benchmarking Competitivo:** Comparar a jornada do cliente com a de concorrentes para identificar vantagens estratégicas e oportunidades de "saltar à frente" na experiência do cliente (CX).
*   **Ideação de Estado Futuro:** Criar visões de longo prazo (12-18 meses) para a experiência do cliente, incorporando novas tecnologias (IA, AR/VR) e eliminando dores atuais.

**4. Suporte ao Cliente e Retenção:**
*   **Desenho do Pós-Compra:** Criar o playbook ideal para os primeiros 90 dias após a compra, focando em ativação, suporte proativo e redução do risco de *churn*.
*   **Revisão de Acessibilidade:** Avaliar a jornada sob a perspectiva de usuários com necessidades específicas (ex: deficiência visual) para garantir conformidade e inclusão.

## Pitfalls
**1. Falta de Contexto Específico:** O erro mais comum é usar prompts genéricos. A IA não pode mapear uma jornada útil sem detalhes sobre o **produto**, a **persona** e o **problema de negócio** que se tenta resolver.

**2. Confundir o Mapa com a Realidade:** O mapa gerado pela IA é uma **hipótese estruturada**, não a verdade absoluta. É um erro usá-lo sem validação por meio de pesquisa real com usuários (entrevistas, dados analíticos).

**3. Ignorar a Voz do Cliente (VoC):** Não incluir dados qualitativos (citações de entrevistas, reclamações de suporte) no prompt resulta em um mapa estéril e baseado em suposições genéricas.

**4. Foco Excessivo em Estágios Positivos:** A IA pode tender a otimizar demais os "altos" emocionais. O valor real do mapeamento está em identificar e resolver os "baixos" (pontos de dor e fricção).

**5. Não Especificar o Formato:** Pedir apenas "Crie um mapa da jornada" sem definir a estrutura (tabela, lista, formato de saída) pode levar a uma resposta desorganizada e difícil de usar.

**6. Não Pedir Ações:** Um mapa sem ações de otimização é apenas um documento descritivo. O erro é não solicitar explicitamente que a IA sugira intervenções e prioridades.

## URL
[https://medium.com/@slakhyani20/10-chatgpt-prompts-for-customer-journey-mapping-14c667b1b451](https://medium.com/@slakhyani20/10-chatgpt-prompts-for-customer-journey-mapping-14c667b1b451)
