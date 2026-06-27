# Contract Review Prompts (Prompts de Revisão de Contratos)

## Description
A técnica de **Prompts de Revisão de Contratos** (Contract Review Prompts) é um conjunto de instruções estruturadas e detalhadas fornecidas a um modelo de Linguagem Grande (LLM) para que ele realize a análise, sumarização, identificação de riscos, verificação de conformidade legal e sugestão de otimizações em documentos contratuais. Essa técnica transforma o LLM em um assistente jurídico virtual, permitindo que profissionais do direito e de negócios acelerem drasticamente o processo de *due diligence* e negociação de contratos, focando em pontos críticos como cláusulas abusivas, lacunas de informação, e alinhamento com a legislação vigente (e.g., LGPD, CDC) [1]. A eficácia reside na capacidade de atribuir um **persona** (ex: advogado de direito empresarial), definir o **objetivo** (ex: identificar riscos para o contratante) e solicitar um **formato de saída** específico (ex: tabela comparativa), elevando a precisão e a utilidade da análise [2].

## Examples
```
**1. Identificação de Risco e Abusividade (Persona Específica)**
`Atue como um advogado de direito do consumidor. Analise o contrato de adesão a seguir e identifique 3 cláusulas que podem ser consideradas abusivas ou que representem alto risco para o consumidor. Justifique cada ponto e sugira uma redação mais equilibrada. [Colar contrato]`

**2. Sumarização e Linguagem Clara**
`Atue como um especialista em linguagem clara. Resuma o contrato de prestação de serviços a seguir em 10 bullet points, focando exclusivamente nos direitos e deveres da minha parte (Contratado), prazos de pagamento e condições de rescisão. Use linguagem simples e direta. [Colar contrato]`

**3. Verificação de Conformidade Legal (LGPD)**
`Atue como um consultor jurídico especializado em LGPD. Analise o contrato de parceria a seguir e identifique se ele está em total conformidade com a Lei Geral de Proteção de Dados (LGPD) em relação ao tratamento de dados pessoais. Aponte 2 pontos de não conformidade e proponha a cláusula de correção ideal. [Colar contrato]`

**4. Otimização de Cláusulas para Negociação**
`Atue como um negociador jurídico. Na cláusula 'Propriedade Intelectual' do contrato de desenvolvimento de software, que favorece a Contratante, sugira 3 opções de redação alternativa que a tornem mais equilibrada ou vantajosa para a minha parte (Desenvolvedor). Explique o impacto legal de cada opção.`

**5. Identificação de Lacunas e Omissões**
`Atue como um auditor de contratos. Analise o seguinte contrato de compra e venda de imóvel. Quais 3 informações ou cláusulas cruciais para a segurança do comprador estão faltando, considerando a legislação brasileira? Sugira onde elas deveriam ser inseridas no documento.`

**6. Prompt Mestre para Análise Completa**
`**Persona:** Advogado especialista em direito empresarial e IA. **Tipo de Contrato:** Contrato de Prestação de Serviços (Marketing Digital). **Meu Papel:** Contratado. **Missão:** Analisar o contrato a seguir e: a) Identificar 3 pontos de alto risco ou ambiguidade para o Contratado; b) Para cada ponto, propor 2 opções de redação alternativa que me protejam mais; c) Sugerir 3 perguntas cruciais para fazer a um advogado humano. **Formato:** Tabela estruturada. [Colar contrato]`
```

## Best Practices
**1. Defina a Persona Jurídica:** Comece o prompt instruindo a IA a atuar como um especialista específico (ex: "Atue como um advogado de direito do consumidor" ou "jurista experiente em M&A"). Isso alinha o foco da análise e a linguagem da resposta. **2. Especifique o Objetivo e o Formato:** Seja claro sobre o que você quer (ex: "Identificar 3 cláusulas de alto risco" ou "Resumir em 10 bullet points"). Peça o resultado em um formato estruturado, como uma tabela, para facilitar a leitura e comparação. **3. Forneça Contexto e Papel:** Indique qual é o seu papel no contrato (Contratante, Contratado, Comprador) e o tipo de contrato. Isso permite que a IA avalie o risco sob a sua perspectiva. **4. Itere e Refine:** Use o resultado da primeira análise para prompts de acompanhamento. Por exemplo, após identificar uma cláusula de risco, peça um novo prompt para "sugerir 3 opções de redação alternativa" para essa cláusula específica. **5. Use o Prompt Mestre:** Para revisões complexas, utilize um "Prompt Mestre" que inclua todos os elementos de contexto, objetivo e formato em uma única estrutura, como o exemplo fornecido, para garantir uma análise completa e multifacetada [1].

## Use Cases
**1. *Due Diligence* Rápida:** Análise inicial de grandes volumes de contratos em fusões e aquisições (M&A) ou auditorias, para identificar rapidamente passivos e riscos. **2. Otimização de Cláusulas:** Geração de sugestões de redação mais favoráveis para a parte do usuário, fortalecendo a posição em negociações. **3. Verificação de Conformidade:** Garantir que os contratos estejam alinhados com novas regulamentações (ex: LGPD, leis setoriais) antes da assinatura. **4. Treinamento e Educação:** Uso dos prompts para simular cenários de negociação e treinar novos advogados ou equipes de vendas sobre os pontos críticos de um contrato. **5. Padronização:** Criação de um *checklist* de revisão automatizado para garantir que todos os contratos de um mesmo tipo (ex: NDA, Termos de Serviço) contenham as cláusulas essenciais e de proteção [1] [2].

## Pitfalls
**1. Confiança Cega (Alucinações):** O maior erro é confiar cegamente na saída da IA. Os LLMs podem **"alucinar"** (inventar) precedentes legais, artigos de lei ou interpretações. A revisão final por um profissional humano é indispensável. **2. Falta de Contexto:** Não especificar o tipo de contrato, o papel do usuário (Contratante/Contratado) ou a jurisdição legal (ex: Brasil, EUA) leva a análises genéricas e irrelevantes. **3. Limitação de *Token*:** Contratos muito longos podem exceder o limite de *tokens* do modelo, resultando em análises incompletas ou truncadas. É necessário dividir o documento em seções para análise. **4. Ausência de Documentos Anexos:** A IA não pode analisar documentos anexos ou referências externas que não foram incluídas no prompt, o que pode levar a uma avaliação incompleta do risco contratual. **5. Prompts Vagos:** Solicitações como "Revise este contrato" são ineficazes. A falta de um objetivo claro (ex: "foco em multas", "foco em rescisão") resulta em uma saída superficial [3].

## URL
[https://treinamentosaf.com.br/contratos-a-prova-de-falhas-7-prompts-ia-revisam-em-minutos-2025/](https://treinamentosaf.com.br/contratos-a-prova-de-falhas-7-prompts-ia-revisam-em-minutos-2025/)
