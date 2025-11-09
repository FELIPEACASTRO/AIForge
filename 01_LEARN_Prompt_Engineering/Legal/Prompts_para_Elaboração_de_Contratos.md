# Prompts para Elaboração de Contratos

## Description
Um prompt para elaboração de contratos é uma instrução específica e detalhada fornecida a um modelo de linguagem de grande escala (LLM) para gerar, revisar ou analisar um documento legal com precisão jurídica. Diferente de um prompt genérico, o prompt legal deve incluir **contexto jurídico e jurisdicional**, especificar **cláusulas obrigatórias**, e indicar o **tom e o formato** desejados. Dominar essa técnica é crucial para profissionais do direito que buscam otimizar o tempo de elaboração e garantir a conformidade legal, utilizando a IA como uma ferramenta de aceleração, mas sempre sob a supervisão final de um advogado.

## Examples
```
1.  **Contrato de Prestação de Serviços:** "Redija um contrato de prestação de serviços de desenvolvimento de software no Brasil, incluindo cláusulas de confidencialidade, prazos de entrega e direitos de propriedade intelectual. Use tom formal e formato com cláusulas numeradas."
2.  **Contrato de Trabalho Internacional:** "Escreva um contrato de trabalho para um funcionário remoto na Espanha que responde a uma empresa com sede nos Estados Unidos. Inclua cláusulas de salário, jornada, benefícios e resolução de disputas sob jurisdição espanhola."
3.  **NDA com Cláusula de Dados:** "Gere um acordo de confidencialidade (NDA) entre duas empresas de varejo no Brasil, com cláusula de proteção de dados pessoais conforme a Lei Geral de Proteção de Dados (LGPD). Use linguagem formal e divida o documento em seções."
4.  **Contrato de Compra e Venda de Imóvel:** "Atue como um advogado imobiliário brasileiro. Elabore a minuta de um contrato de promessa de compra e venda de um imóvel residencial localizado em São Paulo, Brasil. O prompt deve incluir: 1) Qualificação completa das partes (vendedor e comprador); 2) Preço total e condições de pagamento (sinal e parcelas); 3) Cláusula de arrependimento e multa; 4) Prazo para entrega das chaves e posse. Use a técnica *Layering* para estruturar o documento."
5.  **Termo de Uso (Software SaaS):** "Crie um Termo de Uso (ToS) para um software SaaS (Software as a Service) B2B. O documento deve ser regido pela lei do estado de Delaware, EUA. Inclua seções sobre: 1) Licença de uso; 2) Limitação de responsabilidade; 3) Política de rescisão por violação; 4) Foro de eleição para resolução de disputas. O tom deve ser claro e conciso."
6.  **Aditivo Contratual:** "Com base no contrato de prestação de serviços anexo (instrução), redija um aditivo contratual para prorrogar o prazo de vigência por mais 12 meses e reajustar o valor mensal em 15% (índice IGPM). O aditivo deve ser conciso e fazer referência clara às cláusulas originais alteradas. Jurisdição: Brasil."
7.  **Cláusula de Não Concorrência (CNO):** "Gere uma cláusula de não concorrência (CNO) robusta para um contrato de trabalho de um executivo sênior. A cláusula deve especificar: 1) Duração de 24 meses após o término do contrato; 2) Área geográfica de restrição (América Latina); 3) Compensação financeira (valor mensal durante a restrição); 4) Penalidade por violação. Peça à IA para justificar a razoabilidade da cláusula com base na jurisprudência brasileira."
```

## Best Practices
1.  **Definição de Papel e Contexto:** Peça à IA para agir como um especialista (ex: "advogado especialista em direito trabalhista") e defina o tipo de contrato e setor.
2.  **Especificação Jurisdicional:** Sempre inclua a jurisdição e a legislação aplicável (ex: "sob a Lei Federal do Trabalho do Brasil").
3.  **Inclusão de Cláusulas Essenciais:** Liste as cláusulas obrigatórias (Objeto, Prazos, Confidencialidade, Rescisão, etc.).
4.  **Formato e Tom Detalhados:** Solicite explicitamente o formato (cláusulas numeradas, seções) e o tom (formal, claro).
5.  **Uso de Técnicas Avançadas:** Utilize *Few-Shot* (com exemplos de cláusulas), *Chain-of-Thought* (divisão em etapas lógicas) ou *Layering* (construção modular) para contratos complexos.
6.  **Revisão Humana Obrigatória:** A IA acelera, mas a validação final por um advogado é inegociável para garantir a precisão jurídica.

## Use Cases
1.  **Elaboração Inicial de Contratos:** Geração de rascunhos de contratos padrão (Prestação de Serviços, Compra e Venda, Locação).
2.  **Adaptação Jurisdicional:** Criação de contratos para diferentes países, especificando a legislação local e normas aplicáveis.
3.  **Inclusão de Cláusulas Específicas:** Geração de cláusulas complexas como NDA (Acordo de Confidencialidade), DPI (Direitos de Propriedade Intelectual) ou cláusulas de proteção de dados (LGPD, GDPR).
4.  **Contratos de Trabalho Remoto:** Elaboração de contratos de trabalho internacionais, considerando diferentes jurisdições e regimes de trabalho.
5.  **Revisão e Análise:** Geração de prompts para análise de contratos existentes, identificação de riscos, resumo de termos e comparação com *benchmarks* legais.

## Pitfalls
1.  **Ambiguidade de Jurisdição:** Não especificar o país e a norma concreta, resultando em um contrato legalmente inválido ou inaplicável.
2.  **Falta de Especificidade:** Prompts vagos que resultam em contratos genéricos e inutilizáveis, exigindo extensa revisão manual.
3.  **Inclusão de Dados Sensíveis:** Inserir informações pessoais ou confidenciais diretamente no prompt, comprometendo a segurança e a privacidade.
4.  **Omissão da Revisão Humana:** Confiar cegamente no resultado da IA sem a validação final de um profissional do direito, o que é inaceitável em documentos legais.
5.  **Uso de Prompt Único para Múltiplas Jurisdições:** Usar o mesmo prompt para países diferentes, ignorando a necessidade de adaptação legal e regulatória.

## URL
[https://blog.getdarwin.ai/pt-br/es/c%C3%B3mo-crear-el-prompt-perfe...](https://blog.getdarwin.ai/pt-br/es/c%C3%B3mo-crear-el-prompt-perfe...)
