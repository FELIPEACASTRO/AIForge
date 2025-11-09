# Patent Application Prompts

## Description
Prompts para Pedidos de Patente (Patent Application Prompts) referem-se à formulação estratégica de instruções e entradas de texto para modelos de Linguagem Grande (LLMs) para auxiliar na redação de documentos de patente. O objetivo é guiar a IA para gerar seções específicas do pedido, como o Relatório Descritivo, as Reivindicações, o Resumo e o Histórico da Invenção, garantindo que o texto seja tecnicamente preciso, legalmente sólido e em conformidade com os requisitos de órgãos reguladores como o INPI (Brasil) ou USPTO (EUA). A técnica se concentra em fornecer contexto detalhado da invenção, definir o papel da IA (por exemplo, como um advogado de patentes júnior) e aplicar restrições de formato e estilo legal para otimizar a eficiência e a qualidade do rascunho inicial. É uma ferramenta de aumento de produtividade para profissionais de Propriedade Intelectual.

## Examples
```
**1. Rascunho de Reivindicação Principal (Método):**
```
Aja como um advogado de patentes sênior. Com base na descrição da invenção fornecida abaixo, rascunhe a Reivindicação 1 (independente) para um método. A reivindicação deve ser escrita no formato 'preâmbulo + compreendendo as etapas de:' e focar apenas nos elementos essenciais que definem a novidade.

[DESCRIÇÃO DA INVENÇÃO]: Um novo método para purificar água usando nanopartículas de grafeno funcionalizadas com um composto de prata, onde as etapas incluem: 1) Síntese das nanopartículas, 2) Funcionalização com prata em solução aquosa, 3) Adição da solução de nanopartículas à água contaminada, 4) Exposição a luz UV por 10 minutos, 5) Filtração para remover as nanopartículas.
```

**2. Geração de Histórico da Invenção (Background):**
```
Redija a seção de 'Histórico da Invenção' para um pedido de patente sobre [TÍTULO DA INVENÇÃO]. O texto deve ter no máximo 400 palavras e abordar o estado da técnica atual, os problemas não resolvidos e as limitações das soluções existentes. Use um tom formal e técnico.

[TÍTULO DA INVENÇÃO]: Sistema de Otimização de Rota Logística baseado em Aprendizado por Reforço.
```

**3. Expansão Detalhada da Especificação:**
```
Expanda o seguinte trecho da especificação técnica para incluir detalhes de implementação, materiais e variações de design. O texto expandido deve ser didático e fornecer suporte para as reivindicações. Mantenha o estilo de escrita em terceira pessoa e voz passiva.

[TRECHO ORIGINAL]: O módulo de processamento (101) é configurado para receber dados de sensores (102) e executar um algoritmo de filtragem.
```

**4. Geração de Resumo (Abstract):**
```
Com base no conjunto de reivindicações e na descrição da invenção, escreva um Resumo conciso (máximo de 150 palavras) para o pedido de patente. O resumo deve descrever a essência da invenção, o problema resolvido e o principal elemento de novidade.
```

**5. Revisão de Clareza e Ambiguidade Legal:**
```
Analise a seguinte reivindicação. Identifique e sugira revisões para qualquer linguagem ambígua, termos vagos ou frases que possam ser interpretadas de forma muito ampla ou muito restrita, comprometendo a clareza legal.

[REIVINDICAÇÃO]: Um dispositivo para melhorar a eficiência energética, compreendendo um componente de controle que ajusta a potência de forma inteligente.
```

**6. Geração de Títulos e Subtítulos para a Especificação:**
```
Gere uma estrutura de títulos e subtítulos (Tabela de Conteúdo) para a seção de 'Descrição Detalhada da Invenção' de uma patente sobre [TEMA]. Inclua seções padrão como 'Breve Descrição dos Desenhos' e 'Exemplos de Implementação'.

[TEMA]: Dispositivo Vestível para Monitoramento Contínuo de Glicose Não Invasivo.
```

**7. Rascunho de Reivindicação Dependente:**
```
Rascunhe três reivindicações dependentes (Reivindicações 2, 3 e 4) que restrinjam a Reivindicação 1 fornecida abaixo. As restrições devem focar em (a) um material específico, (b) uma faixa de parâmetro numérico e (c) uma etapa adicional opcional.

[REIVINDICAÇÃO 1]: Um método para purificar água, compreendendo as etapas de: adicionar nanopartículas funcionalizadas a uma fonte de água contaminada; e filtrar a água para remover as nanopartículas.
```
```

## Best Practices
**1. Separe Instruções do Contexto:** Use separadores claros (como `###` ou `---`) para distinguir as instruções do prompt do texto de entrada (o contexto da invenção). Isso garante que o modelo de IA entenda o que é comando e o que é dado. **2. Seja Específico e Detalhado:** Forneça instruções precisas sobre o resultado desejado, incluindo contexto, formato, estilo, e limites de palavras/parágrafos. Evite descrições vagas como "curto" ou "várias frases". **3. Use Modelos de Última Geração:** Priorize modelos mais avançados (como GPT-4 ou superior) para tarefas complexas de redação de patentes, pois eles são mais confiáveis, criativos e capazes de lidar com instruções mais matizadas. **4. Ajuste o Nível de Criatividade (Temperatura):** Mantenha o nível de 'temperatura' (criatividade/aleatoriedade) baixo (próximo de 0) para a maioria das seções técnicas e legais, como reivindicações e especificações, para reduzir o risco de 'alucinações' e garantir precisão. **5. Encadear Prompts (Fine-Tuning):** Use a saída de um prompt como entrada para um prompt subsequente (encadeamento) para refinar o texto. Por exemplo, gere um rascunho e, em seguida, use um segundo prompt para revisar a concisão ou o tom legal.

## Use Cases
**1. Rascunho Inicial de Seções:** Geração rápida de rascunhos iniciais para seções menos críticas ou mais padronizadas do pedido, como o Histórico da Invenção (Background) e o Resumo (Abstract). **2. Expansão da Especificação:** Detalhamento de conceitos e elementos técnicos para garantir que as reivindicações tenham suporte descritivo suficiente (enablement). **3. Geração de Reivindicações Dependentes:** Criação de um conjunto de reivindicações dependentes que restrinjam a reivindicação principal, explorando diferentes escopos de proteção. **4. Tradução e Adaptação de Terminologia:** Tradução de documentos técnicos ou adaptação de terminologia para o jargão legal de patentes exigido (por exemplo, transformar a linguagem de engenharia em linguagem de patentes). **5. Análise de Estado da Técnica:** Sumarização e análise de documentos de patentes existentes (estado da técnica) para identificar lacunas e definir a novidade da invenção. **6. Revisão de Estilo e Formato:** Revisão de rascunhos para garantir consistência de estilo, gramática e aderência a formatos específicos de numeração e referência de desenhos. **7. Criação de Perguntas para o Inventor:** Geração de uma lista de perguntas detalhadas para o inventor para preencher lacunas de informação técnica antes da redação final.

## Pitfalls
**1. Alucinações e Imprecisão Técnica:** A IA pode gerar detalhes técnicos incorretos ou inconsistentes (alucinações), o que é fatal em documentos legais como patentes. **2. Violação de Confidencialidade:** O uso de LLMs públicos com dados confidenciais da invenção pode violar acordos de confidencialidade (NDAs) e comprometer a novidade da invenção. **3. Linguagem Vaga ou Ambiguidade Legal:** A IA pode usar linguagem genérica ou ambígua que não atende ao padrão de clareza e precisão exigido pelas leis de patentes (por exemplo, a exigência de 'suficiência descritiva'). **4. Falha em Atender aos Requisitos Formais:** A IA pode não aderir estritamente aos requisitos de formato e estrutura específicos de cada escritório de patentes (INPI, USPTO, EPO), exigindo revisão manual extensiva. **5. Ausência de Suporte para Reivindicações:** Gerar reivindicações sem garantir que cada elemento esteja explicitamente suportado e descrito na especificação (Relatório Descritivo), o que pode levar à rejeição. **6. Problemas de Inventorship:** O uso da IA levanta questões complexas sobre quem é o inventor legal, especialmente em jurisdições que não reconhecem a IA como inventora. **7. Confiança Excessiva:** Confiar cegamente no rascunho da IA sem uma revisão técnica e legal aprofundada por um profissional qualificado.

## URL
[https://www.patentclaimmaster.com/blog/best-practices-for-gpt-prompt-engineering-when-patent-drafting/](https://www.patentclaimmaster.com/blog/best-practices-for-gpt-prompt-engineering-when-patent-drafting/)
