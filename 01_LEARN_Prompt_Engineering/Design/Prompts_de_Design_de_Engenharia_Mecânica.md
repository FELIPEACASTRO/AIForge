# Prompts de Design de Engenharia Mecânica

## Description
**Prompts de Design de Engenharia Mecânica** são instruções estruturadas e detalhadas fornecidas a modelos de linguagem grandes (LLMs) ou ferramentas de IA generativa para auxiliar em tarefas complexas de engenharia. Esta categoria de prompts visa alavancar a capacidade da IA para: **1. Geração de Conceitos:** Criar designs inovadores, mecanismos e soluções para problemas de engenharia. **2. Análise e Simulação:** Acelerar a configuração de simulações (FEA, CFD), interpretar resultados e prever o comportamento de sistemas. **3. Otimização:** Sugerir melhorias em geometria, material e processo de fabricação para atender a critérios de desempenho, custo e peso. **4. Documentação e Conformidade:** Gerar esboços de Especificações de Design de Produto (PDS), listas de verificação de segurança e documentação técnica. A eficácia desses prompts reside na sua **especificidade**, na **inclusão de restrições de engenharia** (materiais, normas, custos) e na **estrutura de saída** que facilita a integração dos resultados da IA no fluxo de trabalho de design assistido por computador (CAD/CAE). Eles transformam a IA de uma ferramenta de conversação em um assistente de design técnico e analítico.

## Examples
```
1. **Geração de Conceito de Mecanismo:** "Crie 3 conceitos de mecanismo inovadores para converter movimento rotacional contínuo em movimento linear intermitente, com uma taxa de 1:5. As restrições são: material de alumínio 6061, espaço máximo de 100x100x50mm e custo de fabricação abaixo de $50. Para cada conceito, detalhe o princípio de operação, as vantagens e as desvantagens."

2. **Otimização Topológica:** "Aplique otimização topológica a uma peça de suporte estrutural com as seguintes condições de contorno: Carga de 500N aplicada no centro superior, fixação nas 4 extremidades inferiores. O objetivo é reduzir a massa em 40% mantendo um fator de segurança de 1.5. O material é Aço Inoxidável 316. Descreva a geometria otimizada resultante e a distribuição de tensão esperada."

3. **Análise de Falha (FEA/CFD):** "Você é um especialista em Análise de Elementos Finitos. Analise o relatório de simulação de fadiga anexo (considere que o relatório foi anexado). Identifique os 3 pontos de maior concentração de tensão e sugira modificações de design específicas (raios de concordância, espessura) para reduzir a tensão em pelo menos 20% nesses pontos críticos."

4. **Seleção de Materiais:** "Recomende um material para um componente que será exposto a um ambiente de alta temperatura (400°C) e alta corrosão (ácido sulfúrico diluído). As propriedades críticas são: resistência à tração > 500 MPa e densidade < 8 g/cm³. Forneça uma tabela comparativa de 3 opções, incluindo custo por kg e justificativa para a seleção."

5. **Design Biomimético:** "Nosso problema de engenharia é criar um sistema de amortecimento de vibração passivo e leve para um drone. Identifique um sistema biológico (ex: estrutura óssea, folha de planta) que resolva um problema semelhante de absorção de energia. Descreva o mecanismo biológico e proponha uma adaptação para o design do amortecedor do drone, incluindo um esboço conceitual."

6. **Esboço de PDS:** "Gere um esboço detalhado de uma Especificação de Design de Produto (PDS) para um 'Braço Robótico Colaborativo de Baixo Custo'. Inclua seções obrigatórias para Métricas de Desempenho (ex: precisão de repetição, carga útil), Restrições de Fabricação (ex: impressão 3D, usinagem CNC) e Normas de Segurança (ex: ISO 10218)."

7. **Resolução de Problemas de Fabricação:** "Estamos enfrentando um problema de empenamento excessivo (warpage) durante a injeção de plástico de uma peça de polipropileno. Analise o problema e sugira 3 alterações no design do molde (ex: localização de porta, espessura da parede, sistema de refrigeração) para minimizar o empenamento, justificando cada sugestão com princípios de moldagem por injeção."
```

## Best Practices
**Defina o Papel e o Contexto:** Comece o prompt definindo o papel da IA (ex: "Você é um Engenheiro Mecânico Sênior especializado em Dinâmica de Fluidos Computacional") e o contexto do projeto (ex: "Estamos projetando um novo sistema de refrigeração para um servidor de alta densidade"). **Seja Específico e Estruturado:** Use listas numeradas, marcadores e seções claras (INPUT, OUTPUT, CONSTRAINTS) para estruturar o prompt. Especifique o formato de saída desejado (ex: "Forneça a resposta em formato de tabela Markdown com colunas para Parâmetro, Valor e Justificativa"). **Forneça Dados de Entrada:** Inclua todos os dados relevantes, como especificações de material, geometria inicial, condições de contorno, e requisitos de desempenho. **Use a Metodologia de Design:** Incorpore metodologias de engenharia, como Design Generativo, Otimização Topológica, Análise de Elementos Finitos (FEA) ou Biomimética, diretamente no prompt. **Iteração e Refinamento:** Use a saída da IA como entrada para o próximo prompt, criando um ciclo de refinamento. Por exemplo, peça uma análise de falha e, em seguida, peça sugestões de design para mitigar a falha identificada.

## Use Cases
nan

## Pitfalls
**Vaguidão:** Prompts genéricos como "Me ajude com o design de um motor" resultam em respostas superficiais. A falta de especificidade sobre o componente, material, carga e objetivo é o erro mais comum. **Ignorar Restrições:** Não incluir restrições de engenharia (custo, peso, normas, material) leva a soluções impraticáveis. A IA precisa de limites claros para gerar designs realistas. **Ausência de Formato de Saída:** Não especificar o formato (tabela, lista, código, texto estruturado) torna a saída da IA difícil de ser processada ou integrada em ferramentas de engenharia. **Confiar Cegamente:** Tratar a saída da IA como verdade absoluta. A IA é uma ferramenta de sugestão e otimização; o engenheiro deve sempre validar os resultados com simulações e testes reais. **Prompts Longos e Desorganizados:** Um prompt com muitas informações sem estrutura clara pode confundir a IA, levando a omissões ou interpretações erradas das instruções. Use a formatação para organizar as seções.

## URL
[https://innovation.world/ai-prompts-for-mechanical-engineering/](https://innovation.world/ai-prompts-for-mechanical-engineering/)
