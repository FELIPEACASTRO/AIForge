# Prompts de Engenharia Química

## Description
A Engenharia de Prompt em Engenharia Química é a aplicação estratégica de modelos de linguagem grandes (LLMs) para acelerar a pesquisa, o desenvolvimento e a otimização de processos químicos. Ela envolve a criação de instruções precisas e contextuais para guiar a IA na realização de tarefas complexas, como simulação de processos, otimização de reatores, síntese de materiais, análise de segurança e conformidade regulatória [1] [2]. O foco principal é superar as limitações dos LLMs, como "alucinações" e falta de conhecimento de domínio específico, fornecendo contexto químico e de engenharia detalhado, exemplos de poucos disparos (few-shot prompting) e formatos de saída estruturados [3]. Esta técnica é crucial para transformar LLMs de ferramentas de propósito geral em assistentes de domínio especializado, capazes de lidar com a complexidade e a necessidade de precisão factual inerentes à engenharia química [4].

## Examples
```
**1. Otimização de Reator:** "Atue como um Engenheiro de Otimização de Processos. Para a reação de hidrogenação de benzeno em ciclo-hexano (C6H6 + 3H2 -> C6H12) em um reator PFR, com temperatura de entrada de 400 K e pressão de 30 bar, sugira 3 conjuntos de condições de vazão e razão molar que maximizem a conversão (acima de 95%) e minimizem o custo de energia. Apresente a saída em uma tabela Markdown com as colunas: 'Cenário', 'Vazão Molar Total (mol/s)', 'Razão H2:Benzeno', 'Conversão Estimada (%)', 'Custo Relativo de Energia'."

**2. Análise de Segurança (HAZOP):** "Como um Especialista em Segurança de Processos, realize uma análise HAZOP (Hazard and Operability Study) para uma coluna de destilação operando sob vácuo. Considere o parâmetro 'Pressão' e o desvio 'Mais Pressão'. Identifique as possíveis causas, as consequências para o processo e as ações de mitigação recomendadas. Formate a resposta como um relatório de segurança conciso."

**3. Síntese de Materiais:** "Sou um químico de materiais. Quero sintetizar uma Estrutura Metalorgânica (MOF) com base em zinco e ligantes tereftalato. Usando o método de síntese solvotermal, forneça um protocolo de laboratório passo a passo, incluindo a massa exata dos reagentes (para 1g de produto final), o solvente ideal, a temperatura e o tempo de reação. Cite a fonte de dados (ex: artigo científico) para o protocolo."

**4. Resolução de Problemas de Processo:** "O reator CSTR da planta de produção de polietileno está apresentando uma queda inesperada na taxa de polimerização. As variáveis de entrada (temperatura, concentração de monômero, concentração de catalisador) estão dentro das especificações. Liste 5 hipóteses de falha, começando pela mais provável, e sugira um teste de diagnóstico para cada uma. Responda no formato de lista numerada."

**5. Conformidade Regulatória:** "Para a descarga de efluentes de uma planta de fertilizantes no Brasil, quais são os principais parâmetros de qualidade da água regulamentados pelo CONAMA (Conselho Nacional do Meio Ambiente)? Crie uma tabela com o 'Parâmetro', o 'Limite Máximo Permitido (LMP)' e a 'Resolução CONAMA' correspondente. Foco em Nitrogênio Total e Fósforo Total."

**6. Projeto de Trocador de Calor:** "Calcule a área de transferência de calor necessária para um trocador de calor de casco e tubo (Shell and Tube) para resfriar 10 kg/s de óleo quente de 150°C para 80°C, usando água de resfriamento que entra a 25°C e sai a 40°C. Forneça o coeficiente global de transferência de calor (U) típico para esta aplicação (óleo/água) e o cálculo da Média Logarítmica da Diferença de Temperatura (LMTD). Apresente o resultado final da área em m²."
```

## Best Practices
**1. Contextualização Profunda:** Sempre forneça o contexto químico e de engenharia completo. Inclua a reação específica, as condições operacionais (temperatura, pressão, vazão), o tipo de reator e as restrições de segurança/custo. **2. Few-Shot Prompting (Exemplos):** Para tarefas complexas como extração de dados de síntese ou otimização de parâmetros, inclua 2-4 exemplos de entrada-saída para demonstrar o formato e o raciocínio desejados. **3. Definição de Papel (Role-Playing):** Comece o prompt definindo o LLM como um "Engenheiro Químico Sênior" ou "Especialista em Segurança de Processos" para alinhar a resposta com o conhecimento de domínio. **4. Saída Estruturada:** Solicite a saída em formatos estruturados (JSON, tabelas Markdown, listas numeradas) para facilitar a análise e a integração com outras ferramentas de engenharia. **5. Validação Cruzada:** Sempre use a saída do LLM como ponto de partida ou sugestão, e não como verdade absoluta. A validação com simulações, dados experimentais ou normas de engenharia é obrigatória.

## Use Cases
nan

## Pitfalls
**1. Alucinações Fatuais (Factual Hallucinations):** O LLM pode gerar dados termodinâmicos, cinéticos ou de segurança incorretos. **Contramedida:** Sempre valide números críticos (pontos de ebulição, entalpias, limites de explosividade) com bases de dados confiáveis (ex: NIST, DIPPR). **2. Falta de Conhecimento de Domínio Específico:** LLMs de propósito geral podem falhar em nuances de engenharia, como a diferença entre um reator CSTR ideal e um real. **Contramedida:** Use prompts de "Few-Shot" ou forneça dados de entrada de simulação (ex: equações de balanço de massa/energia) para contextualizar o modelo. **3. Viés de Dados de Treinamento:** O modelo pode favorecer soluções comuns ou antigas, ignorando inovações recentes ou soluções proprietárias. **Contramedida:** Peça explicitamente por "soluções inovadoras" ou "alternativas não convencionais" e restrinja a pesquisa a um período de tempo (ex: "publicado após 2023"). **4. Ignorar Restrições de Engenharia:** O LLM pode sugerir soluções termodinamicamente possíveis, mas economicamente inviáveis ou mecanicamente impraticáveis. **Contramedida:** Inclua restrições de custo, material e operabilidade no prompt (ex: "A solução deve usar aço inoxidável 316 e ter um custo de capital 10% menor que o design atual").

## URL
[https://pubs.acs.org/doi/10.1021/acscentsci.4c01935](https://pubs.acs.org/doi/10.1021/acscentsci.4c01935)
