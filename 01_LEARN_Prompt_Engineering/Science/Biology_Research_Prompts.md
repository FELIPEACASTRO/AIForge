# Biology Research Prompts

## Description
**Prompts de Pesquisa em Biologia** referem-se à prática de elaborar instruções detalhadas e contextualmente ricas para **Grandes Modelos de Linguagem (LLMs)**, como GPT-4 ou Gemini, com o objetivo de otimizar a pesquisa e a análise de dados no campo da Biologia e Ciências da Vida [1] [2]. Essa técnica é crucial para transformar LLMs de ferramentas de propósito geral em assistentes especializados capazes de lidar com a complexidade e a terminologia específica de áreas como **Genômica, Proteômica, Biologia Estrutural e Descoberta de Medicamentos** [3]. A engenharia de prompt eficaz em Biologia envolve a definição clara do papel do modelo, a inclusão de dados biológicos de entrada (sequências de DNA/proteínas, artigos científicos) e a especificação de um formato de saída estruturado para facilitar a análise e a validação dos resultados. O objetivo principal é mitigar a tendência dos LLMs a "alucinar" (gerar informações factualmente incorretas) e garantir que as respostas sejam cientificamente precisas e relevantes para o contexto biológico fornecido [2].

## Examples
```
1. **Análise de Sequência de Proteína (Proteômica):**
```
Atue como um bioinformata. Analise a seguinte sequência de aminoácidos: [SEQUÊNCIA DE AMINOÁCIDOS]. Preveja sua estrutura secundária, possíveis domínios funcionais (usando a notação Pfam) e sugira três potenciais interações proteína-proteína relevantes em humanos. Apresente a resposta em uma tabela Markdown.
```

2. **Revisão de Literatura e Extração de Dados (Farmacologia):**
```
Extraia os seguintes dados do artigo científico fornecido: [TEXTO DO ARTIGO]. 1. Nome do composto principal estudado. 2. Concentração inibitória (IC50) contra a linhagem celular [NOME DA LINHAGEM]. 3. Mecanismo de ação molecular proposto. 4. Efeitos colaterais mais comuns. Responda estritamente no formato JSON com as chaves 'composto', 'ic50', 'mecanismo', 'efeitos_colaterais'.
```

3. **Geração de Hipóteses (Genômica):**
```
Considere o gene humano [NOME DO GENE] e a doença [NOME DA DOENÇA]. Com base na função conhecida do gene e nas vias metabólicas associadas, formule três hipóteses de pesquisa testáveis sobre como uma mutação de perda de função neste gene poderia levar ao fenótipo da doença. Estruture cada hipótese com: Título, Racional e Método Experimental Sugerido.
```

4. **Tradução de Código Biológico (Genética):**
```
Traduza a seguinte sequência de DNA (5' para 3') em todas as três possíveis fases de leitura. Identifique o quadro de leitura aberto (ORF) mais longo e a sequência de proteína correspondente. A sequência de DNA é: [SEQUÊNCIA DE DNA].
```

5. **Simulação de Experimento e Resolução de Problemas (Biologia Molecular):**
```
Você é um pesquisador de bancada. Descreva o protocolo passo a passo para realizar uma PCR em tempo real (qPCR) para quantificar a expressão do mRNA do gene [NOME DO GENE] em amostras de tecido hepático. Inclua a lista de reagentes, as condições de ciclagem e como calcular o valor de Ct.
```

6. **Análise de Dados Estruturados (Ecologia):**
```
Analise a tabela de dados de diversidade de espécies fornecida: [TABELA DE DADOS]. Calcule o Índice de Shannon-Wiener (H) e o Índice de Uniformidade de Pielou (J) para cada local de amostragem. Interprete os resultados e sugira a principal causa biológica para a diferença de diversidade entre o Local A e o Local B.
```
```

## Best Practices
**Especificidade e Contextualização:** Sempre defina o papel do LLM (ex: "Atue como um bioinformata experiente") e forneça o máximo de contexto biológico possível (ex: espécie, via metabólica, tipo de dado). **Uso de Dados de Entrada:** Para tarefas de extração ou análise, inclua o texto ou dados brutos (sequências, artigos, relatórios) diretamente no prompt (ou use a funcionalidade de upload de arquivo, se disponível). **Formato de Saída Estruturado:** Peça o resultado em um formato fácil de processar, como JSON, Markdown (tabelas), ou CSV, para facilitar a análise subsequente. **Validação Cruzada:** Sempre valide os resultados gerados pelo LLM com bases de dados biológicas confiáveis (ex: UniProt, PDB) ou literatura científica revisada por pares, especialmente para predições e sínteses críticas. **Prompt Iterativo:** Comece com um prompt simples e refine-o em várias etapas, adicionando restrições e detalhes até obter a precisão desejada (Processo Iterativo).

## Use Cases
**Descoberta de Medicamentos:** Acelerar a identificação de alvos terapêuticos, prever a toxicidade de compostos e otimizar a estrutura de moléculas candidatas. **Análise de Genômica e Proteômica:** Previsão de estrutura de proteínas, identificação de variantes genéticas patogênicas e anotação funcional de genes desconhecidos. **Revisão Sistemática de Literatura:** Extração rápida de dados específicos (ex: IC50, doses, resultados de ensaios clínicos) de grandes volumes de artigos científicos. **Design Experimental:** Geração de protocolos de laboratório detalhados, otimização de condições de reação e sugestão de controles experimentais. **Educação e Treinamento:** Criação de cenários de estudo de caso complexos e simulação de discussões científicas para estudantes e pesquisadores juniores. **Bioinformática:** Auxílio na depuração de scripts de análise de dados e na tradução de pseudocódigo para linguagens de programação como Python ou R.

## Pitfalls
**Alucinação Factual:** O LLM pode gerar informações que parecem plausíveis, mas são cientificamente incorretas ou inexistentes (o maior risco). **Viés de Treinamento:** O modelo pode favorecer informações de fontes mais comuns ou em inglês, ignorando descobertas críticas em outras línguas ou de periódicos menos populares. **Falta de Contexto Biológico:** Prompts genéricos levam a respostas superficiais ou irrelevantes. A ausência de detalhes cruciais (ex: organismo, condição experimental) é um erro comum. **Limitação de Token:** A incapacidade de processar grandes volumes de dados brutos (sequências genômicas inteiras, múltiplos artigos) em um único prompt devido aos limites de token do modelo. **Interpretação de Dados Complexos:** LLMs não são ferramentas de análise estatística e podem interpretar mal dados numéricos complexos ou gráficos sem o contexto adequado. **Confiança Excessiva:** A aceitação cega da saída do LLM sem a devida verificação e validação científica.

## URL
[https://www.certara.com/blog/best-practices-for-ai-prompt-engineering-in-life-sciences/](https://www.certara.com/blog/best-practices-for-ai-prompt-engineering-in-life-sciences/)
