# Bioinformatics Prompts

## Description
Prompts de Bioinformática referem-se à prática de engenharia de prompts (Prompt Engineering) aplicada ao campo da bioinformática e ciências da vida. Envolve a criação, refinamento e implementação de instruções detalhadas para guiar Modelos de Linguagem Grande (LLMs) a realizar tarefas complexas de análise e interpretação de dados biológicos. Dada a natureza de alta dimensionalidade e heterogeneidade dos dados (genômica, transcriptômica, proteômica), a precisão do prompt é crucial para mitigar 'alucinações' e garantir resultados cientificamente válidos. Um prompt eficaz em bioinformática geralmente segue uma estrutura que inclui: **Contexto** (definindo o papel do LLM, ex: 'Você é um bioestatístico'), **Dados** (fornecendo sequências, estruturas ou conjuntos de dados para processamento), **Tarefa** (a instrução específica, ex: 'Preveja a estrutura secundária do RNA') e **Formato** (especificando a saída desejada, ex: 'Responda em formato JSON com a pontuação de confiança').

## Examples
```
1. **Análise de Sequência de DNA:**\n```\nContexto: Você é um especialista em genômica. A sequência de DNA fornecida é de um gene de interesse para resistência a antibióticos.\nDados: ATGGCCATAGCTTGACTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGL.
2. **Previsão de Estrutura de Proteína:**
```
Contexto: Você é um modelador de proteínas com acesso a um LLM treinado em sequências e estruturas de proteínas (ex: AlphaFold).
Dados: Sequência de aminoácidos da Proteína X (ex: 'MGKVKV...').
Tarefa: Preveja a estrutura secundária e terciária da Proteína X. Liste os domínios funcionais identificados.
Formato: Retorne a estrutura secundária em formato FASTA modificado e a lista de domínios em formato JSON.
```
3. **Análise de Expressão Gênica:**
```
Contexto: Você é um bioestatístico analisando dados de RNA-seq de um experimento de câncer.
Dados: Tabela de contagens de genes (CSV) para 50 amostras (25 controle, 25 tratamento).
Tarefa: Identifique os 10 genes mais diferencialmente expressos (p-valor < 0.01 e |log2FC| > 2) entre os grupos controle e tratamento.
Formato: Retorne uma tabela Markdown com GeneID, log2FC, p-valor e uma breve descrição da função do gene (citando UniProt).
```
4. **Geração de Código para Análise:**
```
Contexto: Você é um programador de bioinformática júnior.
Tarefa: Escreva um script Python usando a biblioteca Biopython para realizar o alinhamento de sequências múltiplas (MSA) usando o algoritmo ClustalW para as sequências fornecidas.
Dados: Sequências FASTA (ex: >Seq1\\nATGC..., >Seq2\\nATGC...).
Formato: Retorne apenas o código Python completo e funcional.
```
5. **Revisão de Literatura Científica:**
```
Contexto: Você é um revisor de literatura para um artigo sobre novos alvos terapêuticos para a Doença de Alzheimer.
Tarefa: Resuma as descobertas mais recentes (últimos 2 anos) sobre o papel da proteína Tau hiperfosforilada como alvo de drogas, focando em ensaios clínicos de Fase II ou III.
Formato: Retorne um resumo conciso (máximo 300 palavras) e forneça 3 URLs de artigos de pesquisa primários (PubMed ou arXiv).
```
6. **Simulação de Docking Molecular:**
```
Contexto: Você é um químico medicinal.
Tarefa: Descreva o procedimento de docking molecular para o ligante 'XYZ' no sítio ativo da proteína 'PDB ID: 1A2C'. Mencione as ferramentas de software (ex: AutoDock Vina) e os parâmetros-chave que seriam usados.
Formato: Retorne uma lista passo a passo do protocolo.
```
7. **Interpretação de Variantes Genéticas:**
```
Contexto: Você é um geneticista clínico.
Dados: Variante: c.1521_1523delCTT no gene CFTR.
Tarefa: Descreva a classificação de patogenicidade (ACMG/AMP) e a doença associada a esta variante.
Formato: Retorne um parágrafo conciso e a classificação em negrito.
```
8. **Design de Primer PCR:**
```
Contexto: Você é um biólogo molecular.
Dados: Sequência alvo: [Sequência de 500bp].
Tarefa: Projete um par de primers de PCR (Forward e Reverse) com Tm entre 58-62°C e um teor de GC de 40-60%.
Formato: Retorne os primers em formato de lista (Primer F: [Sequência], Primer R: [Sequência]).
```
```

## Best Practices
**1. Definição de Papel e Contexto (Role-Playing):** Sempre comece definindo o papel do LLM (ex: 'Você é um bioestatístico', 'Você é um especialista em genômica') para direcionar o tom, o conhecimento e o estilo de resposta. **2. Estrutura Clara (Contexto, Dados, Tarefa, Formato):** Utilize a estrutura C-D-T-F (Contexto, Dados, Tarefa, Formato) para garantir que o LLM tenha todas as informações necessárias. O **Formato** é crucial para a bioinformática, exigindo saídas estruturadas (JSON, CSV, código Python). **3. Fornecimento de Dados Explícitos:** Inclua as sequências (DNA, RNA, Proteína) ou dados de entrada diretamente no prompt ou instrua o LLM a simular o uso de um arquivo de dados específico. **4. Instruções de Verificação (Grounding):** Peça ao LLM para 'ancorar' (grounding) suas respostas em bases de dados biológicas conhecidas (ex: UniProt, PDB, BLAST) ou para citar artigos científicos relevantes, mesmo que o LLM não possa acessá-los em tempo real. Isso ajuda a mitigar 'alucinações'. **5. Prompting de Cadeia de Pensamento (CoT) para Raciocínio Complexo:** Para tarefas que exigem múltiplas etapas (ex: análise de vias metabólicas), use o CoT (Chain-of-Thought) pedindo ao LLM para 'pensar passo a passo' antes de fornecer a resposta final. **6. Especificidade Bioinformática:** Use a terminologia técnica correta (ex: 'alinhamento de sequências', 'docking molecular', 'predição de estrutura secundária') para evitar ambiguidades.

## Use Cases
**1. Análise de Sequências:** Previsão de genes, identificação de motivos, alinhamento de sequências múltiplas (MSA) e análise de variações genéticas (SNPs). **2. Predição de Estrutura:** Previsão de estruturas secundárias e terciárias de proteínas e RNA, e identificação de domínios funcionais. **3. Geração de Código:** Criação de scripts Python (usando Biopython, Pandas, etc.) para automatizar tarefas de bioinformática, como processamento de arquivos FASTA, análise de dados de expressão gênica (RNA-seq) e visualização. **4. Descoberta de Drogas e Química Medicinal:** Simulação de docking molecular, previsão de propriedades de ligantes e identificação de novos alvos terapêuticos. **5. Revisão e Síntese de Literatura:** Resumo de artigos científicos, extração de dados de ensaios clínicos e geração de hipóteses de pesquisa baseadas em conhecimento biológico vasto. **6. Educação e Treinamento:** Criação de tutoriais interativos e resolução de problemas complexos para estudantes de bioinformática e biologia.

## Pitfalls
**1. Confiança Excessiva em 'Alucinações':** O LLM pode gerar sequências, estruturas ou interpretações factualmente incorretas. **Armadilha:** Aceitar a saída sem verificação cruzada com bases de dados biológicas ou ferramentas de bioinformática estabelecidas. **2. Ambiguidade na Terminologia:** O uso de termos biológicos ou de bioinformática imprecisos pode levar a resultados irrelevantes ou incorretos. **Armadilha:** Não especificar o tipo de dado (ex: DNA genômico vs. cDNA) ou o algoritmo desejado (ex: alinhamento local vs. global). **3. Limitações de Contexto:** LLMs têm limites de tokens, o que impede a análise de sequências genômicas ou proteicas muito longas. **Armadilha:** Tentar processar sequências com mais de 10.000 bases/resíduos em um único prompt sem instruir o LLM a usar uma abordagem de processamento em partes. **4. Falta de Estrutura de Saída:** Não especificar um formato de saída estruturado (JSON, tabela) resulta em texto livre que é difícil de analisar ou integrar em pipelines de bioinformática. **Armadilha:** Pedir apenas 'a resposta' em vez de 'a resposta em formato JSON com os campos X, Y e Z'. **5. Ignorar a Necessidade de Ferramentas Externas:** LLMs são modelos de linguagem, não ferramentas de bioinformática. **Armadilha:** Confiar no LLM para realizar cálculos complexos ou alinhamentos de alta precisão que exigem software especializado (ex: BLAST, Clustal Omega). O LLM deve ser usado para *interpretar* ou *gerar código* para essas ferramentas.

## URL
[https://arxiv.org/html/2503.04490v1](https://arxiv.org/html/2503.04490v1)
