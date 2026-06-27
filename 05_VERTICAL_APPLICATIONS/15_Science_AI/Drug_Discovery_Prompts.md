# Drug Discovery Prompts

## Description
A Engenharia de Prompt para Descoberta de Medicamentos (Drug Discovery Prompts) é a arte e a ciência de criar instruções otimizadas para Modelos de Linguagem Grandes (LLMs) e Modelos Multimodais (MLLMs) com o objetivo de acelerar e aprimorar as etapas do ciclo de desenvolvimento de fármacos. Esta técnica permite que pesquisadores e químicos medicinais utilizem a capacidade de processamento de linguagem natural dos LLMs para tarefas complexas, como a **triagem virtual**, **otimização de moléculas**, **previsão de propriedades ADMET** (Absorção, Distribuição, Metabolismo, Excreção e Toxicidade) e a **análise de vastas quantidades de literatura científica** [1] [2].

Ao invés de depender apenas de modelos de aprendizado de máquina (ML) treinados especificamente, o Prompt Engineering adapta modelos de propósito geral para o domínio científico, atuando como um "avaliador, colaborador e cientista" no processo de P&D [3].

## Referências
[1] Certara. Best Practices for AI Prompt Engineering in Life Sciences in 2025. Disponível em: [https://www.certara.com/blog/best-practices-for-ai-prompt-engineering-in-life-sciences/](https://www.certara.com/blog/best-practices-for-ai-prompt-engineering-in-life-sciences/)
[2] Othman, Z. K. et al. Advancing drug discovery and development through GPT models: a review on challenges, innovations and future prospects. *Intelligence-Based Medicine*, 2025.
[3] Zhang, H. et al. The evolving role of large language models in scientific innovation: Evaluator, collaborator, and scientist. *arXiv preprint arXiv:2507.11810*, 2025.

## Examples
```
A seguir, 5 exemplos de prompts concretos e acionáveis, focados em diferentes etapas da descoberta de medicamentos:

1.  **Previsão de Propriedades ADMET (ADMET Prediction)**
    \`\`\`
    Atue como um químico medicinal sênior. Sua tarefa é prever as propriedades ADMET para a molécula com a string SMILES: C1=CC=C(C=C1)C(C(=O)O)N.
    
    Forneça a resposta em formato JSON com as seguintes chaves:
    - 'Molecule_SMILES'
    - 'Toxicity_Prediction' (ex: 'Baixa', 'Média', 'Alta')
    - 'Solubility_Prediction' (ex: 'Alta', 'Baixa')
    - 'LogP_Value' (Valor numérico)
    - 'Rationale' (Breve explicação do porquê das previsões)
    \`\`\`

2.  **Geração de Moléculas *De Novo***
    \`\`\`
    Gere 5 strings SMILES para moléculas que atuam como inibidores seletivos do receptor de [Proteína Alvo, ex: EGFR].
    
    **Restrições:**
    - Peso molecular entre 350 e 450 Da.
    - LogP inferior a 3.5.
    - Deve conter um anel de piridina.
    
    Liste as 5 moléculas em um formato de lista numerada, sem texto adicional.
    \`\`\`

3.  **Análise de Literatura Científica (Extração de Dados)**
    \`\`\`
    Você é um assistente de pesquisa. Analise o seguinte resumo de artigo científico: [TEXTO DO RESUMO AQUI].
    
    **Tarefa:** Extraia todas as interações droga-droga mencionadas e liste-as em formato de tabela Markdown com as colunas 'Droga 1', 'Droga 2' e 'Efeito Observado' (ex: 'Potencialização', 'Inibição'). Se não houver interações, responda "Nenhuma interação droga-droga encontrada".
    \`\`\`

4.  **Otimização de *Leads* (Melhoria de Biodisponibilidade)**
    \`\`\`
    A molécula candidata com a string SMILES: [SMILES STRING AQUI] demonstrou baixa biodisponibilidade oral (F% < 10) em estudos pré-clínicos.
    
    **Tarefa:** Sugira três modificações estruturais distintas para aumentar a biodisponibilidade, focando em melhorar a solubilidade e a permeabilidade.
    
    Para cada sugestão, forneça:
    1. O racional químico para a modificação.
    2. A nova string SMILES da molécula modificada.
    \`\`\`

5.  **Validação de Alvo e Mecanismo de Ação**
    \`\`\`
    Explique o papel da proteína [Nome da Proteína, ex: JAK2] na patogênese da doença [Nome da Doença, ex: Mielofibrose].
    
    **Instruções:**
    - Use uma linguagem acessível, mas cientificamente precisa.
    - Mencione o mecanismo de ação e as vias de sinalização envolvidas.
    - Responda em 3 parágrafos concisos.
    \`\`\`
```

## Best Practices
As melhores práticas para "Drug Discovery Prompts" se baseiam na clareza, especificidade e na incorporação de conhecimento de domínio [1]:

*   **Definir a Persona e o Contexto:** Comece o prompt instruindo o LLM a assumir um papel específico (ex: "Atue como um químico medicinal sênior" ou "Você é um especialista em bioinformática").
*   **Estrutura do Prompt (CDTF):** Utilize os quatro componentes essenciais: **Contexto** (o papel e o objetivo), **Dados** (a string SMILES, o texto do artigo), **Tarefa** (a ação a ser executada) e **Formato** (JSON, tabela Markdown, lista numerada) [1].
*   **One-Shot/Few-Shot Prompting:** Fornecer um ou mais exemplos de entrada e saída desejada é crucial para tarefas técnicas, como a geração de SMILES ou a classificação de toxicidade.
*   **Restrição de Alucinação:** Inclua instruções explícitas para mitigar a "alucinação" (geração de informações falsas). Ex: "Use apenas informações de artigos revisados por pares publicados após 2023. Se a informação não estiver disponível, responda 'Não sei'."
*   **Uso de Formatos Estruturados:** Sempre que possível, solicite a saída em formatos estruturados (JSON, CSV, Tabela Markdown) para facilitar a análise e o processamento posterior.

## Use Cases
*   **Previsão de Propriedades (ADMET):** Prever rapidamente a toxicidade, solubilidade, permeabilidade e outras propriedades farmacocinéticas de milhares de moléculas candidatas.
*   **Geração *De Novo* de Moléculas:** Criar novas estruturas moleculares com base em um perfil de propriedades alvo (Target Product Profile - TPP) e restrições químicas específicas.
*   **Revisão e Síntese de Literatura:** Extrair informações específicas (ex: doses, resultados de ensaios clínicos, interações medicamentosas) de artigos científicos e patentes de forma automatizada.
*   **Otimização de *Leads*:** Sugerir modificações químicas para melhorar uma propriedade indesejada (ex: aumentar a potência, reduzir a toxicidade) de um composto líder.
*   **Síntese Retrossintética:** Propor rotas de síntese química viáveis para uma molécula alvo.

## Pitfalls
*   **Alucinação de Dados Científicos:** O risco mais grave. O LLM pode gerar strings SMILES inválidas, dados de toxicidade incorretos ou citar artigos inexistentes. **Solução:** Restrição rigorosa e verificação cruzada com bases de dados confiáveis.
*   **Falta de Especificidade Química:** Prompts vagos resultam em sugestões quimicamente irrelevantes ou inviáveis. **Solução:** Usar termos técnicos precisos (ex: "inibidor seletivo", "grupo funcional hidroxila", "LogP").
*   **Dependência Excessiva do LLM:** O LLM é uma ferramenta de auxílio, não um químico. Confiar cegamente nas sugestões sem validação experimental ou computacional pode levar a becos sem saída.
*   **Limitação de Contexto:** A capacidade de processar longas strings SMILES ou grandes conjuntos de dados pode ser limitada pelo tamanho da janela de contexto do modelo. **Solução:** Dividir tarefas complexas em subtarefas menores (Chain-of-Thought ou uso de agentes).

## URL
[https://www.certara.com/blog/best-practices-for-ai-prompt-engineering-in-life-sciences/](https://www.certara.com/blog/best-practices-for-ai-prompt-engineering-in-life-sciences/)
