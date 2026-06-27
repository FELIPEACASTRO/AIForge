# AutoMedPrompt e Prompt de Raciocínio Clínico Estruturado

## Description

AutoMedPrompt é uma nova estrutura que utiliza 'gradientes textuais' (TextGrad) para otimizar prompts de sistema em Large Language Models (LLMs) de propósito geral, a fim de elicitar raciocínio clinicamente relevante. Em vez de depender de fine-tuning ou métodos de prompting como Chain-of-Thought (CoT) que podem ser inadequados para subespecialidades, o AutoMedPrompt ajusta o prompt do sistema para melhorar a capacidade de raciocínio médico do LLM, superando modelos proprietários de ponta.

O Prompt de Raciocínio Clínico Estruturado é uma técnica de engenharia de prompt de dois passos que visa melhorar a capacidade de diagnóstico de LLMs, forçando o modelo a seguir uma metodologia de raciocínio clínico padronizada. O primeiro passo envolve a sumarização e categorização sistemática das informações clínicas (história, sintomas, exames) em um formato estruturado. O segundo passo utiliza essa informação estruturada para realizar o raciocínio diagnóstico, imitando o processo de um médico.

## Statistics

**AutoMedPrompt:** Alcançou o novo Estado da Arte (SOTA) no benchmark PubMedQA com uma precisão de 82.6%. Superou o desempenho de modelos proprietários como GPT-4, Claude 3 Opus e Med-PaLM 2. Também obteve precisão de 77.7% no MedQA e 63.8% no NephSAP (subespecialidade de nefrologia) usando o Llama 3 de código aberto.

**Prompt de Raciocínio Clínico Estruturado:** Aumentou significativamente a precisão do diagnóstico primário para 60.6% (vs. 56.5% da linha de base) e a precisão dos três principais diagnósticos para 70.5% (vs. 66.5% da linha de base) em 322 casos de quiz de diagnóstico (_Radiology’s Diagnosis Please_). O estudo utilizou o modelo Claude 3.5 Sonnet.

## Features

**AutoMedPrompt:** Otimização de prompt de sistema; Utiliza gradientes textuais (TextGrad); Não requer fine-tuning extensivo; Melhora o raciocínio médico em LLMs de propósito geral; Focado em benchmarks de QA médica.

**Prompt de Raciocínio Clínico Estruturado:** Abordagem de dois passos (Sumarização + Raciocínio); Imita o fluxo de trabalho clínico; Reduz a variabilidade de desempenho; Melhora a capacidade de diagnóstico em casos complexos.

## Use Cases

**AutoMedPrompt:** Melhoria da precisão diagnóstica em sistemas de suporte à decisão clínica; Otimização de LLMs de código aberto para tarefas médicas específicas; Pesquisa e desenvolvimento de prompts avançados para educação médica e testes de conhecimento.

**Prompt de Raciocínio Clínico Estruturado:** Treinamento de LLMs para simulação de raciocínio clínico; Sistemas de triagem e diagnóstico diferencial em ambientes clínicos; Ferramenta de apoio à decisão para médicos residentes e estudantes.

## Integration

**AutoMedPrompt:** A técnica envolve a otimização do prompt do sistema (instrução inicial) do LLM. Embora o prompt final otimizado não esteja explicitamente detalhado no resumo, o princípio é: 'Otimizar o prompt do sistema para elicitar o raciocínio médico mais relevante, usando o TextGrad para guiar a otimização'.

**Prompt de Raciocínio Clínico Estruturado:**
*   **Passo 1 (Sumarização):** 'Você é um Radiologista Diagnóstico experiente. Sua tarefa é resumir o seguinte caso clínico, categorizando a informação em: informações do paciente, histórico da doença atual, histórico médico, sintomas, achados do exame físico, resultados laboratoriais, achados de imagem, etc.'
*   **Passo 2 (Raciocínio):** 'Como médico, utilize o resumo estruturado para me guiar através do processo de diagnóstico diferencial até o diagnóstico mais provável e os próximos dois diagnósticos diferenciais mais prováveis, passo a passo.'

## URL

https://arxiv.org/abs/2502.15944; https://pmc.ncbi.nlm.nih.gov/articles/PMC11953165/