# Models-Vote Prompting (MVP)

## Description

O **Models-Vote Prompting (MVP)** é uma técnica avançada de *prompt engineering* baseada em **ensemble** e **Few-Shot Learning (FSL)**, projetada especificamente para melhorar a precisão na **Identificação e Classificação de Doenças Raras** a partir de notas clínicas (EHRs). A metodologia central envolve submeter a mesma tarefa a múltiplos Modelos de Linguagem Grande (LLMs) e, em seguida, realizar uma **votação por maioria** sobre as respostas geradas para determinar o diagnóstico final. Esta abordagem é particularmente eficaz em cenários de FSL, onde a escassez de dados de treinamento para doenças raras é um desafio significativo. O MVP também incorpora o uso de formatos de saída estruturados, como **JSON**, para facilitar a avaliação automatizada e a integração com sistemas clínicos.

## Statistics

O MVP demonstrou consistentemente o melhor desempenho geral em tarefas de Identificação e Classificação de Doenças Raras, superando modelos individuais e o Self-Consistency Prompting (SC).

**Melhores Métricas (F-score) em Contexto de 64 palavras:**
- **Identificação de Doença Rara (F-score):** MVP alcançou **0.69**, superando o melhor modelo individual (Llama 2: 0.58) e o SC (Llama 2 + SC: 0.49).
- **Classificação de Doença Rara (F-score):** MVP alcançou **0.69**, superando o melhor modelo individual (Vicuna: 0.67) e empatando com o SC (Llama 2 + SC: 0.70).

**Modelos LLM Utilizados no Ensemble:** Llama 2, MedAlpaca, Stable Platypus 2, e Vicuna.
**Citação:** Oniani, D. et al. (2024). *Large Language Models Vote: Prompting for Rare Disease Identification*. arXiv:2308.12890v3.

## Features

- **Ensemble Prompting:** Utiliza um conjunto de modelos (e.g., Llama 2, MedAlpaca, Vicuna) para aumentar a robustez e reduzir o viés de um único modelo.
- **Votação por Maioria:** A decisão final é baseada no consenso dos modelos, superando o desempenho de qualquer modelo individual.
- **Few-Shot Learning (FSL):** Otimizado para tarefas com dados limitados, como o diagnóstico de doenças raras.
- **CoT-Augmented:** Pode ser combinado com técnicas como Chain-of-Thought (CoT) para melhorar o raciocínio e a explicabilidade.
- **JSON-Augmented:** Utiliza JSON para garantir um formato de saída analisável e facilitar a avaliação automatizada.

## Use Cases

- **Diagnóstico Diferencial:** Auxiliar médicos na triagem e diagnóstico diferencial de pacientes com sintomas atípicos que sugerem doenças raras.
- **Análise de EHRs:** Extração e classificação de menções de doenças raras a partir de grandes volumes de notas clínicas não estruturadas.
- **Pesquisa em FSL:** Aplicação em qualquer domínio médico ou biológico onde a anotação de dados é cara e a disponibilidade de exemplos é limitada.

## Integration

O MVP utiliza um template de prompt que combina a descrição da tarefa, um exemplo de Chain-of-Thought (CoT) para instrução *in-context* e a pergunta real baseada na nota clínica. A saída é estruturada em JSON para facilitar a votação e a análise.

**Exemplo de Estrutura de Saída JSON Sugerida:**
```json
{
  "disease_identified": [
    "Babesiosis",
    "Giant Cell Arteritis",
    "Graft Versus Host Disease",
    "Cryptogenic Organizing Pneumonia"
  ],
  "task_disease": "None"
}
```
**Guia de Integração:**
1.  Defina um conjunto de LLMs (e.g., 4 modelos).
2.  Crie um prompt CoT-Augmented com o formato de saída JSON desejado.
3.  Submeta a nota clínica ao prompt em todos os LLMs.
4.  Colete as saídas JSON e realize uma votação por maioria no campo `"task_disease"` ou nos elementos da lista `"disease_identified"`.
5.  A doença com mais votos é o diagnóstico final do ensemble.

## URL

https://arxiv.org/abs/2308.12890