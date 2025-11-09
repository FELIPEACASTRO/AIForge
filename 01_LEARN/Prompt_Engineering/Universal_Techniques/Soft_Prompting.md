# Soft Prompting

## Description

**Soft Prompting** (também conhecidos como *learned prompts*, *continuous prompts* ou *prompt embeddings*) é uma técnica de **Parameter-Efficient Fine-Tuning (PEFT)** que adapta Large Language Models (LLMs) pré-treinados para tarefas específicas sem a necessidade de treinar todos os seus parâmetros. Ao contrário dos **Hard Prompts** (prompts textuais discretos e criados manualmente), os Soft Prompts são **tensores aprendíveis** (vetores de *tokens virtuais*) que são concatenados com os *embeddings* de entrada do modelo e otimizados diretamente em um conjunto de dados de treinamento.

Essa abordagem permite que o modelo permaneça congelado, enquanto apenas um pequeno conjunto de parâmetros do prompt é treinado, resultando em uma adaptação significativamente mais eficiente em termos de tempo e custo computacional. O principal ponto negativo é que esses *tokens virtuais* não são legíveis por humanos.

## Statistics

*   **Eficiência de Parâmetros:** Prefix Tuning demonstrou desempenho comparável ao *fine-tuning* completo, mas com **1000x menos parâmetros** treináveis.
*   **Escalabilidade:** O desempenho do Prompt Tuning se **escala** com o aumento do tamanho do modelo, equiparando-se ao *fine-tuning* tradicional em modelos maiores.
*   **Pesquisa Recente (2024):** O trabalho "Nemesis: Normalizing the Soft-prompt Vectors of Vision-Language Models" (ICLR 2024) investigou o **Efeito de Baixa Norma (*Low-Norm Effect*)** em *soft-prompts* para Vision-Language Models (VLMs), sugerindo que a redução da norma de certos prompts aprendidos pode **melhorar o desempenho** dos VLMs.

## Features

O Soft Prompting engloba várias sub-técnicas de PEFT, cada uma com variações na forma como os *embeddings* do prompt são inseridos e otimizados:

1.  **Prompt Tuning:** Adiciona *tokens* de prompt aprendíveis apenas aos *embeddings* de entrada. Bom para classificação de texto e escalável com o tamanho do modelo.
2.  **Prefix Tuning:** Insere parâmetros de prefixo otimizáveis em **todas** as camadas do modelo. Ideal para Geração de Linguagem Natural (NLG).
3.  **P-Tuning:** Utiliza um codificador de prompt (como LSTM) e permite que os *tokens* de prompt sejam inseridos em **qualquer lugar** na sequência de entrada. Projetado para Compreensão de Linguagem Natural (NLU).
4.  **Multitask Prompt Tuning (MPT):** Aprende um único prompt para múltiplos tipos de tarefas, permitindo *transfer learning* eficiente.
5.  **Context-Aware Prompt Tuning (CPT):** Refina apenas *embeddings* de *tokens* de contexto específicos para aprimorar a classificação *few-shot*.

## Use Cases

*   **Adaptação de Modelos:** Adaptação eficiente de LLMs pré-treinados para uma ampla variedade de tarefas *downstream* (ex: classificação, geração, NLU) sem a necessidade de *fine-tuning* completo.
*   **Ambientes de Baixa Quantidade de Dados:** Prefix Tuning é particularmente eficaz em cenários com poucos dados (*low-data settings*).
*   **Transfer Learning:** Multitask Prompt Tuning permite o *transfer learning* de um único prompt aprendido para múltiplas tarefas.
*   **Modelos Multimodais:** Pesquisas recentes aplicam Soft Prompting em Vision-Language Models (VLMs) como o CLIP para adaptação de tarefas.

## Integration

Como os Soft Prompts são tensores aprendíveis e não texto legível por humanos, não há "exemplos de prompt" no sentido tradicional de texto de entrada. A integração se dá através da implementação de uma das sub-técnicas (PEFT).

**Melhores Práticas (PEFT):**
*   **Escolha da Técnica:** A escolha da sub-técnica depende da tarefa: **Prompt Tuning** para classificação, **Prefix Tuning** para NLG e **P-Tuning** para NLU.
*   **Implementação:** Utilizar bibliotecas como **🤗 PEFT (Parameter-Efficient Fine-Tuning)** da Hugging Face, que fornece implementações prontas.
*   **Otimização:** A otimização é feita via *backpropagation* no conjunto de dados de treinamento, atualizando apenas os parâmetros do prompt enquanto o modelo base permanece congelado.

## URL

https://huggingface.co/docs/peft/en/conceptual_guides/prompting