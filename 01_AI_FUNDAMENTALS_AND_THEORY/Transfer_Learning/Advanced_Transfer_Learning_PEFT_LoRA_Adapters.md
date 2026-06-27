# Transfer Learning Avançado: Estratégias PEFT (LoRA, Adapters)

## Description

O Transfer Learning Avançado, focado em técnicas de **Parameter-Efficient Fine-Tuning (PEFT)**, é uma metodologia crucial para a adaptação de Modelos de Linguagem Grandes (LLMs) e outros modelos pré-treinados a tarefas específicas. Ao invés de realizar o fine-tuning completo de todos os bilhões de parâmetros do modelo base, o PEFT introduz um pequeno número de parâmetros treináveis (os 'adaptadores') enquanto mantém a vasta maioria dos pesos originais **congelados**. Isso permite que o modelo retenha seu conhecimento geral, ao mesmo tempo que aprende nuances específicas da nova tarefa, resultando em uma adaptação rápida, eficiente e de alto desempenho.

## Statistics

As técnicas PEFT oferecem ganhos substanciais em eficiência. Por exemplo, o **LoRA (Low-Rank Adaptation)** pode reduzir o número de parâmetros treináveis em mais de **99%** e o uso de memória de GPU em até **80%** em comparação com o fine-tuning completo. Isso permite o treinamento de modelos de dezenas de bilhões de parâmetros em hardware de consumo (GPUs de 16GB ou 24GB), democratizando o acesso ao fine-tuning de LLMs.

## Features

As principais técnicas PEFT incluem:\n\n*   **Adapter Layers:** Injeção de pequenas camadas *bottleneck* na arquitetura do modelo. Apenas os pesos dessas camadas são treinados.\n*   **LoRA (Low-Rank Adaptation):** Decomposição de matrizes de peso em duas matrizes de baixo rank. Apenas os pesos dessas matrizes de baixo rank são otimizados.\n*   **Prefix-Tuning:** Otimização de um pequeno conjunto de vetores contínuos ('prefixos') que são concatenados à sequência de entrada, guiando o modelo para a tarefa.\n*   **Congelamento de Pesos:** O princípio central é manter os pesos do modelo pré-treinado (a 'base de conhecimento') fixos, preservando a estabilidade e o conhecimento geral.

## Use Cases

O Transfer Learning Avançado é aplicado em:\n\n*   **Adaptação de LLMs:** Personalização de modelos como Llama, Mistral ou T5 para tarefas específicas de domínio (e.g., jurídico, médico, financeiro) com recursos limitados.\n*   **Multi-Task Learning:** Treinamento de um único modelo base para múltiplas tarefas, onde cada tarefa possui seu próprio conjunto de adaptadores PEFT, permitindo o *switching* rápido entre tarefas.\n*   **Implantação em Edge Devices:** Criação de modelos menores e mais eficientes para implantação em dispositivos com restrições de memória e computação.\n*   **Pesquisa Rápida:** Prototipagem e experimentação acelerada de novas tarefas e conjuntos de dados.

## Integration

A integração é amplamente facilitada pela biblioteca **PEFT (Parameter-Efficient Fine-Tuning)** da Hugging Face, que abstrai a complexidade das diferentes técnicas. O fluxo de trabalho típico envolve:\n\n1.  Instalação: `pip install peft transformers`\n2.  Carregamento do Modelo Base:\n    ```python\n    from transformers import AutoModelForCausalLM\n    model = AutoModelForCausalLM.from_pretrained(\"meta-llama/Llama-2-7b-hf\")\n    ```\n3.  Configuração do Adaptador (Exemplo LoRA):\n    ```python\n    from peft import LoraConfig, get_peft_model\n    config = LoraConfig(\n        r=8, \n        lora_alpha=16, \n        target_modules=[\"q_proj\", \"v_proj\"], \n        lora_dropout=0.05,\n        bias=\"none\", \n        task_type=\"CAUSAL_LM\"\n    )\n    peft_model = get_peft_model(model, config)\n    peft_model.print_trainable_parameters()\n    # Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06220\n    ```\n4.  Treinamento: O `peft_model` é treinado como um modelo `transformers` padrão, mas apenas os parâmetros LoRA são atualizados.

## URL

https://huggingface.co/docs/peft/en/index