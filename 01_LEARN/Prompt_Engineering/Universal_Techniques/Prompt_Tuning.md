# Prompt Tuning

## Description

O **Prompt Tuning** é uma técnica de *Parameter-Efficient Fine-Tuning* (PEFT) que adapta Large Language Models (LLMs) pré-treinados, mantendo seus parâmetros congelados. A adaptação é feita através da otimização de um pequeno conjunto de vetores contínuos treináveis, chamados **soft prompts**, que são anexados à entrada do modelo. Ao contrário do *fine-tuning* completo, que ajusta milhões ou bilhões de parâmetros, o Prompt Tuning ajusta apenas alguns milhares de parâmetros, reduzindo drasticamente os custos computacionais e o tempo de treinamento. O processo envolve a inicialização dos *soft prompts* (geralmente com valores aleatórios ou *embeddings* de palavras), a execução de um ciclo de treinamento (*forward* e *backward pass*) e a otimização desses vetores com uma função de perda até que atinjam o desempenho desejado para a tarefa específica. Essa abordagem permite a criação de prompts modulares e específicos para tarefas, facilitando a implantação eficiente e a adaptação a novos domínios.

## Statistics

- **Eficiência Comprovada:** O Prompt Tuning original (Lester et al., 2021) demonstrou desempenho comparável ao *full fine-tuning* em modelos T5-XXL (11B parâmetros) em tarefas SuperGLUE, ajustando apenas 0,01% dos parâmetros.
- **P-Tuning v2:** Variação que alcança desempenho comparável ao *full fine-tuning* em diferentes escalas de modelo (330M a 10B de parâmetros), demonstrando forte eficiência de parâmetros.
- **Custo:** Redução drástica no consumo de GPU e tempo de treinamento em comparação com o *full fine-tuning*.
- **Citação Primária:** Lester, B., Al-Rfou, R., & Constant, N. (2021). *The Power of Scale for Parameter-Efficient Prompt Tuning*. arXiv:2104.08691.

## Features

- **Eficiência de Parâmetros:** Atualiza apenas um pequeno subconjunto de vetores contínuos (soft prompts), mantendo o LLM base congelado.
- **Adaptação Modular:** Os prompts são específicos para tarefas, permitindo que o modelo base seja reutilizado para múltiplas tarefas com diferentes *soft prompts*.
- **Soft Prompts:** Vetores contínuos no espaço de *embedding*, irrestritos por vocabulário, o que permite uma otimização mais flexível e eficaz do que os *hard prompts* (texto natural).
- **Redução de Custos:** Diminui significativamente os requisitos de memória e computação para treinamento e implantação.
- **Flexibilidade de Framework:** Suporta mecanismos de transferência de conhecimento e composição, sendo a base para variações como P-Tuning, Prefix Tuning e Decomposed Prompt Tuning.

## Use Cases

- **Adaptação de Domínio:** Ajustar um LLM genérico para domínios específicos (ex: medicina, direito, finanças) com um conjunto limitado de dados específicos.
- **Geração de Conteúdo Específico:** Treinar o modelo para gerar tipos específicos de texto, como descrições de produtos para e-commerce, resumos de notícias ou respostas a perguntas em um formato padronizado.
- **Classificação de Texto:** Tarefas como análise de sentimento, classificação de documentos e moderação de conteúdo.
- **Cenários de Baixos Recursos (*Few-Shot Learning*):** Ideal para situações onde há poucos dados rotulados disponíveis, pois a otimização dos *soft prompts* é mais eficiente em extrair conhecimento do modelo pré-treinado.
- **Implantação em Dispositivos:** Sua eficiência de memória e computação o torna adequado para implantação em ambientes com recursos limitados.

## Integration

O Prompt Tuning é implementado através de bibliotecas PEFT, como a do Hugging Face. As melhores práticas incluem:

**1. Exemplo de Implementação (Python/PEFT):**
A configuração do Prompt Tuning é feita definindo o número de tokens virtuais e o tipo de tarefa:
```python
from peft import PromptTuningConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuração do Prompt Tuning
config = PromptTuningConfig(
    num_virtual_tokens=50,  # 20-50 para classificação, 50-100+ para geração
    task_type="CAUSAL_LM",  # Ex: Geração de Texto
    prompt_tuning_init="RANDOM" # Inicialização
)

# Carregar modelo e adicionar capacidade de Prompt Tuning
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = get_peft_model(model, config)
```

**2. Melhores Práticas:**
- **Tamanho do Prompt:** O número de tokens virtuais (*num_virtual_tokens*) é um hiperparâmetro crucial. Deve ser ajustado: 20-50 para tarefas de classificação e 50-100 ou mais para tarefas complexas de geração de texto.
- **Verbalizers:** Para tarefas de classificação, o uso de *verbalizers* (mapeamento da saída do modelo para rótulos de classe) é essencial.
- **Sensibilidade a Hiperparâmetros:** O Prompt Tuning é sensível à taxa de aprendizado e ao método de inicialização.
- **Escala do Modelo:** A eficácia do Prompt Tuning é maior em modelos de grande escala (acima de 10B de parâmetros), onde pode igualar o desempenho do *full fine-tuning*.

## URL

https://arxiv.org/html/2507.06085v2