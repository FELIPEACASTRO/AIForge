# Hugging Face Transformers

## Description

Transformers é a estrutura de definição de modelos de código aberto da Hugging Face para modelos de aprendizado de máquina de última geração em texto, visão, áudio e multimodais. Sua proposta de valor única é centralizar a definição de modelos para garantir consistência e interoperabilidade em todo o ecossistema de IA, suportando a maioria das estruturas de treinamento e mecanismos de inferência. É a espinha dorsal para a geração de texto moderna baseada em LLMs.

## Statistics

GitHub Stars: 152k; GitHub Forks: 31.1k; Modelos no Hub: Mais de 1 milhão de checkpoints de modelos.

## Features

Suporte a modelos de última geração para PNL, visão computacional, áudio e multimodais; API unificada para mais de 100 arquiteturas de modelos; Suporte para treinamento distribuído e inferência de alto desempenho (via TGI); Integração com o Hugging Face Hub para compartilhamento e reutilização de modelos.

## Use Cases

Geração de texto (LLMs); Tradução automática; Resposta a perguntas; Classificação de texto e reconhecimento de entidade nomeada (NER); Tarefas multimodais (texto-imagem, texto-áudio).

## Integration

A integração é feita via `pip install transformers` e utilizando a API de pipeline para tarefas rápidas ou a API de modelo/tokenizador para controle mais granular.

**Exemplo de Geração de Texto (LLM):**
```python
from transformers import pipeline

# Usando um pipeline para geração de texto
generator = pipeline("text-generation", model="gpt2")
result = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=1)
print(result[0]['generated_text'])
# Saída: Hello, I'm a language model, and I'm here to help you with your writing. I'm a language model, and I'm here to help you with your writing.
```

## URL

https://github.com/huggingface/transformers